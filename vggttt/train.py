# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import logging
import os
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import hydra
import lightning as L
import torch
from aim import Run
from aim.pytorch_lightning import AimLogger
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, TQDMProgressBar
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from lightning_fabric.utilities.seed import pl_worker_init_function
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from vggttt.data.augment import get_augmentations
from vggttt.data.sampler import WeightedBatchSampler

_logger = logging.getLogger(__name__)


def load_model_only(ckpt: Path, overrides: ListConfig | None = None) -> torch.nn.Module:
    _logger.info("Loading model from %s with overrides %s", ckpt, overrides)
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    model_config = OmegaConf.create(ckpt_data["hyper_parameters"]["cfg"]["model"])
    model_config.merge_with_dotlist(list(overrides) if overrides else [])

    state_dict = {k.removeprefix("model."): v for k, v in ckpt_data["state_dict"].items()}
    model = instantiate(model_config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    _logger.warning("Missing keys: %s, \nUnexpected keys: %s", missing_keys, unexpected_keys)
    return model


def load_model(
    ckpt: Path | None,
    overrides: Sequence[str] = (),
    cfg: OmegaConf | None = None,
    use_ckpt_cfg: bool = True,
    load_weights: bool = True,
):
    """Load a model from a checkpoint.

    Passed-in config is updated (in order!) with:
    - the model-specific parts of the config from the checkpoint (if `use_ckpt_cfg` is True)
    - the passed `overrides`.
    """
    from hydra import compose, initialize
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if cfg is None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name="config", overrides=overrides)

    # Potentially update config from checkpoint
    ckpt_data = None
    if ckpt is not None:
        if ckpt.is_file():
            if ckpt.suffix == ".ckpt":
                ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=False)
                state_dict_key = "state_dict"
            else:  # .pth
                ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=False)
                state_dict_key = None
        elif ckpt.is_dir():
            # Deepspeed ZeRO Stage 1/2 checkpoint
            ckpt_data = torch.load(ckpt / "checkpoint" / "mp_rank_00_model_states.pt", map_location="cpu", mmap=True)
            state_dict_key = "module"
        else:
            raise ValueError(f"Invalid checkpoint path: {ckpt}")

        OmegaConf.set_struct(cfg, False)  # Allow arbitrary updates from stored config
        if use_ckpt_cfg and state_dict_key is not None:
            _logger.info("Loading model config from checkpoint")
            ckpt_model_cfg = ckpt_data["hyper_parameters"]["cfg"]["model"]
            if ckpt_model_cfg is not None:
                cfg.model.merge_with(ckpt_model_cfg)

    # Initialize trainer
    lightning_module = cfg["trainer_class"]
    model = instantiate(lightning_module, cfg=cfg)

    # Load weights into the model
    if ckpt is not None and load_weights:
        model_weights = ckpt_data[state_dict_key] if state_dict_key is not None else ckpt_data
        missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=False)
        _logger.warning("Missing keys: %s, \nUnexpected keys: %s", missing_keys, unexpected_keys)
    del ckpt_data
    return model, cfg


def get_dataset(
    cfg: OmegaConf,
    split: Literal["train", "test", "val"],
    dp_world_size: int,
    dp_rank: int,
    **kwargs,
) -> tuple[ConcatDataset, Sampler]:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    img_transform = get_augmentations(split)

    dataset_kwargs = dict(cfg.data.kwargs)
    dataset_kwargs.update(**kwargs)
    _logger.info("%s split data kwargs: %s", split, dataset_kwargs)

    all_datasets = []
    config_dir = Path(__file__).parent / "config" / "data"
    for dataset_name in cfg.data.datasets[split]:
        dataset_config_path = config_dir / f"{dataset_name}.yaml"
        if not dataset_config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")

        dataset_cfg = OmegaConf.load(dataset_config_path)
        cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        OmegaConf.set_struct(cfg_copy, False)
        merged_cfg = OmegaConf.merge(cfg_copy, dataset_cfg)

        try:
            OmegaConf.resolve(merged_cfg)
            dataset_conf = merged_cfg[dataset_name]
        except Exception as e:
            _logger.warning("Failed to resolve interpolations for dataset %s: %s", dataset_name, e)
            dataset_conf = dataset_cfg[dataset_name]

        dataset_split_conf = dataset_conf[split]

        # Instantiate the dataset
        start = time.perf_counter()
        dataset = instantiate(dataset_split_conf)(augment_func=img_transform, **dataset_kwargs)
        _logger.info(
            "Initialized dataset '%s' for split '%s' in %s seconds",
            dataset_name,
            split,
            f"{time.perf_counter() - start:.2f}",
        )
        all_datasets.append(dataset)

    all_datasets = ConcatDataset(all_datasets)

    # Sample from each dataset equally.
    sampler = WeightedBatchSampler(
        all_datasets,
        **cfg.data.sampler,
        shuffle=split == "train",
        num_batches=cfg.data.num_batches_per_epoch.get(split, None),
        world_rank=dp_rank,
        world_size=dp_world_size,
    )
    return all_datasets, sampler


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rank_zero_info(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.dry_run:
        _logger.info("Dry run requested. Exiting...")
        return

    torch.set_printoptions(sci_mode=False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # Variable size tensors

    run_hash = cfg.run_hash

    continue_ckpt = None
    use_ckpt_cfg = False
    ckpt_path = None

    # Continuing a training run
    if run_hash:
        continue_ckpt = "last"

    # Continuing from a provided checkpoint
    if cfg.ckpt_path:
        continue_ckpt = ckpt_path = Path(cfg.ckpt_path)
        use_ckpt_cfg = True

    # Just initialize model weights
    if cfg.just_weights_path and not continue_ckpt:
        ckpt_path = Path(cfg.just_weights_path)

    logger = AimLogger(
        repo=cfg.log_dir,
        experiment=cfg.experiment_name,
        run_name=cfg.run_name,
        capture_terminal_logs=True,
        log_system_params=True,
        run_hash=run_hash,
    )

    # Newly created run
    # - if rank 0, a run hash is created inside the logger and assigned
    # - if rank != 0, it will be broadcasted from rank 0 in the `ModelCheckpoint` constructor
    if isinstance(logger.experiment, Run) and not run_hash:
        run_hash = logger.version

    # NOTE: This is actually required to be set here explicitly for continuing distributed trainings with specifying
    # `ckpt_path="last"` in `trainer.fit`.
    # This is since otherwise
    #   - `ModelCheckpoint` tries to recover the checkpoint directory from the logger when `dirpath=None`.
    #   - Some loggers (e.g., MLFlow) only expose the required attributes (e.g, `version`) on rank 0.
    #   - Last checkpoint is not found for non-rank 0 workers.
    #   - Best previous metric are not recovered on non-rank 0 workers.
    #   - Rank 0 and non-rank 0 get out of sync and try to save different checkpoints.
    #   - Training gets stuck due to ranks waiting on each other.
    run_log_dir = Path(cfg.log_dir) / "checkpoints" / run_hash if run_hash else None

    world_size = max(1, int(os.environ.get("WORLD_SIZE", 1)))
    global_rank = int(os.environ.get("RANK", 0))
    sp_world_size = cfg.get("sequence_parallel_size", 1)

    dp_world_size = world_size // sp_world_size
    dp_rank = global_rank // sp_world_size
    sp_rank = global_rank % sp_world_size

    _logger.info(
        f"GLOBAL: {global_rank + 1}/{world_size}, DP: {dp_rank + 1}/{dp_world_size}, SP: {sp_rank + 1}/{sp_world_size}"
    )


    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        strategy = DDPStrategy(accelerator="cpu")
    else:
        process_group_backend = "cpu:gloo,cuda:nccl" if sp_world_size > 1 else None

        strategy = DDPStrategy(
            process_group_backend=process_group_backend,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=25,
            broadcast_buffers=False,  # Fine to disable if there is no batch norm.
            timeout=timedelta(minutes=10),
            static_graph=True,
        )

    L.seed_everything(cfg.seed)
    # Preprocessing should be the same when DP rank is the same
    worker_init_fun = partial(pl_worker_init_function, rank=dp_rank)
    ds, sampler = get_dataset(cfg, split="val", dp_world_size=dp_world_size, dp_rank=dp_rank, **cfg.data.test_overrides)
    val_dl = DataLoader(ds, **cfg.data.loader, batch_sampler=sampler, worker_init_fn=worker_init_fun)
    if hasattr(cfg.trainer, "overfit_batches"):
        _logger.info("Overfitting to %s batches", cfg.trainer.overfit_batches)
        train_ds, train_sampler = get_dataset(
            cfg, split="val", dp_world_size=dp_world_size, dp_rank=dp_rank, **cfg.data.test_overrides
        )
    else:
        train_ds, train_sampler = get_dataset(cfg, split="train", dp_world_size=dp_world_size, dp_rank=dp_rank)
    train_dl = DataLoader(train_ds, **cfg.data.loader, batch_sampler=train_sampler, worker_init_fn=worker_init_fun)

    if cfg.benchmark_dataloading:
        from tqdm import tqdm

        for _ in tqdm(train_dl):
            pass
        1 / 0

    callbacks = []
    if not hasattr(cfg.trainer, "overfit_batches"):
        checkpoint_callback = ModelCheckpoint(
            dirpath=run_log_dir,
            **cfg.ckpt_callback,
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
            verbose=True,
        )
        checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "_"
        callbacks.append(checkpoint_callback)

    trainer: L.Trainer = instantiate(cfg.trainer)(
        default_root_dir=run_log_dir,
        strategy=strategy,
        logger=logger,
        callbacks=[
            *callbacks,
            TQDMProgressBar(refresh_rate=10),
            LearningRateMonitor(logging_interval="step"),
            ModelSummary(max_depth=3),
        ],
        num_nodes=int(os.environ.get("NUM_NODES", 1)),
        use_distributed_sampler=False,  # We split data per scene for each rank manually to reduce memory usage.
        plugins=[LightningEnvironment()],  # Prevent SLURM environment from being loaded since we don't need it.,
    )

    model, _ = load_model(
        ckpt_path,
        cfg=cfg,
        use_ckpt_cfg=use_ckpt_cfg,
        overrides=HydraConfig.get().overrides.task,
        load_weights=False,
    )
    lightning_module = cfg["trainer_class"]
    model = instantiate(lightning_module, cfg=cfg)

    starting_from_weights_and_start_of_training = (
        cfg.run_initial_validation or cfg.just_weights_path or cfg.just_validate
    ) and continue_ckpt != "last"
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        if starting_from_weights_and_start_of_training:
            trainer.validate(model, val_dl, ckpt_path=ckpt_path)
            if cfg.just_validate:
                return

        trainer.fit(model, train_dl, val_dl, ckpt_path=continue_ckpt)


if __name__ == "__main__":
    main()
