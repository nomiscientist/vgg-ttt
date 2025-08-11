# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaOneWayNoncommercial

import pytest
import torch
import tqdm

from vggttt.nets.ttt import TTTOperator, fast_weight_swish_glu_weight_norm_mini_batch_apply

# set to highest precision
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def test_fast_weight_swish_glu(benchmark=False, benchmark_iters=1000):
    batch_size = 2

    # MLP weights
    d = 32
    d_hidden = 64
    w0 = torch.randn(batch_size, d, d_hidden, device=device)
    w1 = torch.randn(batch_size, d_hidden, d, device=device)
    w2 = torch.randn(batch_size, d, d_hidden, device=device)

    # Inputs
    seq_len = 12
    q = torch.randn(batch_size, seq_len, d, device=device)
    k = torch.randn(batch_size, seq_len, d, device=device)
    v = torch.randn(batch_size, seq_len, d, device=device)

    # Learning rates
    lr0 = torch.randn(batch_size, seq_len, 1, device=device)
    lr1 = torch.randn(batch_size, seq_len, 1, device=device)
    lr2 = torch.randn(batch_size, seq_len, 1, device=device)

    ttt_ua_order = [
        TTTOperator(start=0, end=seq_len // 2, compute_grad=True, update=True, apply=False),
        TTTOperator(start=0, end=seq_len, compute_grad=False, update=False, apply=True),
    ]

    # Test forward pass
    for _ in tqdm.trange(
        benchmark_iters if benchmark else 1,
        disable=not benchmark,
        desc="[test_fast_weight_swish_glu] auto_grad=False",
    ):
        o, _, _ = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, ttt_ua_order, muon_update_steps=5, auto_grad=False
        )
    assert o.shape == (batch_size, seq_len, d)

    for _ in tqdm.trange(
        benchmark_iters if benchmark else 1,
        disable=not benchmark,
        desc="[test_fast_weight_swish_glu] auto_grad=True",
    ):
        o_autodiff, _, _ = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, ttt_ua_order, muon_update_steps=5, auto_grad=True
        )
    assert o_autodiff.shape == (batch_size, seq_len, d)

    torch.testing.assert_close(o, o_autodiff)
    print("Test passed")


@pytest.mark.parametrize("num_prefix_tokens, num_suffix_tokens", [(0, 0), (2, 4)])
@pytest.mark.parametrize("short_conv_size_qkv", [(0, 0, 0), (3, 3, 3)])
def test_chunked(num_prefix_tokens, num_suffix_tokens, short_conv_size_qkv):
    from vggttt.nets.ttt_attention import FastWeightAttention

    layer = FastWeightAttention(
        dim=16,
        num_heads=2,
        muon_update_steps=2,
        short_conv_size_qkv=short_conv_size_qkv,
    )
    patch_h = 3
    patch_w = 4
    num_imgs = 2
    seq_len = num_imgs * patch_h * patch_w + num_prefix_tokens + num_suffix_tokens
    inp = torch.randn(2, seq_len, 16)

    ops = [
        TTTOperator(start=0, end=seq_len, compute_grad=True, update=True, apply=False),
        TTTOperator(start=0, end=seq_len, compute_grad=True, update=True, apply=True),
    ]

    x = layer(
        inp.clone(),
        num_prefix_tokens=num_prefix_tokens,
        num_suffix_tokens=num_suffix_tokens,
        info={"ttt_op_order": ops},
        patch_h=patch_h,
        patch_w=patch_w,
    )
    x_chunked = layer(
        inp.clone(),
        num_prefix_tokens=num_prefix_tokens,
        num_suffix_tokens=num_suffix_tokens,
        info={"ttt_op_order": ops},
        patch_h=patch_h,
        patch_w=patch_w,
        chunk_size=4,
    )

    assert x.shape == (2, seq_len, 16)
    assert torch.allclose(x, x_chunked)
