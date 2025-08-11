## Evaluation

### Prelimary
To setup data for evaluation, follow the instructions in Pi3. Specifically, see [here](https://github.com/yyfz/Pi3/tree/evaluation#dataset-preparation).
Follow the instructions [here](https://nianticlabs.github.io/ace/wayspots.html) to download the Wayspots dataset.

Set paths to the dataset directories in [corresponding config file](./config/cluster/local.yaml).


Install evaluation dependencies
```
pip install .[evaluation]
```

Then you can run evaluation to reproduce the results reported in the paper:

```bash
METHOD="model=vggttt"
OUTPUT_DIR="/path/to/output/dir"
```

Alternatively, use `METHOD="model=vggt"`

### Pointmap evaluation
To reproduce Table 1,

```bash
python vggttt/evaluation/pointmaps/eval.py $METHOD data=dtu output_dir=$OUTPUT_DIR
```
`data` can be any of `dtu` `eth3d` `nrgbd_sparse` `nrgbd_dense` `7scenes_sparse` `7scenes_dense`.


### Visual localization
For Table 5,

```bash
python vggttt/evaluation/visloc/eval.py $METHOD output_dir=$OUTPUT_DIR data=7scenes_visloc
```
and `data=wayspots`.


### Scalability benchmark

```bash
torchrun --nproc_per_node 4 vggttt/evaluation/pointmaps/eval.py $METHOD output_dir=$OUTPUT_DIR data=7scenes_strided data.additional_views=100
```
`data.additional_views=x` with `x in {100, 500, 1000}` reproduces results shown in Figure 4. `data=nrgbd_strided` is the other option. 

In the paper we scale the number of additional "support" views to up to 2,000 (Table 4) using distributed inference.
