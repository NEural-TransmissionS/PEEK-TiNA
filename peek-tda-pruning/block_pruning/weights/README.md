# Base model weights (not committed)

Model weights are kept out of git. The code loads these 8 checkpoints from this folder
(`block_pruning/weights/<dataset>_<size>.pt`): the block ranking (`run_experiment.py`)
and every synthetic-shapes channel-pruning run start from them. Copy your local copies
here and check them with `sha256sum -c SHA256SUMS` (run in this folder).

They are YOLO26n/s trained from scratch for 40 epochs on the synthetic shape datasets
(`../make_shapes.py`); full training args: `../../results/base_models/shapes_<name>.yaml`.
The WSD base models are not here either — see `../../scripts/train_wsd_baselines.slurm`.

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `circles_n.pt` | 5385214 | `9b7a623a2e3011847e35eb34aee25969f3ee76e134dba502d49ffc53abe2722f` |
| `circles_s.pt` | 20309630 | `0fc39a4692f93d92e417491d4ea578bed4a94ffe17c1a6ba3d80450a75b9961b` |
| `mixed_n.pt` | 5386750 | `b11a8177079cc742db07bac89ca13b26eefdaf2e8daa80e0947a91fc625ba20e` |
| `mixed_s.pt` | 20312766 | `847c1221e976cf55a953214f1d570b3f3da60da4d15e05123b96f85948211d3c` |
| `squares_n.pt` | 5385214 | `bc1e449e1894deaaad5011355b7983f2fff584ef8391b59cb5964bc8dfc03f5b` |
| `squares_s.pt` | 20309630 | `4fc9886d88f17fed58e131840272023c9322f49d55959baac10c40b510dc885f` |
| `triangles_n.pt` | 5385214 | `981cf1c019188665c28d1dd274492e9f7a2eca5bdce13f6fcf0362e1abb6add8` |
| `triangles_s.pt` | 20309630 | `a398438cbf8c17dab76306a20c40b3c334f21c9d6035a871e4260d8e7de83192` |
