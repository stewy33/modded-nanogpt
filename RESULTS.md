# Experiment Results — lowest fineweb val_loss in 5-min single-GPU

Environment: 1× H100 80GB (only 1 GPU available, runs are sequential).
Budget: `total_train_minutes=5.0`, `--nproc_per_node=1`. Metric = final `val_loss`.

| name | key changes vs baseline | final val_loss | steps | step_avg | notes |
|------|------------------------|---------------:|------:|---------:|-------|
| baseline | bs=512, dbs=64, lr=3.6e-3, wd=0, warmdown=0.28 | 4.1955 | 288 | 1081ms | reference; 8 grad-accum microbatches |
| **arch_l8** | **n_layer=8, bs64** | **3.8252** | **2670** | **113ms** | **BEST** |
| bs128 | bs=128, dbs=128, no grad-accum (~4x steps) | 3.8619 | 1098 | 276ms | big win over baseline |
| bs64 | bs=64, dbs=64, no grad-accum (~8x steps) | 3.8621 | 1956 | 154ms | ties bs128; batch benefit saturated |
| bs64_lr0050 | bs64, lr=5.0e-3 | 3.8694 | 1943 | 155ms | ~tie; LR insensitive |
| bs64_lr0024 | bs64, lr=2.4e-3 | ~4.18@180s | (stopped) | 153ms | tracked lr0036; LR insensitive, stopped early |
