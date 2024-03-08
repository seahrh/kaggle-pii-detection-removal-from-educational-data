#!/usr/bin/env bash

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node 2 \
  --nnodes 1 \
  -m mylib.train \
  --conf "train.ini" \
  --task "ner"

wait
