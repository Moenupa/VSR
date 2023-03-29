#!/usr/bin/env bash

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=10.22.5.52 \
    --master_port=29998 \
    mmediting/tools/train.py \
    vsr/src/model/config_edvr.py \
    --seed 0 \
    --launcher pytorch ${@:3}