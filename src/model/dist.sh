#!/usr/bin/env bash

CONFIG="C:/vsr/vsr/src/model/config_edvr.py"
GPUS=1
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"10.22.5.52"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    C:/vsr/mmediting/tools/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}