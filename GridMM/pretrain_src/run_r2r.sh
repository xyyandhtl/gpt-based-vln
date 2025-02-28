#NODE_RANK=0
#NUM_GPUS=1
#outdir=../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new
#
## train
#CUDA_VISIBLE_DEVICES='0' python3 -m torch.distributed.launch --master_port 29500 \
#    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --use_env \
#    train_r2r.py --world_size ${NUM_GPUS} \
#    --vlnbert cmt \
#    --model_config config/r2r_model_config.json \
#    --config config/r2r_pretrain.json \
#    --output_dir $outdir


#!/bin/bash

# 设置输出目录
outdir=../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new

# 启动训练进程
CUDA_VISIBLE_DEVICES='0' python3 train_r2r.py \
--vlnbert cmt \
--model_config config/r2r_model_config.json \
--config config/r2r_pretrain.json \
--output_dir ../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker-new
