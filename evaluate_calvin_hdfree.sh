#!/bin/bash

export MESA_GL_VERSION_OVERRIDE=3.3

export CALVIN_ROOT="/home/geyuan/code/github/calvin"
export EVAL_DIR=eval_logs/
export DEVICE="0,1,2,3,4,5,6,7,8,9"

GOOGLE_PATH="/data3/geyuan/pretrained/google/"
GOOGLE_LINK="google"
if [ ! -e ${GOOGLE_LINK} ]; then
  ln -s ${GOOGLE_PATH} ${GOOGLE_LINK}  # link to your downloaded google models (t5, siglip)
fi

# HDFree
python3 evaluate_calvin_rdt.py \
    --weight_path "/home/geyuan/code/rdt24/checkpoints/rdt-finetune-calvin-1b-v3-hdfree/checkpoint-4000"  \
    --eval_dir ${EVAL_DIR} \
    --dataset_dir "task_D_D/" \
    --device ${DEVICE} \
    --hdfree  \
    --seed 42
