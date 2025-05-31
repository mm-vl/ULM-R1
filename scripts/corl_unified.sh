#!/bin/bash

# *****************  ***************** #
CKPT_PATH=path/to/checkpoint/Janus-Pro-1B
DATA_PATH=path/to/data/x2x_rft_22k

task_format=unify
SAVE_DIR=path/to/experiment/JanusPro-1B-CoRL-Unified

reward_funcs="t2i_bid_cycle_reward t2i_ti_sim qa_accuracy format"
#beta=0.04
beta=0.0
unify_advantage=False
unify_reward=True

learning_rate=4e-6
num_train_epochs=1

max_prompt_length=1024
max_completion_length=576

num_generation=8
gradient_accumulation_steps=4
per_device_train_batch_size=1

SAVE_PATH=${SAVE_DIR}/RFT22k-CycleMatchAccFormat-UniReward-G8-beta004-bs16
mkdir -p $SAVE_PATH
cp $0 $SAVE_PATH/run.sh

# --deepspeed scripts/zero3.json
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    corl/open_r1/grpo_janus_unify.py \
    --reward_funcs ${reward_funcs} \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --report_to wandb \
    --logging_steps 1 \
    --unify_advantage $unify_advantage \
    --unify_reward $unify_reward \
    --beta $beta \
    --task_format ${task_format} \
    --max_prompt_length $max_prompt_length \
    --max_completion_length $max_completion_length \
    --num_generations $num_generation \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs $num_train_epochs \
    --learning_rate $learning_rate \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing false \
    --save_steps 200 \
    --save_total_limit 1 \
    --save_only_model true

