#!/bin/bash

MODEL_PATH="/data/vlm/zxj/result/reasoning/llava_video_reason_nextqa-4.8"
MODEL_NAME="llava_video_reason_nextqa-4.8"
EVAL_DIR="/data/vlm/zxj/data/MMVU"

# num_frame=-1 means 1fps
python -m tinyllava.eval.eval_mmvu \
    --model_path $MODEL_PATH \
    --image_folder $EVAL_DIR \
    --question_file $EVAL_DIR/validation.json \
    --answers_file $EVAL_DIR/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv_mode qwen2_base \
    --num_frame 16 \
    --max_frame 16 