#!/bin/bash

MODEL_PATH="/mnt/data/zxj/result/reasoning/llava_video_reason_nextqa-3.23"
MODEL_NAME="llava_video_reason_nextqa-3.23"
EVAL_DIR="/mnt/data/zxj/data/MVBench"

# num_frame=-1 means 1fps
python -m tinyllava.eval.eval_mvbench \
    --model-path $MODEL_PATH \
    --image-folder $EVAL_DIR/video \
    --question-file $EVAL_DIR/json \
    --answers-file $EVAL_DIR/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode qwen2_base \
    --num_frame 16 \
    --max_frame 16 
