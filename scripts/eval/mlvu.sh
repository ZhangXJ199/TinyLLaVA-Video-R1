#!/bin/bash

MODEL_PATH="/data/vlm/zxj/result/reasoning/llava_video_reason_nextqa-4.8"
MODEL_NAME="llava_video_reason_nextqa-4.8"
EVAL_DIR="/data/vlm/zxj/data/MLVU"

# num_frame=-1 means 1fps
python -m tinyllava.eval.eval_mlvu \
    --model-path $MODEL_PATH \
    --video-folder $EVAL_DIR/video \
    --question-file $EVAL_DIR/json \
    --answers-file $EVAL_DIR/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode qwen2_base \
    --num_frame 16 \
    --max_frame 16 
    
