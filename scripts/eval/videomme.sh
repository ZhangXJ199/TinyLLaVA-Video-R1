#!/bin/bash

MODEL_PATH="/data/vlm/zxj/result/reasoning/llava_video_reason_nextqa-4.8"
MODEL_NAME="llava_video_reason_nextqa-4.8"
EVAL_DIR="/data/vlm/zxj/data/Video-MME"

# num_frame=-1 means 1fps
python -m tinyllava.eval.eval_videomme \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/videomme/test-00000-of-00001.parquet \
    --image-folder $EVAL_DIR/data \
    --answers-file $EVAL_DIR/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode qwen2_base \
    --duration long \
    --num_frame 16 \
    --max_frame 16 
