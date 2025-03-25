from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os

import torch
from transformers import PreTrainedModel
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import functional as F
from torchvision.io import read_video

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *
import numpy as np

MODEL_PATH= "/mnt/data/zxj/result/reasoning/llava_video_reason_count-3.23"
OUTPUT_PATH= "/mnt/data/zxj/result/reasoning/llava_video_reason_count-3.23/test.json"
PROMPT_PATH="/mnt/data/zxj/data/DVD-counting/test_dvd.jsonl"
VIDEO_PATH = "/mnt/data/zxj/data/DVD-counting"

#We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model, tokenizer, image_processor, context_len = load_pretrained_model(MODEL_PATH)
text_processor = TextPreprocess(tokenizer, "qwen2_base")
data_args = model.config
video_preprocess = VideoPreprocess(image_processor, data_args)
model.cuda()

data = []
with open(PROMPT_PATH, "r") as f:
    for line in f:
        data.append(json.loads(line))

# detailed step-by-step
QUESTION_TEMPLATE = "First output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
#"Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
#"First output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

all_outputs = []  # List to store all answers
# Process data in batches
for input_example in tqdm(data):
    video_file = os.path.abspath(os.path.join(VIDEO_PATH, input_example["video_filename"].lstrip("./")))
    video = EncodedVideo.from_path(video_file, decoder="decord", decode_audio=False)
    duration = video.duration
    video_data = video.get_clip(start_sec=0.0, end_sec=duration)
    video_data = video_data['video'].permute(1, 0, 2, 3) #torch.Size([l, 3, W, H])

    total_frames = video_data.shape[0]
    num_frame = 16
    max_frame = 64
    if num_frame > 0:
        frame_indices = np.linspace(0, total_frames - 1, num_frame, dtype=int)
    else:
        num_frames_to_extract = min(max_frame, max(1, int(duration)))
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
    video_data = video_data[frame_indices]
            
    videos = []
    for video in video_data:
        video = video_preprocess(video)
        videos.append(video)
    video_tensor = torch.stack(videos)
    video_tensor = video_tensor.unsqueeze(dim=0)
    
    qs = input_example["problem"]
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs + QUESTION_TEMPLATE
    
    msg = Message()
    msg.add_message(qs)

    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids']
    input_ids = input_ids.unsqueeze(0).cuda()
    
    output_ids = model.generate(
        inputs=input_ids,
        video=video_tensor,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=1024,
        use_cache=True,)
    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    print("outputs:",outputs)
    all_outputs.append(outputs)


def extract_number_answer(output_str):
    # Try to find the number within <answer> tags, if can not find, return None
    answer_pattern = r'<answer>\s*(\d+)\s*</answer>'
    match = re.search(answer_pattern, output_str)
    
    if match:
        return int(match.group(1))
    return None


final_output = []
correct_number = 0
print("all_outputs:",all_outputs)
for input_example, model_output in zip(data, all_outputs):
    ground_truth = extract_number_answer(input_example['solution'])
    print("ground_truth:",ground_truth)
    model_answer = extract_number_answer(model_output)
    print("model_answer:",model_answer)
    
    # Create a result dictionary for this example
    result = {
        'question': input_example,
        'ground_truth': ground_truth,
        'model_output': model_output,
        'extracted_answer': model_answer
    }
    final_output.append(result)
    
    # Count correct answers
    if model_answer is not None and model_answer == ground_truth:
        correct_number += 1

# Calculate and print accuracy
accuracy = correct_number / len(data) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Save results to a JSON file
output_path = OUTPUT_PATH
with open(output_path, "w") as f:
    json.dump({
        'accuracy': accuracy,
        'results': final_output
    }, f, indent=2)

print(f"Results saved to {output_path}")





