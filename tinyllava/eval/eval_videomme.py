import argparse
import torch
import os
import re
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import shortuuid

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from PIL import Image
import math
import av
import bisect


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_multi_choice_response(response, all_choices, index2ans):
    all_choices_lower = [choice.lower() for choice in all_choices]
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(answer_pattern, response)
    if match:
        response = match.group(1).strip()
        if len(response) > 0:
            if response[0] in all_choices or response[0] in all_choices_lower:
                return response[0].upper()
            else:
                for choice in all_choices:
                    if f"({choice})" in response or f"({choice.lower()})" in response:
                        return choice
                return random.choice(all_choices)
    else:
        return random.choice(all_choices)
    

def get_video_frames_naive(video_path, num_frames=16, max_frames=16):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    duration = container.streams.video[0].duration
    if num_frames > 0:
        num_frames_to_extract = num_frames
    else:
        num_frames_to_extract = min(max_frames, max(1, int(duration)))
    frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in frame_indices:
            img = frame.to_image()
            frames.append(img)
        if len(frames) >= num_frames_to_extract:
            break
    return frames


def get_video_frames(video_path, num_frames=16, max_frames=16):
    # Open the video container
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Get the metadata of the video
    if video_stream.average_rate == video_stream.base_rate:
        frame_rate = float(video_stream.base_rate)
    else:
        # Variable frame rate mode -- hard to calculate PTSs
        # Fallback to naive method
        return get_video_frames_naive(video_path, num_frames, max_frames)
    time_base = video_stream.time_base
    total_frames = video_stream.frames

    # Calculate the number of frames to extract
    if num_frames > 0:
        num_frames_to_extract = num_frames
    else:
        duration = video_stream.duration * time_base
        num_frames_to_extract = min(max_frames, max(1, int(duration)))

    frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
    frame_indices = np.unique(frame_indices)

    # Calculate timestamps (PTS) of target frames, ensuring reverse order and no duplicates
    remaining_PTSs = (frame_indices / time_base / frame_rate).astype(int).tolist()

    frame_dict = {}

    while remaining_PTSs:
        # Jump to the nearest keyframe (I-frame) before the target frame
        container.seek(
            remaining_PTSs[-1],
            stream=video_stream,
            backward=True,
            any_frame=False
        )
        init = True

        # Decode and find the target frame,
        # considering the case where the keyframe is followed by multiple target frames
        for frame in container.decode(video=0):
            frame_PTS = frame.pts
            if init:
                index = bisect.bisect_left(remaining_PTSs, frame_PTS)
                this_PTSs = remaining_PTSs[index:]
                if not this_PTSs:
                    # In some cases (maybe some variable frame rate videos),
                    # the first keyframe is not indexed as frame 0.
                    index = 0
                    this_PTSs = remaining_PTSs[index:]
                remaining_PTSs = remaining_PTSs[:index]
                init = False

            if frame_PTS >= this_PTSs[0]:
                frame_dict[frame_PTS] = frame.to_image()
                this_PTSs.pop(0)

                if not this_PTSs:
                    break

    sorted_PTS = sorted(frame_dict.keys())
    return [frame_dict[pts] for pts in sorted_PTS]

def select_from_options(options):
    all_choices = [option.split('.')[0] for option in options]
    index2ans = {option.split('.')[0]: option.split('. ')[1][:-1] for option in options}
    return all_choices, index2ans

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    video_processor = VideoPreprocess(image_processor, data_args)

    questions_df = pd.read_parquet(os.path.expanduser(args.question_file))
    questions_df = questions_df[questions_df["duration"] == args.duration]
    questions = questions_df.to_dict(orient="records")
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    model.to(device="cuda")

    total = 0
    correct = 0
    for i, line in enumerate(tqdm(questions)):
        idx = line["videoID"]
        question = line["question"]
        answer = line["answer"]
        print("answer:",answer)
        options = line["options"]
        options_text = "\n".join(options)
        video_path = os.path.join(args.image_folder, f"{line['videoID']}.mp4")
        
        frames = get_video_frames(video_path, args.num_frame, args.max_frame)
        video_tensor = torch.stack([video_processor(frame) for frame in frames])
        video_tensor = video_tensor.unsqueeze(dim=0)

        question = "<image>" + "\n" + question + "\nOptions:\n" + options_text
        question = question + "\n" + "Output the thinking process in <think> </think> and final answer (option) in <answer> </answer> tags."
        #question = question + "\n" + "Answer with the option's letter from the given choices directly."

        msg = Message()
        msg.add_message(question)

        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        input_ids = input_ids.unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                video=video_tensor,
                do_sample=True,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            print("outputs:",outputs)
        all_choices, index2ans = select_from_options(options)
        pred_ans = parse_multi_choice_response(outputs, all_choices, index2ans)
        print("pred_ans:",pred_ans)
            
        if pred_ans==answer:
            correct += 1
            print("correct!")
        total += 1
        print(f"{args.duration} Acc: {correct / total * 100 :.2f}%")
    print(f"{args.duration} num: {total}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="videos/")
    parser.add_argument("--question-file", type=str, default="tables/question.parquet")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--duration", type=str, default="short")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--num_frame", type=int, default=1)
    parser.add_argument("--max_frame", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    eval_model(args)

