import torch
import argparse
from torchvision import transforms
import json
from tqdm import tqdm
import os
import re
import numpy as np
from torch.utils.data import Dataset

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

import av
import random
import bisect

def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret

class MLVU(Dataset):
    def __init__(self, data_dir, data_list):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'data': data
                })
        
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    

    def qa_template(self, data):
        question = f"{data['question']}\n"
        answer = data['answer']
        answer_idx = -1
        all_choices = []
        index2ans = {}
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
            all_choices.append(chr(ord('A') + idx))
            index2ans[f"{chr(ord('A') + idx)}"] = c
        question = question.rstrip()
        answer = f"{chr(ord('A') + answer_idx)}"
        
        return question, answer, all_choices, index2ans

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        question, answer, all_choices, index2ans = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': video_path, 
            'question': question, 
            'answer': answer,
            'all_choices': all_choices,
            'index2ans': index2ans,
            'task_type': self.data_list[idx]['task_type']
        }

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

data_list = {
        "count": ("4_count.json", f"/4_count", "video"),
        "ego": ("3_ego.json", f"/3_ego", "video"),
        "needle": ("2_needle.json", f"/2_needle", "video"),
        "order": ("5_order.json", f"/5_order", "video"),
        "plotQA": ("1_plotQA.json", f"/1_plotQA", "video"),
        "anomaly_reco": ("6_anomaly_reco.json", f"/6_anomaly_reco", "video"),
        "topic_reasoning": ("7_topic_reasoning.json", f"/7_topic_reasoning", "video")
    }

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


def eval_model(args):

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device="cuda")

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    video_processor = VideoPreprocess(image_processor, data_args)
    
    dataset = MLVU(args.question_file, data_list)

    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    for example in tqdm(dataset):
        task_type = example['task_type']
        print(f"Video: {example['video']}")
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        video_path = args.video_folder + example["video"]
        question = "<image>" + "\n" + example["question"] + "\n" + "Output the thinking process in <think> </think> and final answer (option) in <answer> </answer> tags."

        msg = Message()
        msg.add_message(question)
        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        input_ids = input_ids.unsqueeze(0).cuda()

        frames = get_video_frames(video_path, args.num_frame, args.max_frame)
        video_tensor = torch.stack([video_processor(frame) for frame in frames])
        video_tensor = video_tensor.unsqueeze(dim=0)
        
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
        gt = example['answer']
        index2ans = example['index2ans']
        all_choices = example['all_choices']
        pred_ans = parse_multi_choice_response(outputs, all_choices, index2ans)
        print("pred_ans:",pred_ans)
        print("gt:", gt)
        
        res_list.append({
            'pred': pred_ans,
            'gt': gt,
            'question':example['question'],
            'question_type':example['task_type'],
            'video':example['video']
        })
        if pred_ans==gt:
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print(f"Overall  Acc: {correct / total * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)

    #final_res = correct / total * 100
    final_res = dict()
    total=0
    idx=0
    for k, v in acc_dict.items():
        idx+=1
        final_res[k] = v[0] / v[1] * 100  
        total+=final_res[k]
    final_res['Avg'] = total /idx 
    print("final_res:", final_res)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w") as f:
        json.dump({
            "final_res": final_res,
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_frame", type=int, default=16)
    parser.add_argument("--max_frame", type=int, default=16)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    args = parser.parse_args()

    eval_model(args)