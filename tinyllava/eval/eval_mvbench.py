import json
import os
import re
import argparse
import torch
from tqdm import tqdm
import shortuuid
import random
import cv2
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

data_list = {
    "Fine-grained Pose": ("fine_grained_pose.json", "/nturgbd/", "video_avi", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Action Sequence": ("action_sequence.json", "/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "/ssv2_video/", "video_webm", False),
    "Fine-grained Action": ("fine_grained_action.json", "/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "/perception/videos/", "video", False),
    "Character Order": ("character_order.json", "/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "/vlnqa/", "video", False),
    "Counterfactual Inference": ("counterfactual_inference.json", "/clevrer/video_validation/", "video", False),
}

class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, video_processor, num_frames=16, max_num_frames=16):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'frame': self.read_frame,
            'video_webm': self.read_video_webm,
            'video_avi': self.read_video_avi
        }
        
        self.video_processor = video_processor
        self.num_frames = num_frames
        self.max_num_frames = max_num_frames

    
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
    
    def get_index(self, bound, fps, max_frame, first_idx=0, num_frame=16):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_frame
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_frame)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        try:
            video_data = video.get_clip(start_sec=0.0, end_sec=duration)
        except Exception as e:
            print(f"Corrupted video found: {video_path}, Error: {e}")
        video_data = video_data['video'].permute(1, 0, 2, 3) #torch.Size([l, 3, W, H])

        total_frames = video_data.shape[0]
        if self.num_frames > 0:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            num_frames_to_extract = min(self.max_num_frames, max(1, int(duration)))
            frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
        video_data = video_data[frame_indices]
        
        videos = []
        for video in video_data:
            video = self.video_processor(video)
            videos.append(video)
        video_tensor = torch.stack(videos)
        video_tensor = video_tensor.unsqueeze(dim=0)
    
        return video_tensor
    
    def read_video_avi(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        duration = len(vr) / fps
        
        images_group = []
        if self.num_frames > 0:
            num_frames_to_extract = self.num_frames
        else:
            num_frames_to_extract = min(self.max_num_frames, max(1, int(duration)))
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_frame=num_frames_to_extract) 
        for frame_index in frame_indices:
            img = torch.tensor(vr[frame_index].asnumpy())
            img = self.video_processor(img)
            images_group.append(img)
        video_tensor = torch.stack(images_group)  # Shape: (num_segments, C, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # Shape: (1, num_segments, C, H, W)

        return video_tensor
    
    def read_video_webm(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        duration = len(vr) / fps
        
        images_group = []
        if self.num_frames > 0:
            num_frames_to_extract = self.num_frames
        else:
            num_frames_to_extract = min(self.max_num_frames, max(1, int(duration)))
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_frame=num_frames_to_extract) 
        for frame_index in frame_indices:
            img = vr[frame_index]
            img = self.video_processor(img)
            images_group.append(img)
        video_tensor = torch.stack(images_group)  # Shape: (num_segments, C, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # Shape: (1, num_segments, C, H, W)

        return video_tensor
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        duration = max_frame / fps
        images_group = list()

        images_group = []
        if self.num_frames > 0:
            num_frames_to_extract = self.num_frames
        else:
            num_frames_to_extract = min(self.max_num_frames, max(1, int(duration)))
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1, num_frame=num_frames_to_extract)
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg")).convert("RGB")
            img = self.video_processor(img)
            images_group.append(img)
        torch_imgs = torch.stack(images_group)
        torch_imgs = torch_imgs.unsqueeze(0)
        return torch_imgs

    def qa_template(self, data):
        question = f"{data['question']}\nOptions:\n"
        answer = data['answer']
        answer_idx = -1
        all_choices = []
        index2ans = {}
        for idx, c in enumerate(data['candidates']):
            question += f"{chr(ord('A') + idx)}. {c}\n"
            all_choices.append(chr(ord('A') + idx))
            index2ans[chr(ord('A') + idx)] = c
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"{chr(ord('A') + answer_idx)}"
        return question, answer, all_choices, index2ans

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer, all_choices, index2ans = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'video_path': video_path,
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

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device="cuda")

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    video_processor = VideoPreprocess(image_processor, data_args)
    
    updated_data_list = {
        key: (val[0], args.image_folder + val[1], val[2], val[3]) for key, val in data_list.items()
    }
    
    dataset = MVBench_dataset(args.question_file, updated_data_list, video_processor, num_frames=args.num_frame, max_num_frames=args.max_frame)
    
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}

    for example in tqdm(dataset):
        print("example:",example['video_path'])
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        #print("example:",example) # video, question, answer, task_type
        
        question = example["question"]
        question = "<image>" + "\n" + question + "\n" + "Output the thinking process in <think> </think> and final answer (option) in <answer> </answer> tags."
        #question = "<image>" + "\n" + question + "\n" + "Answer with the option's letter from the given choices directly."
        print("question:",question)
        
        msg = Message()
        msg.add_message(question)
        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        input_ids = input_ids.unsqueeze(0).cuda()
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                video=example["video"],
                do_sample=True,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print("outputs:",outputs)
    
        gt = example['answer']
        print("ground truth:", gt)
        pred_ans = parse_multi_choice_response(outputs, example['all_choices'], example['index2ans'])
        print("pred_ans:", pred_ans)
        
        res_list.append({
            'outputs': outputs,
            'pred': pred_ans,
            'gt': gt,
            'path': example['video_path']
        })
        
        if pred_ans==gt:
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print(f"Total Acc: {correct / total * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)

    final_res = dict()
    correct = 0
    total = 0
    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]    
    final_res['Avg'] = correct / total * 100

    print(final_res)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w") as f:
        json.dump(final_res, f)
        f.write("\n")
        for item in res_list:
            json.dump(item, f)
            f.write("\n")
        


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_frame", type=int, default=1)
    parser.add_argument("--max_frame", type=int, default=1)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    args = parser.parse_args()

    eval_model(args)
