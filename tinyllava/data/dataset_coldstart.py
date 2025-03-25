import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os
from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from .video_preprocess import VideoPreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *


import transformers
import torch
from torch.utils.data import Dataset as LSDataset

import numpy as np
import av
from pytorchvideo.data.encoded_video import EncodedVideo

ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(LSDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.num_frames = data_args.num_frames
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)
        self.video_preprocess = VideoPreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample or 'video' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample or 'video' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image
        elif 'video' in sources:
            video_file = self.list_data_dict[i]['video']
            video_folder = os.path.join(self.data_args.video_folder, video_file)

            video = EncodedVideo.from_path(video_folder, decoder="decord", decode_audio=False)
            duration = video.duration
            try:
                video_data = video.get_clip(start_sec=0.0, end_sec=duration)
            except Exception as e:
                print(f"Corrupted video found: {video_folder}, Error: {e}")
            video_data = video_data['video'].permute(1, 0, 2, 3)

            total_frames = video_data.shape[0]
            if self.num_frames > 0:
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                #num_frames_to_extract = min(64, max(1, int(duration))) # 99.75% of train data < 1min
                num_frames_to_extract = max(1, int(duration))
                frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
            video_data = video_data[frame_indices] #torch.Size([8, 3, W, H])

            videos = []
            for video in video_data:
                video = self.video_preprocess(video)
                videos.append(video)
            videos = torch.stack(videos)
            
            data_dict['video'] = videos
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        elif 'video' in instances[0]:
            video = [instance['video'] for instance in instances]
            if all(x is not None and x.shape == video[0].shape for x in video):
                batch['video'] = torch.stack(video)
            else:
                batch['video'] = video

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
