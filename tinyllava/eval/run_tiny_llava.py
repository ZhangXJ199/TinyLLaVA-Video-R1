import argparse
import re
import requests
from PIL import Image
from io import BytesIO

import time

import torch
from transformers import PreTrainedModel
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import functional as F
from torchvision.io import read_video

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *
import numpy as np

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def video_parser(args):
    out = args.video_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def save_frames(frames, save_dir="/mnt/data/zxj/others/demo"):
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray((frame.cpu().numpy().transpose(1, 2, 0)).astype('uint8'))
        img.save(os.path.join(save_dir, f"frame_{i}.png"))

def eval_model(args):
    # Model
    disable_torch_init()

    if args.model_path is not None:
        model, tokenizer, image_processor, context_len = load_pretrained_model(args.model_path)
    else:
        assert args.model is not None, 'model_path or model must be provided'
        model = args.model
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
        tokenizer = model.tokenizer
        image_processor = model.vision_tower._image_processor
    qs = args.query
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_preprocess = ImagePreprocess(image_processor, data_args)
    video_preprocess = VideoPreprocess(image_processor, data_args)

    model.cuda()

    msg = Message()
    msg.add_message(qs)

    result = text_processor(msg.messages, mode='eval')
    print("result:",result)
    input_ids = result['input_ids']
    prompt = result['prompt']
    input_ids = input_ids.unsqueeze(0).cuda()
    
    images_tensor = None
    video_tensor = None
    if args.image_file is not None:
        image_files = image_parser(args)
        images = load_images(image_files)[0]
        images_tensor = image_preprocess(images)
        images_tensor = images_tensor.unsqueeze(0).half().cuda()
    
    if args.video_file is not None:
        video = EncodedVideo.from_path(args.video_file, decoder="decord", decode_audio=False)
        duration = video.duration
        video_data = video.get_clip(start_sec=0.0, end_sec=duration)
        video_data = video_data['video'].permute(1, 0, 2, 3) #torch.Size([l, 3, W, H])

        total_frames = video_data.shape[0]
        if args.num_frame > 0:
            frame_indices = np.linspace(0, total_frames - 1, args.num_frame, dtype=int)
        else:
            num_frames_to_extract = min(args.max_frame, max(1, int(duration)))
            frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)
        video_data = video_data[frame_indices]

        #save_frames(video_data)

        videos = []
        for video in video_data:
            video = video_preprocess(video)
            videos.append(video)
        video_tensor = torch.stack(videos)
        video_tensor = video_tensor.unsqueeze(dim=0)

    stop_str = text_processor.template.separator.apply()[1]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        print("tokenizer.pad_token_id:",tokenizer.pad_token_id)
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            video=video_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    
    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model", type=PreTrainedModel, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_frame", type=int, default=1)
    parser.add_argument("--max_frame", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)