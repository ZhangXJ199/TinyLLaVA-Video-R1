from packaging import version
import re
from datetime import datetime
from math_verify import parse, verify

import tokenizers
import transformers
from datasets import Dataset, DatasetDict

from tinyllava.train.tinyllava_trainer_reason import LLaVATrainer_Reason
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data import *


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments)
    return model_args

def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args['attn_implementation'] = model_arguments.attn_implementation # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args

def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args

def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.0
        answer = parse(content)
        if float(verify(answer, parse(sol))) > 0:
            reward = 1.0

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
            # Extract answer from content if it has think/answer tags
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
                
            # Compare the extracted answers
            if len(student_answer) > 0:
                if student_answer == ground_truth or ground_truth==student_answer[0]:
                    reward = 1.0  
        rewards.append(reward)
    return rewards

def extract_first_think_answer(content):
    think_pattern = r"<think>(.*?)</think>"
    #answer_pattern = r"<answer>(.*?)</answer>"

    think_match = re.search(think_pattern, content, re.DOTALL)
    #answer_match = re.search(answer_pattern, content, re.DOTALL)

    think_content = think_match.group(1).strip() if think_match else None
    #answer_content = answer_match.group(1).strip() if answer_match else None

    return think_content

def recheck_format(content):
    think_open_count = content.count("<think>")
    think_close_count = content.count("</think>")
    answer_open_count = content.count("<answer>")
    answer_close_count = content.count("</answer>")

    if think_open_count == 1 and think_close_count == 1 and answer_open_count == 1 and answer_close_count == 1:
        return True
    else:
        return False

def has_repeated_content(text):
    sentences = re.split(r'[,.!?]', text)
    seen = set()
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence in seen:
            return True
        seen.add(sentence)
    return False

"""
def format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
"""

def format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    reward = []
    for content in completion_contents:
        for_re = 0.0
        if re.fullmatch(pattern, content, re.DOTALL) and recheck_format(content):
            for_re += 0.5
            think = extract_first_think_answer(content)
            for_re += min(len(think) / 1200, 1) * 0.5
        reward.append(for_re)
    return reward

QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer (option) in <answer> </answer> tags."
def make_conversation_video(example):
    return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments) 
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)

    model = TinyLlavaForConditionalGeneration.from_pretrained(training_arguments.pretrained_model_path).to('cuda')
    model = training_recipe(model)
    model.config.use_cache = False
    tokenizer = model.tokenizer
    #model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    #data_arguments.image_processor = model.vision_tower._image_processor
    #data_arguments.is_multimodal = True
    #log_trainable_params(model)  # not work well with zero3
    
    reward_funcs = [accuracy_reward, format_reward]
    dataset =  DatasetDict({"train": Dataset.from_json(data_arguments.video_folder)})
    dataset = dataset.map(make_conversation_video)
    
    text_processor = TextPreprocess(tokenizer, "qwen2_base")
    image_processor = model.vision_tower._image_processor
    data_args = model.config
    video_preprocess = VideoPreprocess(image_processor, data_args)
    
    trainer = LLaVATrainer_Reason(
        model=model,
        text_processor=text_processor,
        video_preprocess=video_preprocess,
        reward_funcs=reward_funcs,
        args=training_arguments,
        train_dataset=dataset["train"],
        processing_class=model.tokenizer,
        attn_implementation=model_arguments.attn_implementation,
        data_path=data_arguments.video_data_path,
    )
    trainer.train()
    training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()
