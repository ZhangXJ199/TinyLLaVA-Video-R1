from tinyllava.eval.run_tiny_llava import eval_model

model_path = "/mnt/data/zxj/result/reasoning/TinyLLaVA-Video-Group-16-512"
#"/mnt/data/zxj/result/reasoning/llava_video_reason-3.16/tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-base-reason"
#"/mnt/data/zxj/result/store_llava_video_factory/llava_video_factory-1.13/tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-base-finetune"
prompt = "What happened before the person held the food?" 
video_file = "/mnt/data/zxj/others/demo_video/CJ58B.mp4"
conv_mode = "qwen2_base" # or llama, gemma, etc

args = type('Args', (), {
    "model_path": model_path,
    "model": None,
    "query": prompt,
    "conv_mode": conv_mode,
    "image_file": None, #image_file,
    "video_file": video_file, #video_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "num_frame": 16,
    "max_frame": 16,
    "max_new_tokens": 512
})()

eval_model(args)