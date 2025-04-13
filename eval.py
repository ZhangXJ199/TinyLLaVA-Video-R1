from tinyllava.eval.run_tiny_llava import eval_model

model_path = "/data/vlm/zxj/result/reasoning/TinyLLaVA-Video-SFT-nextqa"
prompt = "What is the action performed by the person in the video?\nA. drop\nB. sit down\nC. pick up\nD. squat down\nOutput the thinking process in <think> </think> and final answer (option) in <answer> </answer> tags." 
video_file = "/data/vlm/zxj/data/MVBench/video/nturgbd/S009C003P016R002A006_rgb.avi"
conv_mode = "qwen2_base" # or llama, gemma, etc

args = type('Args', (), {
    "model_path": model_path,
    "model": None,
    "query": prompt,
    "conv_mode": conv_mode,
    "image_file": None, #image_file,
    "video_file": video_file, #video_file,
    "sep": ",",
    "temperature": 0.1,
    "top_p": None,
    "num_beams": 1,
    "num_frame": 16,
    "max_frame": 16,
    "max_new_tokens": 512
})()

eval_model(args)
