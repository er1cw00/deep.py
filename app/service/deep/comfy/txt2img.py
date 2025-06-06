import os
import sys
import json
import torch
import random
import argparse
import contextlib
from diffusers import StableDiffusionPipeline
from typing import Sequence, Mapping, Any, Union
from safetensors.torch import load_file
from app.deepfake.utils import check_device, check_resolution


parser = argparse.ArgumentParser(
    description="txt2img, Required inputs listed below. Values passed should be valid JSON (assumes string if not valid JSON)."
)
parser.add_argument(
    "--device",
    "-d",
    default='cpu',
    help="Device, should be cuda, mps, rocm, cpu",
)
parser.add_argument(
    "--step",
    "-s",
    type=int,
    default=28,
    help="The location to save the output image. should be a file path",
)
parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="The location to save the output image. should be a file path",
)
parser.add_argument(
    "--model_path",
    "-m",
    default=None,
    help="The model path",
)
parser.add_argument(
    "--checkpoint",
    "-c",
    default=None,
    help="The checkpoint model name",
)
parser.add_argument(
    "--width",
    "-W",
    type=int,
    default=1024,
    help="Max width of output resolution",
)
parser.add_argument(
    "--height",
    "-H",
    type=int,
    default=768,
    help="Max height of output resolution",
)
parser.add_argument(
    "--positive_prompt",
    "-P",
    default='beautiful scenery nature glass',
    help="positive_prompt",
)
parser.add_argument(
    "--negative_prompt",
    "-N",
    default='text, watermark, nsfw, nipples',
    help="negative_prompt",
)

def main(*func_args, **func_kwargs):
    args = parser.parse_args()
    use_xformers = False
    use_memory_efficient_attention = False
    
    if args.model_path is None or os.path.isdir(args.model_path) == False:
        print(f'model path: {args.checkpoint} not exist')
        sys.exit(1001)
        
    checkpoint = os.path.join(args.model_path, f'checkpoints/{args.checkpoint}')
    if os.path.isfile(checkpoint) == False:
        print(f'checkpoint path: {checkpoint} not exist')
        sys.exit(1001)
        
    cache_path = os.path.join(args.model_path, 'cache')
    if os.path.isdir(cache_path) == False:
        os.makedirs(cache_path, exist_ok=True)
        
    config_file = os.path.join(args.model_path, 'configs/v1-inference.yaml')
    if os.path.isfile(checkpoint) == False:
        print(f'config file: {checkpoint} not exist')
        sys.exit(1001)
        
    if args.output is None or os.path.isdir(os.path.dirname(args.output)) == False:
        print(f'output path: {args.output} not exist')
        sys.exit(1001) 
        
    if args.step <= 0:
        args.step = 1
    elif args.step >= 100:
        args.step = 100
        
    width, height = check_resolution(args.width, args.height)
    device = check_device(args.device.lower())
    if device == 'cuda':
        use_xformers = True
        use_memory_efficient_attention = True
    
    print(f'step: {args.step}')
    print(f'resolution: {width}x{height}')
    print(f'cache path: {cache_path}')
    print(f'config file: {config_file}')
    print(f'ckpt path: {checkpoint}')
    
    pipe = StableDiffusionPipeline.from_single_file(
        checkpoint,  # 你可以换成其他模型，比如 stabilityai/stable-diffusion-2-1
        torch_dtype=torch.float16,
        cache_dir=cache_path,
        original_config=config_file,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        variant="fp16",
    )
    print(f'device: {device}')
    pipe = pipe.to(device)  # 如果你有GPU，使用CUDA加速
    pipe.enable_attention_slicing()
    
    if use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[Info] xformers memory efficient attention enabled.")
        except Exception as e:
            print(f"[Warning] xformers not available: {e}")

    # 是否启用 diffusers 内置的 memory efficient attention
    if use_memory_efficient_attention:
        try:
            pipe.enable_model_cpu_offload()  # 让不必要的部分放到CPU，进一步省显存
            print("[Info] model CPU offload enabled.")
        except Exception as e:
            print(f"[Warning] CPU offload failed: {e}")
    #if use_        
    #pipe.enable_attention_slicing()
    with torch.no_grad():
        image = pipe(
            prompt=args.positive_prompt,
            negative_prompt=args.negative_prompt,
            width=width,         # 生成图片宽度，必须是64的倍数
            height=height,        # 生成图片高度，必须是64的倍数
            num_inference_steps= args.step,  # 步数（越大越清晰，但也越慢）
            guidance_scale=7.5       # 提示词引导强度（7-8一般比较好）
        ).images[0]
    
    image.save(args.output)
    
if __name__ == "__main__":
    main()
