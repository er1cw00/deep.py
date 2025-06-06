import os
import cv2
import sys
import json
import torch
import random
import argparse
import contextlib
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from typing import Sequence, Mapping, Any, Union
from safetensors.torch import load_file
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from app.deepfake.ip_adapter.ip_adapter import IPAdapterPlus 
from app.deepfake.ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from app.deepfake.utils import check_resolution, check_device, get_providers_from_device, constrain_image



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
    "--model_path",
    "-m",
    default=None,
    help="The location of models path. should be a dir path",
)
parser.add_argument(
    "--input",
    "-i",
    default=None,
    help="The location of input image path. should be a file path",
)
parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="The location to save the output image. should be a file path",
)
parser.add_argument(
    "--checkpoint",
    "-c",
    default=None,
    help="The checkpoint model name",
)
parser.add_argument(
    "--max-width",
    "-W",
    type=int,
    default=1024,
    help="Max width of output",
)
parser.add_argument(
    "--max-height",
    "-H",
    type=int,
    default=1024,
    help="Max height of output",
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
    face_detection_threshold = 0.5
    
    if args.model_path is None or os.path.isdir(args.model_path) == False:
        print(f'model path: {args.model_path} not exist')
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
        
    if args.input is None or os.path.isfile(args.input) == False:
        print(f'input image: {args.input} not exist')
        sys.exit(1000) 
        
    if args.output is None or os.path.isdir(os.path.dirname(args.output)) == False:
        print(f'output path: {args.output} not exist')
        sys.exit(1001) 
        
    max_width, max_height = check_resolution(args.max_width, args.max_height)
    device = check_device(args.device.lower())
    if device == 'cuda':
        use_xformers = True
        use_memory_efficient_attention = True
        
    providers = get_providers_from_device(device)
    positive_prompt = "1girl, masterpiece, high quality, fantasy style"
    negative_prompt = "nsfw,watermark"
    direction = 'large-small'
    
    # model = load_file(args.checkpoint)
    image = Image.open(args.input).convert("RGB")
    image = constrain_image(image, max_width=max_width, max_height=max_height, min_width=512, min_height=512, crop_if_required=True)
    
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(    
#    pipe = StableDiffusionPipeline.from_single_file(
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
            
    with torch.no_grad():
        image = pipe(
            image=image,
            strength=0.5,  # 越小越接近原图，范围 0.0~1.0
            guidance_scale=7.5,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=26
        ).images[0]

    image.save(args.output)
    
    if device == 'mps' and hasattr(torch, 'mps'):
        torch.mps.empty_cache()
    elif device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
if __name__ == "__main__":
    main()
