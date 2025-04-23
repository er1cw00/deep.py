import os
import sys
import random
import imageio
import ffmpeg
from loguru import logger
from typing import Sequence, Mapping, Any, Union
from app.base.error import Error



def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]
    
    
def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        if args is None or args.comfyui_directory is None:
            path = os.getcwd()
        else:
            path = args.comfyui_directory
    
    print(f'path: {path}')
    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def get_providers_from_device(device):
    providers = ['CPUExecutionProvider']
    if device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif device == 'coreml' or device == 'mps':
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    elif device == 'rocm':
        providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
    return providers

def add_tbox_path_to_sys_path(tbox_path) -> None:
    if os.path.isdir(tbox_path):
        sys.path.append(tbox_path)
    else: 
        print(f"Path not found: {tbox_path}")

def add_comfy_path_to_sys_path(comfyui_path) -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    #comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")

def restore_audio(target_path, output_path, duration):
    temp_path = output_path.replace('.mp4', '_noaudio.mp4')
    os.rename(output_path, temp_path)

    try:
        input_video = ffmpeg.input(temp_path)
        input_audio = ffmpeg.input(target_path)

        kwargs = {
            'map': ['0:v:0', '1:a:0?'],
            'c': 'copy',
            'shortest': None
        }
        if duration is not None:
            input_video = ffmpeg.input(temp_path, ss=0, to=duration)
            input_audio = ffmpeg.input(target_path, ss=0, to=duration)
            
        ffmpeg.output(input_video, input_audio, output_path, **kwargs).run(overwrite_output=True)
    except ffmpeg.Error as e:
        logger.error('ffmpeg error:', e.stderr.decode() if e.stderr else str(e))
        return Error.FFmpegError
    except Exception as e:
        logger.error('ffmpeg error:', e.stderr.decode() if e.stderr else str(e))
        return Error.Unknown
    
    return Error.OK
    # ffmpeg.input(temp_path).output(
    #     output_path,
    #     #vf='fps={}'.format(fps),  # 保持帧率一致
    #     i=target_path,
    #     map='0:v:0',  # 使用处理后的视频
    #     map='1:a:0?',  # 如果存在音频，则使用原音频
    #     c='copy',
    #     shortest=None
    # ).run(overwrite_output=True)
    
def get_video_writer(output_path, fps):
    video_format = 'mp4'     # default is mp4 format
    codec = 'libx265'        # default is libx264 encoding
    #quality = quality        # video quality
    pixelformat = 'yuv420p'  # video pixel format
    image_mode = 'rbg'
    macro_block_size = 2
    ffmpeg_params = ['-crf', '22', '-preset', 'medium', '-tag:v', 'hvc1']
    writer = imageio.get_writer(uri=output_path,
                        format=video_format,
                        fps=fps, 
                        codec=codec, 
                        ffmpeg_params=ffmpeg_params, 
                        pixelformat=pixelformat, 
                        macro_block_size=macro_block_size)
    return writer


