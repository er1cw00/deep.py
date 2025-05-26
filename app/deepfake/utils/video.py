import os
import random
import imageio
import ffmpeg
from loguru import logger
from app.base.error import Error
from app.base.config import config

def has_audio(input_path):
    try:
        probe = ffmpeg.probe(input_path)
        has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
        return has_audio
    except ffmpeg.Error as e:
        logger.error('ffmpeg error:', e.stderr.decode() if e.stderr else str(e))
    except Exception as e:
        logger.error('ffmpeg unknown error:', str(e))
    return False

def restore_audio(target_path, output_path, duration):
    if has_audio(target_path) == False:
        logger.warning(f"target video {target_path} has no audio")
        return Error.OK
    temp_path = output_path.replace('.mp4', '_noaudio.mp4')
    os.rename(output_path, temp_path)

    try:
        kwargs = {
            'c': 'copy',
            'loglevel': 'quiet'
        }
        if duration is not None:
            input_video = ffmpeg.input(temp_path, ss=0, to=duration)
            input_audio = ffmpeg.input(target_path, ss=0, to=duration)
        else:
            input_video = ffmpeg.input(temp_path)
            input_audio = ffmpeg.input(target_path)
            
        o = ffmpeg.output(input_video['v:0'], input_audio['a:0?'], output_path, **kwargs).global_args('-hide_banner') 
        #print(f"ffmpeg output: {o.compile()}")
        o.run(overwrite_output=True)
        os.remove(temp_path)
    except ffmpeg.Error as e:
        logger.error('ffmpeg error:', e.stderr.decode() if e.stderr else str(e))
        return Error.FFmpegError
    except Exception as e:
        logger.error('ffmpeg unknown error:', str(e))
        return Error.Unknown
    
    return Error.OK
    
    
def get_video_writer(output_path, fps):
    codec = config.get('video_codec', 'libx265')
    if codec not in ['libx264', 'libx265', 'hevc_nvenc', 'h264_nvenc']:
        codec = 'libx265'
    logger.info(f'video codec: {codec}') 
    video_format        = 'mp4'     # default is mp4 format
    codec               = 'libx265'        # default is libx264 encoding
    #quality            = quality        # video quality
    pixelformat         = 'yuv420p'  # video pixel format
    image_mode          = 'rbg'
    macro_block_size    = 2
    ffmpeg_params       = ['-crf', '22', '-preset', 'medium', '-tag:v', 'hvc1', '-loglevel', 'quiet']
    
    writer = imageio.get_writer(uri=output_path,
                        format=video_format,
                        fps=fps, 
                        codec=codec, 
                        ffmpeg_params=ffmpeg_params, 
                        pixelformat=pixelformat, 
                        macro_block_size=macro_block_size)
    return writer


if __name__ == "__main__":
    output = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250423/c666a8e4bfd36a0179924e71f85dca20/output.mp4"
    target = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250423/c666a8e4bfd36a0179924e71f85dca20/target.mp4"
    restore_audio( target, output, None)