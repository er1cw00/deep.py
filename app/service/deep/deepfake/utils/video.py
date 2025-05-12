import imageio



def get_video_writer(output_path, fps):
    video_format = 'mp4'     # default is mp4 format
    codec = 'libx265'        # default is libx264 encoding
    #quality = quality        # video quality
    pixelformat = 'yuv420p'  # video pixel format
    image_mode = 'rbg'
    macro_block_size = 2
    ffmpeg_params = ['-crf', '22', '-preset', 'medium', '-tag:v', 'hvc1', '-loglevel', 'quiet']
    writer = imageio.get_writer(uri=output_path,
                        format=video_format,
                        fps=fps, 
                        codec=codec, 
                        ffmpeg_params=ffmpeg_params, 
                        pixelformat=pixelformat, 
                        macro_block_size=macro_block_size)
    return writer