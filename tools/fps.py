import cv2
import imageio

def get_video_writer(output_path, fps):
        
    video_format = 'mp4'     # default is mp4 format
    codec = 'libx264'        # default is libx264 encoding
    #quality = quality        # video quality
    pixelformat = 'yuv420p'  # video pixel format
    image_mode = 'rbg'
    macro_block_size = 2
    ffmpeg_params = ['-crf', '18']
    writer = imageio.get_writer(uri=output_path,
                        format=video_format,
                        fps=fps, 
                        codec=codec, 
                        ffmpeg_params=ffmpeg_params, 
                        pixelformat=pixelformat, 
                        macro_block_size=macro_block_size)
    return writer

def convert_fps(input_video, output_video, target_fps=25):
    cap = cv2.VideoCapture(input_video)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    target_fps = min(target_fps, fps)
    frame_interval = fps / target_fps  # 用于均匀采样
    
    print(f'origin fps: {fps}, target fps: {target_fps}, frame interval: {frame_interval}')
    writer = get_video_writer(output_video, fps=target_fps)

    frame_index = 0
    new_frame_id = 0  # 目标视频的帧编号

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 读完了

        # 选择关键帧进行存储（下采样 / 插值）
        if new_frame_id * frame_interval <= frame_index:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.append_data(frame)  # 仅写入符合间隔的帧
            new_frame_id += 1

        frame_index += 1
    
    cap.release()
    writer.close()
    print(f"转换完成：{input_video} -> {output_video} ({target_fps} FPS)")
    print(f'total frame: {frame_count}, total process frame: {new_frame_id}')
input = "/Users/wadahana/Desktop/sis/faceswap/test/测试人脸检测视频.mp4"
#input = "/Users/wadahana/Desktop/sis/m2m/yummy1.mp4"
output = "/Users/wadahana/workspace/AI/tbox.ai/deep.py/tools/output.mp4"

convert_fps(input, output)