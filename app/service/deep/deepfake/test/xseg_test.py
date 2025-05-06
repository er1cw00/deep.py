import os
import cv2
import time
import imageio
import numpy as np
from rich.progress import track
from deepfake.facefusion.modules.yoloface import YoloFace
from deepfake.facefusion.modules.occluder import Occluder
from deepfake.facefusion.modules.xseg import XSeg
from deepfake.facefusion.utils.mask import overlay_mask_on_face
from deepfake.facefusion.utils.affine import arcface_128_v2, ffhq_512, warp_face_by_landmark, paste_back, blend_frame
from deepfake.utils import Timer

os.environ["ORT_LOGGING_LEVEL"] = "VERBOSE"        
    
# input_path = '/Users/wadahana/Desktop/sis/faceswap/test/sq/suck2.mp4'
# input_path = '/Users/wadahana/Desktop/sis/faceswap/test/mask/0ef45196ed648cb592f89dd89d436dec/target.jpg'

def test_image(yolo, input_path, output_path, xseg1, xseg2):

    image = cv2.imread(input_path)
    face_list = yolo.detect(image=image, conf=0.7)
    face = face_list[0]
    x1, y1, x2, y2 = map(int, face[0])
    face_crop = image[y1:y2, x1:x2]
    resized_face = cv2.resize(face_crop, (256, 256))
    t1 = Timer()
    t2 = Timer()
    t1.tik()
    mask1 = xseg1.detect(image=resized_face)
    t1.tok()
    t2.tik()
    mask2 = xseg2.detect(image=resized_face)
    t2.tok()
    
    mask1 = (mask1 * 255).clip(0, 255).astype(np.uint8)
    mask2 = (mask2 * 255).clip(0, 255).astype(np.uint8)
    
    output1 = overlay_mask_on_face(resized_face, mask1, alpha=0.5, color=(0, 0, 255))
    output2 = overlay_mask_on_face(resized_face, mask2, alpha=0.5, color=(0, 0, 255))
    #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    combined = cv2.hconcat([resized_face, output1, output2])
    cv2.imwrite(output_path, combined)
    t1.show(f'xseg1 parse photo')
    t2.show(f'xseg2 parse photo')
    
def test_video(yolo, input_path, output_path, xseg1, xseg2):
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
    
    writer = get_video_writer(output_path, fps)
    #while True:
    t1 = Timer()
    t2 = Timer()
    
    for i in track(range(total), description='Detecting....', transient=True):
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_list = yolo.detect(image=frame, conf=0.7)
        if face_list != None and len(face_list) > 0:
            face = face_list[0]
            resized_face, affine = warp_face_by_landmark(frame, face[1], arcface_128_v2, (256,256))
            
            x1, y1, x2, y2 = map(int, face[0])
            # face_crop = frame[y1:y2, x1:x2]
            # resized_face = cv2.resize(face_crop, (256, 256))
            t1.tik()
            mask1 = xseg1.detect(image=resized_face)
            t1.tok()
            mask1 = (mask1 * 255).clip(0, 255).astype(np.uint8)
            t2.tik()
            mask2 = xseg2.detect(image=resized_face)
            t2.tok()
            mask2 = (mask2 * 255).clip(0, 255).astype(np.uint8)
            
            output1 = overlay_mask_on_face(resized_face, mask1, alpha=0.5, color=(0, 0, 255))
            output2 = overlay_mask_on_face(resized_face, mask1, alpha=0.5, color=(0, 0, 255))
            
            #output = paste_back(frame, resized_face, mask1, affine)
            
            
            # mask1 = (mask1 * 255).clip(0, 255).astype(np.uint8)
            # mask1 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
            
            # mask2 = (mask2 * 255).clip(0, 255).astype(np.uint8)
            # mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
            
            combined = cv2.hconcat([resized_face, output1, output2])
            writer.append_data(combined[..., ::-1])

    cap.release()
    writer.close()
    
    t1.show(f'xseg1 parse video {total} frames')
    t2.show(f'xseg2 parse video {total} frames')
        
    def get_video_writer(outout_path, fps):
        video_format = 'mp4'     # default is mp4 format
        codec = 'libx264'        # default is libx264 encoding
        #quality = quality        # video quality
        pixelformat = 'yuv420p'  # video pixel format
        image_mode = 'rbg'
        macro_block_size = 2
        ffmpeg_params = ['-crf', '20']
        writer = imageio.get_writer(uri=outout_path,
                            format=video_format,
                            fps=fps, 
                            codec=codec, 
                            #quality=quality, 
                            ffmpeg_params=ffmpeg_params, 
                            pixelformat=pixelformat, 
                            macro_block_size=macro_block_size)
        return writer
        
    providers = ['CoreMLExecutionProvider']
    yolo_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'
    seg_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/dfl_xseg.onnx'
    seg1_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/xseg_1.onnx'
    seg0_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/occluder.onnx'
    
    yolo = YoloFace(model_path=yolo_path, providers=providers)
    xseg0 = Occluder(model_path=seg0_path, providers=providers)
    xseg1 = XSeg(model_path=seg1_path, providers=providers)
    
    # input_path = '/Users/wadahana/Desktop/sis/faceswap/test/sq/suck2.mp4 '
    # output_path = './suck2_mask.mp4'
    
    MASK_DIR = '/home/eric/workspace/AI/sd/temp/mask'

    photo_list = []
    video_list = []

    for subdir in os.listdir(MASK_DIR):
        subdir_path = os.path.join(MASK_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue

        target_jpg = os.path.join(subdir_path, 'target.jpg')
        target_mp4 = os.path.join(subdir_path, 'target.mp4')

        if os.path.isfile(target_jpg):
            photo_list.append(subdir_path)
        elif os.path.isfile(target_mp4):
            video_list.append(subdir_path)
    
    print("Photo directories:")
    for path in photo_list:
        print(path)

    print("\nVideo directories:")
    for path in video_list:
        print(path)
    #test_video(yolo, input_path, output_path, xseg0, xseg1)

    
    