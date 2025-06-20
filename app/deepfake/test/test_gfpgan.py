import os
import cv2
import time
import imageio
import numpy as np
from rich.progress import track
from app.deepfake.facefusion.modules.yoloface import YoloFace
from app.deepfake.facefusion.modules.occluder import Occluder
from app.deepfake.facefusion.modules.gfpgan import GFPGAN
from app.deepfake.facefusion.utils.mask import overlay_mask_on_face, create_bbox_mask
from app.deepfake.facefusion.utils.affine import arcface_128_v2, ffhq_512, warp_face_by_landmark_5, paste_back, blend_frame
from app.deepfake.utils.timer import Timer
from app.deepfake.utils.video import get_video_writer
from .file import get_test_files


os.environ["ORT_LOGGING_LEVEL"] = "VERBOSE"        
    
def test_image(yolo, gfpgan, input_path, output_path):
    image = cv2.imread(input_path)
    t1 = Timer()
    t2 = Timer()
    
    t1.tic()
    face_list = yolo.get(image=image, order='best-worst')
    t1.toc()
    
    if face_list != None and len(face_list) > 0:
        face = face_list[0]
        face_image, affine = warp_face_by_landmark_5(image, face.landmark_5, ffhq_512, gfpgan.input_size)
        box_mask = create_bbox_mask(face_image.shape[:2][::-1], 0.3, (0,0,0,0))
        crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
        
        print(f'face_image shape: {face_image.shape}')
        
        t2.tic()
        output = gfpgan.run(face_image)
        t2.toc()
        
        output = paste_back(image, output, crop_mask, affine)
        output = blend_frame(image, output, 0.6)
        
        cv2.imwrite(output_path, output)
        
    t1.show("photo yolo face detect")
    t2.show("photo gfpgan face enhance")

def test_video(yolo, gfpgan, input_path, output_path):    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
    
    writer = get_video_writer(output_path, fps)

    t1 = Timer()
    t2 = Timer()
    print(f'video [{width}x{height}@{fps}] {total} frames!')
    
    for i in track(range(total), description='Detecting....', transient=True):
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        t1.tic()
        face_list = yolo.get(image=frame, order='best-worst')
        t1.toc()
        
        if face_list != None and len(face_list) > 0:
            face = face_list[0]
            face_image, affine = warp_face_by_landmark_5(frame, face.landmark_5, ffhq_512, gfpgan.input_size)
            box_mask = create_bbox_mask(face_image.shape[:2][::-1], 0.3, (0,0,0,0))
            crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
            
            #print(f'face_image shape: {face_image.shape}')
            
            t2.tic()
            output = gfpgan.run(face_image)
            t2.toc()
            
            output = paste_back(frame, output, crop_mask, affine)
            output = blend_frame(frame, output, 0.6)
            writer.append_data(output[..., ::-1])
            
    cap.release()
    writer.close()
    
    t1.show(f'yolo detece video {total} frames')
    t2.show(f'gfpgan enhance video {total} frames')
    
# yolo_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'
# gfpgan_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/gfpgan_1.4.onnx'
yolo_path = '/data/models/yoloface_8n.onnx'
gfpgan_path = '/data/models/gfpgan_1.4.onnx'

providers=['CUDAExecutionProvider', 'CPUExecutionProvider']

yolo = YoloFace(model_path=yolo_path, providers=providers, threshold=0.4)
gfpgan = GFPGAN(model_path=gfpgan_path, providers=providers)
    
# input_path = "/home/eric/workspace/AI/sd/temp/mask/0d4fc3a041d6befd7a2ee218da2d820b/source.jpg"
# output_path = "/home/eric/workspace/AI/sd/temp/mask/0d4fc3a041d6befd7a2ee218da2d820b/output_face2.jpg" 
input_path = "/data/task/20250505/0d4fc3a041d6befd7a2ee218da2d820b/source.jpg"
output_path = "/data/task/20250505/0d4fc3a041d6befd7a2ee218da2d820b/output_face2.jpg" 


test_image(yolo=yolo, gfpgan=gfpgan, input_path=input_path, output_path=output_path)

# input_path = "/home/eric/workspace/AI/sd/temp/mask/1e42b87f42559936a9447be1bce59165/target.mp4"
# output_path = "/home/eric/workspace/AI/sd/temp/mask/1e42b87f42559936a9447be1bce59165/output_face2.mp4" 
input_path = "/data/task/20250505/1e42b87f42559936a9447be1bce59165/target.mp4"
output_path = "/data/task/20250505/1e42b87f42559936a9447be1bce59165/output_face2.mp4" 


test_video(yolo=yolo, gfpgan=gfpgan, input_path=input_path, output_path=output_path)


# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib