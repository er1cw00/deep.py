import os
import cv2
import time
import imageio
import numpy as np
from rich.progress import track
from app.deepfake.facefusion.modules.yoloface import YoloFace
from app.deepfake.facefusion.modules.occluder import Occluder
from app.deepfake.facefusion.modules.xseg import XSeg
from app.deepfake.facefusion.utils.mask import overlay_mask_on_face
from app.deepfake.facefusion.utils.affine import arcface_128_v2, ffhq_512, warp_face_by_landmark_5, paste_back, blend_frame
from app.deepfake.utils.timer import Timer
from app.deepfake.utils.video import get_video_writer
from .file import get_test_files

os.environ["ORT_LOGGING_LEVEL"] = "VERBOSE"        
    
def test_image(yolo, xseg1, xseg2, input_path, output_path):

    image = cv2.imread(input_path)
    face_list = yolo.get(image=image, order='best-worst')
    if face_list != None and len(face_list) > 0:
        face = face_list[0]
        resized_face, affine = warp_face_by_landmark_5(image, face.landmark_5, arcface_128_v2, (256,256))
        t1 = Timer()
        t2 = Timer()
        t1.tic()
        mask1 = xseg1.detect(image=resized_face)
        t1.toc()
        t2.tic()
        mask2 = xseg2.detect(image=resized_face)
        t2.toc()
        
        mask1 = (mask1 * 255).clip(0, 255).astype(np.uint8)
        mask2 = (mask2 * 255).clip(0, 255).astype(np.uint8)
        
        output1 = overlay_mask_on_face(resized_face, mask1, alpha=0.5, color=(0, 0, 255))
        output2 = overlay_mask_on_face(resized_face, mask2, alpha=0.5, color=(0, 0, 255))
        #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        combined = cv2.hconcat([resized_face, output1, output2])
        cv2.imwrite(output_path, combined)
        t1.show(f'xseg1 parse photo')
        t2.show(f'xseg2 parse photo')
    
def test_video(yolo, xseg1, xseg2, input_path, output_path):
    
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
        face_list = yolo.get(image=frame, order='best-worst')
        if face_list != None and len(face_list) > 0:
            face = face_list[0]
            resized_face, affine = warp_face_by_landmark_5(frame, face[1], arcface_128_v2, (256,256))
            
            t1.tic()
            mask1 = xseg1.detect(image=resized_face)
            t1.toc()
            mask1 = (mask1 * 255).clip(0, 255).astype(np.uint8)
            t2.tic()
            mask2 = xseg2.detect(image=resized_face)
            t2.toc()
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
        

        
providers = ['CUDAExecutionProvider']
yolo_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'
seg_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/dfl_xseg.onnx'
seg1_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/xseg_1.onnx'
#seg0_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/occluder.onnx'
seg0_path = '/home/eric/workspace/AI/sd/ComfyUI/models/facefusion/xseg_sim_2.onnx'

yolo = YoloFace(model_path=yolo_path, providers=providers)
#xseg0 = Occluder(model_path=seg0_path, providers=providers)
xseg0 = XSeg(model_path=seg0_path, providers=providers)
xseg1 = XSeg(model_path=seg1_path, providers=providers)
    
    # input_path = '/Users/wadahana/Desktop/sis/faceswap/test/sq/suck2.mp4 '
    # output_path = './suck2_mask.mp4'
    
photo_list, video_list = get_test_files()

print("Photo directories:")
for path in photo_list:
    input_path = os.path.join(path, 'target.jpg')
    output_path = os.path.join(path, 'output_mask2.png')
    print(path)
    test_image(yolo, xseg0, xseg1, input_path, output_path)


# print("\nVideo directories:")
# for path in video_list:
#     input_path = os.path.join(path, 'target.mp4')
#     output_path = os.path.join(path, 'output_mask2.mp4')
#     print(path)
#     print(input_path)
#     print(output_path)
#     test_video(yolo, xseg0, xseg1, input_path, output_path)
    
#test_video(yolo, input_path, output_path, xseg0, xseg1)

    
    