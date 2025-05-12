#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime


class Occluder:
    
    def __init__(self, model_path, providers):
        self.session  = onnxruntime.InferenceSession(model_path, providers=providers)
        print(f'current providers: {self.session.get_providers()}') 
        print(f"available  providers: {onnxruntime.get_available_providers()}")
        inputs = self.session.get_inputs()
        for input in inputs:
            print(f'inputs name: {input.name}, shape: {input.shape}')
        
        self.input_size = (inputs[0].shape[2], inputs[0].shape[3])
        self.input_name = inputs[0].name
        self.affine = False
        
    def pre_process(self, image):
        img = cv2.resize(image, self.input_size)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255
        img = np.expand_dims(img, axis = 0)
        return img
    
    def post_process(self, output, height, width):
        mask = output.transpose(1, 2, 0).clip(0, 1).astype(np.float32)
        mask = cv2.resize(mask, (width, height))
        mask = (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
#        kernel = np.ones((3,3), np.uint8)
#        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def detect(self, image):
        height, width = image.shape[0], image.shape[1]
        img = self.pre_process(image)
        outputs = self.session.run(None, {self.input_name: img})
        output = outputs[0][0]
        output = self.post_process(output, height, width) 
        return output
    
      
if __name__ == "__main__":
    from .yoloface import YoloFace
    from .occluder import Occluder
    from rich.progress import track
    from ..utils.mask import overlay_mask_on_face
    from ..utils.affine import arcface_128_v2, ffhq_512, warp_face_by_landmark, paste_back, blend_frame
    #from deep.utils import get_providers_from_device, get_video_writer
    
    import imageio
    import cv2
    import time
    import os
    

    os.environ["ORT_LOGGING_LEVEL"] = "VERBOSE"
        
   
# input_path = '/Users/wadahana/Desktop/sis/faceswap/test/sq/suck2.mp4'
# input_path = '/Users/wadahana/Desktop/sis/faceswap/test/mask/0ef45196ed648cb592f89dd89d436dec/target.jpg'
    def test_image(yolo, xseg):
        input_path = '/Users/wadahana/Desktop/sis/faceswap/test/mask/0ef45196ed648cb592f89dd89d436dec/target.jpg'
        output_path = '../output_mask.png'
        image = cv2.imread(input_path)
        face_list = yolo.detect(image=image, conf=0.7)
        face = face_list[0]
        x1, y1, x2, y2 = map(int, face[0])
        face_crop = image[y1:y2, x1:x2]
        resized_face = cv2.resize(face_crop, (256, 256))
        mask = xseg.detect(image=resized_face)
        mask = (mask * 255).clip(0, 255).astype(np.uint8)

        output = overlay_mask_on_face(resized_face, mask, alpha=0.5, color=(0, 0, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = cv2.hconcat([resized_face, output, mask])
        cv2.imwrite(output_path, combined)
        
        
    def test_video(yolo, xseg1, xseg2):
        input_path = '../suck2.mp4'
        output_path = '../suck2_mask.mp4'
        
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
        writer = get_video_writer(output_path, fps)
        #while True:
        t = 0
        for i in track(range(total), description='Detecting....', transient=True):
            ret, frame = cap.read()
            if not ret:
                break
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_list = yolo.detect(image=frame, conf=0.7)
            if face_list != None and len(face_list) > 0:
                start = time.time()
                face = face_list[0]
                resized_face, affine = warp_face_by_landmark(frame, face[1], arcface_128_v2, (256,256))
                
                x1, y1, x2, y2 = map(int, face[0])
                # face_crop = frame[y1:y2, x1:x2]
                # resized_face = cv2.resize(face_crop, (256, 256))
                mask1 = xseg.detect(image=resized_face)
                mask2 = xseg.detect(image=resized_face)
                #output = paste_back(frame, resized_face, mask, affine)
                
                stop = time.time()
                t = t + (stop - start)
                mask1 = (mask1 * 255).clip(0, 255).astype(np.uint8)
                mask1 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
                mask2 = (mask2 * 255).clip(0, 255).astype(np.uint8)
                mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
                combined = cv2.hconcat([resized_face, mask1, mask2])
                
                #output = paste_back(frame, output, mask, affine)
                writer.append_data(combined[..., ::-1])
    
        cap.release()
        writer.close()
        print(f'total time: {t:.4f} sec; total frames: {total}; average time per frame: {t/total:.4f} sec')
        
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
    seg1_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/xseg_1_simplified.onnx'
    seg1_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/xseg_1.onnx'
    seg0_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/occluder.onnx'
    
    yolo = YoloFace(model_path=yolo_path, providers=providers)
    xseg = Occluder(model_path=seg0_path, providers=providers)
    #xseg1 = XSeg(model_path=seg1_path, providers=providers)
    test_image(yolo, xseg)

    
