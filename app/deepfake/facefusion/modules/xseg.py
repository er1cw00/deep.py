#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import numpy as np
import onnxruntime

class XSeg:
    def __init__(self, model_path, providers):
        
        self.session  = onnxruntime.InferenceSession(model_path, providers=providers)
        print(f'XSeg providers:{providers}; current providers: {self.session.get_providers()}') 
        inputs = self.session.get_inputs()
        for input in inputs:
            print(f'inputs name: {input.name}, shape: {input.shape}')
            
        self.input_size = (inputs[0].shape[1], inputs[0].shape[2])
        self.input_name = inputs[0].name
        self.affine = False
    
    def pre_process(self, image):
        img = cv2.resize(image, self.input_size)
        img = np.expand_dims(img, axis = 0).astype(np.float32) / 255
        img = img.transpose(0, 1, 2, 3)
        return img
    
    def post_process(self, output, height, width):
        mask = output.transpose(0, 1, 2).clip(0, 1).astype(np.float32)
        mask = cv2.resize(mask, (width, height))
        mask = (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
        return mask
    
    def detect(self, image):
        height, width = image.shape[0], image.shape[1]
        img = self.pre_process(image)
        outputs = self.session.run(None, {self.input_name: img})
        output = outputs[0][0]
        output = self.post_process(output, height, width)
        
        #print('infer time:',timeit.default_timer()-t)  
        return output
        
if __name__ == "__main__":
    from app.deepfake.facefusion.modules.yoloface import YoloFace
    from app.deepfake.facefusion.modules.occluder import Occluder
    from app.deepfake.facefusion.utils.mask import overlay_mask_on_face
    from app.deepfake.facefusion.utils.affine import arcface_128_v2, ffhq_512, warp_face_by_landmark_5, paste_back, blend_frame
    from app.deepfake.utils.video import get_video_writer
    from app.deepfake.utils.timer import Timer
    from rich.progress import track

    import imageio
    import time
    import cv2
    import os
    

    os.environ["ORT_LOGGING_LEVEL"] = "VERBOSE"
        
    def test_image(yolo, xseg, input_path, output_path):
        image = cv2.imread(input_path)
        face_list = yolo.get(image=image, order='best-worst')
        
        if face_list != None and len(face_list) > 0:
            face = face_list[0]
            resized_face, affine = warp_face_by_landmark_5(image, face.landmark_5, arcface_128_v2, (256,256))
                
            mask = xseg.detect(image=resized_face)
            mask = (mask * 255).clip(0, 255).astype(np.uint8)

            output = overlay_mask_on_face(resized_face, mask, alpha=0.5, color=(0, 0, 255))
            
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = cv2.hconcat([resized_face, output, mask])
            cv2.imwrite(output_path, combined)
        
        
    def test_video(yolo, xseg, input_path, output_path):

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
        writer = get_video_writer(output_path, fps)

        t = Timer()
        for i in track(range(total), description='Detecting....', transient=True):
            ret, frame = cap.read()
            if not ret:
                break
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_list = yolo.get(image=frame, order='best-worst')
            if face_list != None and len(face_list) > 0:

                face = face_list[0]
                resized_face, affine = warp_face_by_landmark_5(frame, face.landmark_5, arcface_128_v2, (256,256))
                
                t.tic()
                mask = xseg.detect(image=resized_face)
                t.toc()
 
                mask = (mask * 255).clip(0, 255).astype(np.uint8)
                #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask = (mask * 255).clip(0, 255).astype(np.uint8)
                output = overlay_mask_on_face(resized_face, mask, alpha=0.5, color=(0, 0, 255))
                
                combined = cv2.hconcat([resized_face, output])
                
                #output = paste_back(frame, output, mask, affine)
                writer.append_data(combined[..., ::-1])
    
        cap.release()
        writer.close()
        t.show('face occluder time')
        
        
    # trt_options = {
    #     #"trt_fp16_enabled": True,          # 启用 FP16 加速（可选）
    #     "trt_engine_cache_enable": True,     # 启用引擎缓存
    #     "trt_engine_cache_path": "/data/trt_cache",  # 缓存文件存储路径
    #     "trt_timing_cache_enable": True,
    #     "trt_timing_cache_path": "/data/trt_cache",
    #     #'trt_builder_optimization_level': 5
    #     "trt_max_workspace_size": 1 << 30  # 可选：设置最大显存工作空间（单位：字节）
    # }
    #providers = [('TensorrtExecutionProvider', trt_options)]
    providers=['CPUExecutionProvider', 'CoreMLExecutionProvider']
    yolo_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/yoloface_8n.onnx'
    seg_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/dfl_xseg.onnx'
    #seg_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/xseg_1_simplified.onnx'
    #seg_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/xseg_1.onnx'
    #seg_path = "/Users/wadahana/workspace/AI/tbox.ai/temp/XSegNet2onnx/xseg_sim_2.onnx"
    #seg_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion/occluder.onnx'
    
    # yolo_path = '/ff/.assets/models/yoloface_8n.onnx'
    # seg_path = '/ff/.assets/models/face_occluder.onnx'
    yolo = YoloFace(model_path=yolo_path, providers=providers)
    #xseg = Occluder(model_path=seg0_path, providers=providers)
    xseg = XSeg(model_path=seg_path, providers=providers)
   
    # input_path = "/data/task/20250505/1e42b87f42559936a9447be1bce59165/target.mp4"
    # output_path = "/data/task/20250505/1e42b87f42559936a9447be1bce59165/output_mask.mp4" 
    
    input_path = "/Users/wadahana/Desktop/blowjob.jpg" #sis/tbox/face/SongY2.jpg"
    #input_path = "/Users/wadahana/workspace/AI/tbox.ai/data/faceswap/task/20240410/914561475fb68155b171c178da847063/target.jpg"
    output_path = "/Users/wadahana/Desktop/sis/mask00.jpg"
    test_image(yolo=yolo, xseg=xseg, input_path=input_path, output_path=output_path)
    
    #providers=['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider']
   
#tbox/photo/Test2.jpg