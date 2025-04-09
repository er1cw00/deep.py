import os
import cv2
import numpy as np
from PIL import Image
from facefusion.modules.gfpgan import GFPGAN
from facefusion.modules.yoloface import YoloFace
from facefusion.utils.mask import create_bbox_mask
from facefusion.utils.affine import ffhq_512, warp_face_by_landmark, paste_back, blend_frame
from app.base.error import Error
from .utils import get_providers_from_device, get_video_writer

class FaceRestore:
    def __init__(self, model_path, device):

        self.providers = get_providers_from_device(device=device)
        self.max_fps = 50
        self.gfpgan_blend = 0.75
        self.face_detect_weight = 0.7
        self.model_path = os.path.join(model_path, "facefusion")
        
        yolo_path = os.path.join(self.model_path,  'yoloface_8n.onnx')
        gfpgan_path = os.path.join(self.model_path, 'gfpgan_1.4.onnx')
        
        self.yolo = YoloFace(model_path=yolo_path, providers=self.providers)
        self.gfpgan = GFPGAN(model_path=gfpgan_path, providers=self.providers)
        
    def process(self, task):
        if task.video:
            return self.process_video(task)
        else:
            return self.process_image(task)
    
    def restore_face(self, image, face_list):
        for index, face in enumerate(face_list):
            if index >= 3:
                break
            landmarks = face[1]
            cropped, affine_matrix = warp_face_by_landmark(image=image, face_landmark_5=landmarks, warp_template=ffhq_512, crop_size=self.gfpgan.input_size)
            box_mask = create_bbox_mask(self.gfpgan.input_size, 0.3, (0,0,0,0))
            crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
            result = self.gfpgan.run(cropped)
            output = paste_back(image, result, crop_mask, affine_matrix)
            if self.gfpgan_blend > 0.01:
                output = blend_frame(image, output, self.gfpgan_blend)
        return output
             
             
    def process_image(self, task):
        task_path = task.get_task_path()
        target_path = os.path.join(task_path, 'target.jpg')
        output_path = os.path.join(task_path, 'output.jpg')

        target = cv2.imread(target_path)
        #target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        
        face_list = self.yolo.detect(image=target, conf=self.face_detect_weight)
        if len(face_list) <= 0:
            return "", Error.NoFaceDetected
        
        output = self.restore_face(target, face_list)
        cv2.imwrite(output_path, output)
        return output_path, Error.OK
    
    def process_video(self, task):
        task_path = task.get_task_path()
        target_path = os.path.join(task_path, 'target.mp4')
        output_path = os.path.join(task_path, 'output.mp4')

        cap = cv2.VideoCapture(target_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
        target_fps = min(self.max_fps, fps) 
        frame_interval = fps / target_fps 
        frame_index = 0
        new_frame_id = 0  # 目标视频的帧编号
        writer = get_video_writer(output_path, target_fps)
        
        while cap.isOpened():
            ret, target = cap.read()
            if not ret:
                break
            if new_frame_id * frame_interval <= frame_index:
                frame = target
                face_list = self.yolo.detect(image=frame, conf=self.face_detect_weight)
                output = self.restore_face(frame, face_list)
                writer.append_data(output[..., ::-1])
                new_frame_id += 1

            frame_index += 1
            
        writer.close()
        cap.release()
        return output_path, Error.OK