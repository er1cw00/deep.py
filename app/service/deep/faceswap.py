import os
import cv2
import imageio
from loguru import logger

from facefusion.modules.inswapper import InSwapper
from facefusion.modules.uniface import UniFace
from facefusion.modules.arcface import ArcFaceW600k
from facefusion.modules.gfpgan import GFPGAN
from facefusion.modules.yoloface import YoloFace
from facefusion.utils.affine import arcface_128_v2, ffhq_512, warp_face_by_landmark, paste_back, blend_frame
from facefusion.facemask import FaceMasker, FaceMaskConfig

from app.base.error import Error
from .utils import get_providers_from_device, get_video_writer, restore_audio

class FaceSwapper:
    def __init__(self, swap_model, model_path, device):
        self.max_fps = 25
        self.dsize = (256,256)
        self.face_detect_weight = 0.6
        self.face_restore_blend = 0.70
        self.device = device
        self.model_path  = os.path.join(model_path, "facefusion")
        self.swap_model = swap_model
        self.providers = get_providers_from_device(self.device)
        if self.swap_model == 'uniface_256':
            uniface_path = os.path.join(self.model_path, 'uniface_256.onnx')
            self.uniface = UniFace(model_path=uniface_path, providers=self.providers)
        elif self.swap_model == 'inswapper_128':
            inswapper_path = os.path.join(self.model_path, 'inswapper_128.onnx')
            arcface_path = os.path.join(self.model_path, 'arcface_w600k_r50.onnx')
            self.inswapper = InSwapper(model_path=inswapper_path, providers=self.providers)
            self.recognizer = ArcFaceW600k(model_path=arcface_path, providers=self.providers)
        
        yolo_path = os.path.join(self.model_path, 'yoloface_8n.onnx')  
        self.yolo = YoloFace(model_path=yolo_path, providers=self.providers)
        
        gfpgan_path = os.path.join(self.model_path, 'gfpgan_1.4.onnx')
        self.gfpgan = GFPGAN(model_path=gfpgan_path, providers=self.providers)
        
        mask_cfg = FaceMaskConfig()
        mask_cfg.bbox = True
        mask_cfg.bbox_blur = 0.3
        mask_cfg.bbox_padding = [0,0,0,0]
        mask_cfg.occlusion = True
        mask_cfg.region = False
        mask_cfg.region_list = [] 
        mask_cfg.model_path = self.model_path
        mask_cfg.providers = self.providers
        self.masker = FaceMasker(mask_cfg)
        
    def process(self, task):
        if task.video:
            return self.process_video(task)
        else:
            return self.process_image(task)
    
    def process_image(self, task):
        task_path = task.get_task_path()
        source_path = os.path.join(task_path, 'source.jpg')
        target_path = os.path.join(task_path, 'target.jpg')
        output_path = os.path.join(task_path, 'output.jpg')
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)
        
        face_list = self.yolo.detect(image=source, conf=self.face_detect_weight, order='large-small')
        if len(face_list) <= 0:
            return "", Error.NoFace
        
        crop_info =  self.yolo.detect(image=target, conf=self.face_detect_weight, order='left-right')
        output = self.swap(source, face_list[0], target=target, crop_info=crop_info)
        cv2.imwrite(output_path, output)
        
        return output_path, Error.OK
        
    def process_video(self, task):
        task_path = task.get_task_path()
        source_path = os.path.join(task_path, 'source.jpg')
        target_path = os.path.join(task_path, 'target.mp4')
        output_path = os.path.join(task_path, 'output.mp4')
        source = cv2.imread(source_path)
        cap = cv2.VideoCapture(target_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
        face_list = self.yolo.detect(image=source, conf=self.face_detect_weight, order='large-small')
        if len(face_list) <= 0:
            return "", Error.NoFace
        

        target_fps = min(self.max_fps, fps)
        frame_interval = fps / target_fps  # 用于均匀采样
        
        max_frame_count = total 
        trim_duration = None
        if task.trim_duration != None :
            count = int(task.trim_duration * target_fps)
            if count < max_frame_count:
                max_frame_count = count
                trim_duration = task.trim_duration
            
        
        logger.info(f"task: {task.task_id}, total: {total}, target_fps: {target_fps}, max_frame_count: {max_frame_count}, trim_duration: {trim_duration}, frame_interval: {frame_interval}")
        
        frame_index = 0
        new_frame_id = 0  # 目标视频的帧编号
        writer = get_video_writer(output_path, target_fps)
        
        while cap.isOpened() and frame_index < max_frame_count:
            ret, target = cap.read()
            if not ret:
                break
            if new_frame_id * frame_interval <= frame_index:
                crop_info = self.yolo.detect(image=target, conf=self.face_detect_weight, order='large-small')
                output = self.swap(source, source_face=face_list[0], target=target, crop_info=crop_info)
                writer.append_data(output[..., ::-1])
                new_frame_id += 1
                
            frame_index += 1
        writer.close()
        cap.release()
       
        err = restore_audio(target_path, output_path, trim_duration)
        return output_path, err 
        
    
    def swap(self, source, source_face, target, crop_info):
        if len(crop_info) == 0:
            return target
        if self.swap_model == 'uniface_256':
            output = self.swap_uniface(source=source, source_crop_info=source_face, target=target, target_crop_info=crop_info[0])
        elif self.swap_model == 'inswapper_128':
            output = self.swap_inswapper(source=source, source_crop_info=source_face, target=target, target_crop_info=crop_info[0])
        return output
    
    def swap_uniface(self, source, source_crop_info, target, target_crop_info):
        source_face, _ = warp_face_by_landmark(source, source_crop_info[1], ffhq_512, self.uniface.source_size)
        target_face, affine = warp_face_by_landmark(target, target_crop_info[1], ffhq_512, self.uniface.target_size)
        
        crop_mask = self.masker.create_mask(target_face)
        output = self.uniface.swap(source_face, target_face)        

        if self.face_restore_blend >= 0.01:
            output = self.gfpgan.run(output)
        
        output = paste_back(target, output, crop_mask, affine)
        
        if self.face_restore_blend >= 0.01:
            output = blend_frame(target, output, self.face_restore_blend)
            
        return output
    
    def swap_inswapper(self, source, source_crop_info, target, target_crop_info):
        embedding = self.recognizer.embedding(image=source, landmarks=source_crop_info[1])
        target_face, affine = warp_face_by_landmark(target, target_crop_info[1], arcface_128_v2, self.dsize)
        
        crop_mask = self.masker.create_mask(target_face)
        swapped = self.inswapper.swap(embedding[0], target_face)
        swapped = paste_back(target, swapped, crop_mask, affine)
        
        if self.face_restore_blend >= 0.01:
            target_face, affine = warp_face_by_landmark(image=swapped, face_landmark_5=target_crop_info[1], warp_template=ffhq_512, crop_size=self.gfpgan.input_size)
            crop_mask = self.masker.create_mask(target_face)
            output = self.gfpgan.run(target_face)
            output = paste_back(swapped, output, crop_mask, affine)
            output = blend_frame(swapped, output, self.face_restore_blend)
        else:
            output = swapped
            
        return output
             
