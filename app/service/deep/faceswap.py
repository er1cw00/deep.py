import os
import cv2
import imageio
import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import Literal, List, Tuple
from dataclasses import dataclass, field 
from app.model.task import TaskType, TaskInfo 
from app.base.printable import Printable
from app.deepfake.utils.face import convert_face_landmark_68_to_5, draw_landmarks
from app.deepfake.facefusion.modules.inswapper import InSwapper
from app.deepfake.facefusion.modules.uniface import UniFace
from app.deepfake.facefusion.modules.arcface import ArcFaceW600k
from app.deepfake.facefusion.modules.gfpgan import GFPGAN
from app.deepfake.facefusion.modules.yoloface import YoloFace
from app.deepfake.facefusion.modules.xseg import XSeg
from app.deepfake.facefusion.modules.occluder import Occluder
from app.deepfake.facefusion.modules.resnet34 import Resnet34
from app.deepfake.facefusion.modules.face_landmark import FaceLandmark_2dFan
from app.deepfake.facefusion.utils.mask import create_bbox_mask, FaceMaskRegion, FaceMaskRegionMap, FaceMaskAllRegion
from app.deepfake.facefusion.utils.affine import arcface_128_v2, ffhq_512, warp_face_by_landmark_5, paste_back, blend_frame


from app.base.error import Error
from app.deepfake.utils import get_providers_from_device, get_video_writer, restore_audio

# mask_cfg = FaceMaskConfig()
# mask_cfg.bbox = True
        # mask_cfg.bbox_blur = 0.3
        # mask_cfg.bbox_padding = [0,0,0,0]
        # mask_cfg.occlusion = True
        # mask_cfg.region = False
        # mask_cfg.region_list = [] 
        # mask_cfg.model_path = self.model_path
        # mask_cfg.providers = self.providers
        # self.masker = FaceMasker(mask_cfg)

@dataclass(repr=False)  # use repr from Printable
class FaceMaskConfig(Printable):
    bbox: bool = True
    occlusion: bool = False
    region: bool = False
    bbox_blur: float = 0.3
    bbox_padding: Tuple[int, int, int, int] = (0,0,0,0)
    region_list: List[FaceMaskRegion] = field( default_factory=lambda: FaceMaskAllRegion.copy() )


class FaceSwapper:
    def __init__(self, model_path, device,  **kwargs):
        self.max_fps = 25
        self.dsize = (256,256)
        self.device = device
        self.model_path = model_path
               
        self.landmark_threshold = kwargs.get('landmark_threshold', 0.5) 
        self.enhance_blend = kwargs.get('enhance_blend', 0.8) 
        self.debug = kwargs.get('debug', False) 
        self.show_progress =  kwargs.get('show_progress', False) 
        self.mask_config = kwargs.get('mask_config', None)
        self.max_fps = kwargs.get('max_fps', 50) 
        self.debug = kwargs.get('debug', False) 
        
        if self.mask_config == None:
            self.mask_config = FaceMaskConfig(bbox=True)
            
        self.load()
        
    def load(self):
        yolo_path       = os.path.join(self.model_path, 'yoloface_8n.onnx')  
        gfpgan_path     = os.path.join(self.model_path, 'gfpgan_1.4.onnx')  
        fan4_path       = os.path.join(self.model_path, '2dfan4.onnx')  
        inswapper_path  = os.path.join(self.model_path, 'inswapper_128.onnx')  
        w600k_path      = os.path.join(self.model_path, 'arcface_w600k_r50.onnx')  
        occluder_path   = os.path.join(self.model_path, 'occluder.onnx')  
        resnet_path     = os.path.join(self.model_path, 'bisenet_resnet_34.onnx')  
        
        providers = get_providers_from_device(self.device)
        
        self.detector   = YoloFace(model_path=yolo_path, providers=providers)
        self.landmarker = FaceLandmark_2dFan(model_path=fan4_path, providers=providers)
        self.recognizer = ArcFaceW600k(model_path=w600k_path, providers=providers)
        self.swapper    = InSwapper(model_path=inswapper_path, providers=providers)
        self.enhancer   = GFPGAN(model_path=gfpgan_path, providers=providers)

        if self.mask_config.occlusion == True:
            self.occluder = Occluder(model_path=occluder_path, providers=providers)
        if self.mask_config.region == True:
            self.resnet = Resnet34(model_path=resnet_path, providers=providers)
    
    def get_source_face(self, image):
        face_list = self.detector.get(image=image, order='large-small')
        if len(face_list) <= 0:
            return "", Error.NoFace
        face = face_list[0]
        if self.landmark_threshold > 0:
            face = self.calibration_face_landmark(image, face)
        face.embedding, face._normed_embedding = self.recognizer.embedding(image=image, landmarks=face.landmark_5)
        return face, Error.OK
    
    def calibration_face_landmark(self, image, face):
        landmark_68, landmark_68_score  = self.landmarker.get(image, face.bbox)
        if landmark_68_score >= self.landmark_threshold:
            landmark_5 = convert_face_landmark_68_to_5(landmark_68)
        else: 
            landmark_5 = face.landmark_5
            
        face.landmark_5 = landmark_5
        face.landmark_68 = landmark_68
        return face
    
    def create_mask(self, crop_face):
        mask_list = []
        if self.mask_config.bbox == True:
            bbox_mask = create_bbox_mask(crop_face.shape[:2][::-1], self.mask_config.bbox_blur, self.mask_config.bbox_padding)
            mask_list.append(bbox_mask)
        if self.mask_config.occlusion == True:
            occlusion_mask = self.occluder.detect(image=crop_face)
            mask_list.append(occlusion_mask)
        if self.mask_config.region == True:
            region_mask = self.resnet.detect(crop_face, self.mask_config.region_list)
            mask_list.append(region_mask)
        crop_mask = np.minimum.reduce(mask_list).clip(0, 1)  
        return crop_mask
    
    def process(self, task):
        if task.task_type == TaskType.FaceRestore:
            if task.video:
                return self.enhance_video(task)
            else:
                return self.enhance_image(task)
        elif task.task_type == TaskType.FaceSwap:
            if task.video:
                return self.swap_video(task)
            else:
                return self.swap_image(task)
    
    def swap_image(self, task):
        task_path = task.get_task_path()
        source_path = os.path.join(task_path, 'source.jpg')
        target_path = os.path.join(task_path, 'target.jpg')
        output_path = os.path.join(task_path, 'output.jpg')
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)
        print(f'source_path: {source_path}')
        
        if source is None or target is None:
            return "", Error.NoFace 
        
        source_face, err = self.get_source_face(image=source)
        if err != Error.OK:
            return "", Error.NoFace
        
        target_faces =  self.detector.get(image=target, order='left-right')
        
        output = self.swap_face(source, source_face, target=target, target_faces=target_faces)
        cv2.imwrite(output_path, output)
        
        return output_path, Error.OK
        
    def swap_video(self, task):
        task_path = task.get_task_path()
        source_path = os.path.join(task_path, 'source.jpg')
        target_path = os.path.join(task_path, 'target.mp4')
        output_path = os.path.join(task_path, 'output.mp4')
        
        source = cv2.imread(source_path)
        cap = cv2.VideoCapture(target_path)
        if source is None or cap is None or not cap.isOpened():
            return "", Error.NoFace 
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
        source_face, err = self.get_source_face(image=source)
        if err != Error.OK:
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
        
        with tqdm(total=max_frame_count, desc='FaceSwap', unit='frame', disable=(not self.show_progress)) as progress:
            while cap.isOpened() and frame_index < max_frame_count:
                ret, target = cap.read()
                if not ret:
                    break
                if new_frame_id * frame_interval <= frame_index:
                    target_faces = self.detector.get(image=target, order='left-right')
                    output = self.swap_face(source=source, source_face=source_face, target=target, target_faces=target_faces)
                    writer.append_data(output[..., ::-1])
                    new_frame_id += 1
                    
                frame_index += 1
                progress.update()
                
        writer.close()
        cap.release()
       
        err = restore_audio(target_path, output_path, trim_duration)
        return output_path, err 
        
    def swap_face(self, source, source_face, target, target_faces):
        if len(target_faces) == 0:
            return target

        target_face = self.calibration_face_landmark(target, target_faces[0])
        output = self.inswapper(source=source, source_face=source_face, target=target, target_face=target_face)
        return output
    
    def inswapper(self, source, source_face, target, target_face):
        embedding = source_face.embedding
        target_face_image, affine = warp_face_by_landmark_5(target, target_face.landmark_5, arcface_128_v2, self.dsize)
        
        crop_mask = self.create_mask(target_face_image)
        swapped = self.swapper.swap(embedding, target_face_image)
        swapped = paste_back(target, swapped, crop_mask, affine)
        
        if self.enhance_blend >= 0.01:
            swapped_face_image, affine = warp_face_by_landmark_5(image=swapped, face_landmark_5=target_face.landmark_5, warp_template=ffhq_512, crop_size=self.enhancer.input_size)
            crop_mask = self.create_mask(swapped_face_image)
            output = self.enhancer.run(swapped_face_image)
            output = paste_back(swapped, output, crop_mask, affine)
            output = blend_frame(swapped, output, self.enhance_blend)
        else:
            output = swapped
            
        return output
             
    def enhance_image(self, task):
        task_path = task.get_task_path()
        target_path = os.path.join(task_path, 'target.jpg')
        output_path = os.path.join(task_path, 'output.jpg')

        target = cv2.imread(target_path)
        
        face_list = self.detector.get(image=target, order='best-worst')
        if len(face_list) <= 0:
            return "", Error.NoFace
        
        output = self.restore_face(target, face_list)
        cv2.imwrite(output_path, output)
        return output_path, Error.OK
    
    def enhance_video(self, task):
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
        
        with tqdm(total=total, desc='FaceRestore', unit='frame', disable=(not self.show_progress)) as progress:
            while cap.isOpened():
                ret, target = cap.read()
                if not ret:
                    break
                if new_frame_id * frame_interval <= frame_index:
                    frame = target
                    face_list = self.detector.get(image=target, order='best-worst')
                    output = self.restore_face(frame, face_list)
                    writer.append_data(output[..., ::-1])
                    new_frame_id += 1

                frame_index += 1
                progress.update()

            
        writer.close()
        cap.release()
        return output_path, Error.OK
    
    def restore_face(self, image, face_list):
        for index, face in enumerate(face_list):
            if index >= 4:
                break
            face = self.calibration_face_landmark(image, face)

            
            cropped, affine_matrix = warp_face_by_landmark_5(image=image, face_landmark_5=face.landmark_5, warp_template=ffhq_512, crop_size=self.enhancer.input_size)
            box_mask = create_bbox_mask(self.enhancer.input_size, 0.3, (0,0,0,0))
            crop_mask = np.minimum.reduce([box_mask]).clip(0, 1)
            result = self.enhancer.run(cropped)
            output = paste_back(image, result, crop_mask, affine_matrix)
            if self.enhance_blend > 0.01:
                output = blend_frame(image, output, self.enhance_blend)
            
            if self.debug:
                x1, y1, x2, y2 = map(int, face.bbox)
                output = draw_landmarks(output, face.landmark_5)
                cv2.rectangle(output, (x1,y1), (x2,y2), (255, 0, 0), 1)

        return output
    
    # def swap_uniface(self, source, source_face, target, target_face):
    #     source_face_image, _ = warp_face_by_landmark_5(source, source_face.landmark_5, ffhq_512, self.uniface.source_size)
    #     target_face_image, affine = warp_face_by_landmark_5(target, target_face.landmark_5, ffhq_512, self.uniface.target_size)
        
    #     crop_mask = self.masker.create_mask(target_face_image)
    #     output = self.uniface.swap(source_face_image, target_face_image)        

    #     if self.face_restore_blend >= 0.01:
    #         output = self.gfpgan.run(output)
        
    #     output = paste_back(target, output, crop_mask, affine)
        
    #     if self.face_restore_blend >= 0.01:
    #         output = blend_frame(target, output, self.face_restore_blend)
            
    #     return output