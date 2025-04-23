import os
import cv2
from liveportrait.config.crop_config import CropConfig
from liveportrait.config.inference_config import InferenceConfig
from liveportrait.human_cropper import HumanCropper
from liveportrait.human_pipeline import HumanPipeline

from app.base.error import Error
from .utils import get_providers_from_device, get_video_writer


class LivePortrait:
    def __init__(self, model_path, device):
        self.device = device
        self.face_detect_weight = 0.65
        self.model_path = model_path
        self.max_fps = 30
        if device == "mps":
            providers = ["CPUExecutionProvider"]
        else:
            providers = get_providers_from_device(device)
        
        cropConfig = CropConfig()
        cropConfig.insightface_root = os.path.join(self.model_path, "insightface")
        cropConfig.landmark_ckpt_path = os.path.join(self.model_path, 'liveportrait/landmark.onnx')

        self.cropper = HumanCropper(crop_cfg=cropConfig, providers=providers)
        
        inferConfig = InferenceConfig()
        inferConfig.device = self.device
        
        inferConfig.checkpoint_F = os.path.join(self.model_path, 'liveportrait/appearance_feature_extractor.safetensors') 
        inferConfig.checkpoint_M = os.path.join(self.model_path,'liveportrait/motion_extractor.safetensors')
        inferConfig.checkpoint_G = os.path.join(self.model_path,'liveportrait/spade_generator.safetensors')
        inferConfig.checkpoint_W = os.path.join(self.model_path,'liveportrait/warping_module.safetensors')
        inferConfig.checkpoint_S = os.path.join(self.model_path,'liveportrait/stitching_retargeting_module.safetensors')

        self.pipeline = HumanPipeline(inference_cfg=inferConfig)   
        
    def process(self, task):
        task_path = task.get_task_path()
        source_path = os.path.join(task_path, 'target.jpg')
        driving_path = os.path.join(task_path, 'source.mp4')
        output_path = os.path.join(task_path, 'output.mp4')
        
        source = cv2.imread(source_path)
        cap = cv2.VideoCapture(driving_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
        
        target_fps = min(self.max_fps, fps)
        frame_interval = fps / target_fps  # 用于均匀采样
        frame_index = 0
        new_frame_id = 0  # 目标视频的帧编号
        writer = get_video_writer(output_path, target_fps)
        
        source_crop_info = self.cropper.crop_source([source])
        if source_crop_info['frame_crop_lst'] == None or len(source_crop_info['frame_crop_lst']) == 0 :
            return "", Error.NoFace
        
        frames = []
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if new_frame_id * frame_interval <= frame_index:
                if i % 600 == 0 and len(frames) > 0:
                    self.do_process(source, source_crop_info, target_fps, frames, writer)
                    frames = []
                frames.append(frame)
            frame_index += 1
        
        if len(frames) > 0:
            self.do_process(source, source_crop_info, target_fps, frames, writer)
        
        writer.close()
        cap.release()
        return output_path, Error.OK
        
    def do_process(self, source, source_crop_info, fps, driving_frames, writer):
        driving_crop_info = self.cropper.crop_driving(driving_frames)
        driving_template = self.pipeline.calc_driving_template(fps=fps, 
                                                            source_rgb_lst=[source], 
                                                            source_crop_info=source_crop_info, 
                                                            driving_rgb_lst=driving_frames, 
                                                            driving_crop_info=driving_crop_info)
        result = self.pipeline.animate(fps=fps, 
                                    source_rgb_lst=[source], 
                                    source_crop_info=source_crop_info, 
                                    driving_template=driving_template)
        for _, image in enumerate(result):
            writer.append_data(image[..., ::-1])
        