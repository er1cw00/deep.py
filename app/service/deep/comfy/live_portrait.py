import os
import cv2
import sys
import json
import random
import argparse


from app.base.error import Error
from app.deepfake.utils import get_providers_from_device, get_video_writer
from app.deepfake.live_portrait.config.crop_config import CropConfig
from app.deepfake.live_portrait.config.inference_config import InferenceConfig
from app.deepfake.live_portrait.human_cropper import HumanCropper
from app.deepfake.live_portrait.human_pipeline import HumanPipeline

# from app.service.deep.utils import add_tbox_path_to_sys_path, add_comfy_path_to_sys_path


parser = argparse.ArgumentParser(
    description="A converted ComfyUI workflow. Required inputs listed below. Values passed should be valid JSON (assumes string if not valid JSON)."
)

parser.add_argument(
    "--device",
    "-d",
    default='CPU',
    help="Device, should be CUDA, CoreML, ROCM",
)

parser.add_argument(
    "--model_path",
    "-m",
    default=None,
    help="Where to load LivePortrail model",
)

parser.add_argument(
    "--task_path",
    "-p",
    default=None,
    help="The source portrait photo to anime . should be a file path",
)



os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


    
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
        
        cropConfig = CropConfig(
            insightface_root = os.path.join(self.model_path, "insightface"),
            landmark_ckpt_path = os.path.join(self.model_path, 'liveportrait/landmark.onnx'),
            xpose_ckpt_path = os.path.join(self.model_path, 'liveportrait/animal/xpose.pth'),
        )

        self.cropper = HumanCropper(crop_cfg=cropConfig, providers=providers)
        
        inferConfig = InferenceConfig(
            checkpoint_F = os.path.join(self.model_path, 'liveportrait/appearance_feature_extractor.safetensors'), 
            checkpoint_M = os.path.join(self.model_path,'liveportrait/motion_extractor.safetensors'),
            checkpoint_G = os.path.join(self.model_path,'liveportrait/spade_generator.safetensors'),
            checkpoint_W = os.path.join(self.model_path,'liveportrait/warping_module.safetensors'),
            checkpoint_S = os.path.join(self.model_path,'liveportrait/stitching_retargeting_module.safetensors'),
            
            checkpoint_F_animal = os.path.join(self.model_path, 'liveportrait/animal/appearance_feature_extractor.safetensors'),
            checkpoint_M_animal = os.path.join(self.model_path, 'liveportrait/animal/motion_extractor.safetensors'),
            checkpoint_G_animal = os.path.join(self.model_path, 'liveportrait/animal/spade_generator.safetensors'),
            checkpoint_W_animal = os.path.join(self.model_path, 'liveportrait/animal/warping_module.safetensors'),
            checkpoint_S_animal = os.path.join(self.model_path, 'liveportrait/animal/stitching_retargeting_module.safetensors'),
        )
        inferConfig.device = self.device

        self.pipeline = HumanPipeline(inference_cfg=inferConfig)   
        
    def process(self, task_path):
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
        


def main(*func_args, **func_kwargs):
    args = parser.parse_args()
    device = args.device.lower()
    if device not in ["cuda", "mps", "rocm"]:
        if device == "coreml":
            device = "mps"
        else:
            print(f'deivce ({device}) unsupport, fallback to CPU')
            device = 'cpu'
            
    print(f'device ({device})')

    if args.task_path is None or os.path.isdir(args.task_path) == False:
        print(f'task_path: {args.task_path} not exist')
        sys.exit(1001) 
        
    if args.model_path is None or os.path.isdir(args.model_path) == False:
        print(f'model_path: {args.model_path} not exist')
        sys.exit(1001) 
        
    print(f'device: {device}, model_path: {args.model_path}, task_path: {args.task_path}')
    liveportrait = LivePortrait(args.model_path, device)
    liveportrait.process(args.task_path)
    
if __name__ == "__main__":
    main()


# python -m app.service.deep.comfy.live_portrait \
#     --device "mps" \
#     --task_path "/Users/wadahana/workspace/AI/tbox.ai/deep.py/task/20250505/e99e58e983130db43bcac9fa0948e27d/" \
#     --model_path "/Users/wadahana/workspace/AI/sd/ComfyUI/models"

