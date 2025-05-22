import os 
import time
from pydantic import BaseModel
from typing import Optional, List
from .task import ObjectKeys, TaskType, TaskState,TaskInfo, FaceSwapConfig
from app.base.file import is_file, is_image, is_video, is_directory
from app.deepfake.utils.face import face_analyser_orders

# curl -v -X POST "http://localhost:7860/facefusion/frame_process" -H 'Content-Type: application/json' -d '{ 
#     "sources": ["file:///home/eric/workspace/AI/data/deep/task/20250505/1e42b87f42559936a9447be1bce59165/source.jpg"],
#     "target": "file:///home/eric/workspace/AI/data/deep/task/20250505/1e42b87f42559936a9447be1bce59165/target.mp4",
#     "output": "file:///home/eric/workspace/AI/data/deep/task/20250505/1e42b87f42559936a9447be1bce59165/output1.mp4",
#     "providers": ["cuda", "coreml"],
#     "processors": [
#         {"name": "face_swapper", "model": "inswapper_128", "blend": 90.0},
#         {"name": "face_enhancer", "model": "gfpgan_1.4", "blend": 70.0}   
#     ],
#     "face_selector": "one",
#     "face_refer_distance": 0.6, 
#     "face_mask_types": ["box","occlusion"],
#     "face_mask_blur": 0.3,
#     "face_analyse_order": "left-right",
#     "face_detect_model": "yoloface",
#     "face_detect_score": 0.6,
#     "face_detect_size": "640x640",
#     "skip_download": true,
#     "watermark": false
# }'

class FrameProcessor(BaseModel):
    name: str
    model: str
    blend: Optional[float]

class FaceSwapRequest(BaseModel):
    sources: List[str]
    target: str
    output: Optional[str] = None
    providers: List[str]
    processors: List[FrameProcessor]
    face_selector: str
    face_refer_distance: float
    face_mask_types: List[str]
    face_mask_blur: float
    face_analyse_order: str
    face_analyse_age: Optional[str] = None
    face_analyse_gender: Optional[str] = None
    face_detect_model: str
    face_detect_score: float
    face_detect_size: str
    skip_download: bool
    trim_time_start: Optional[int] = None
    trim_time_end: Optional[int] = None
    watermark: Optional[bool]=False


    def toTaskInfo(self):
        source = ""
        target = ""
        output = ""
        video = False
        if self.sources is not None and len(self.sources) > 0:
            source = self.sources[0]
            if source.startswith('file://'):
                source = source[7:]
        if not is_image(source):
            return (None, 'no source images')
        
        if self.target.startswith('file://'):
            target = self.target[7:]
            if not is_file(target):
                return (False, 'No target image or video')
        if is_image(target):
            video = False
        elif is_video(target):
            video = True
        else:
            return (None, "unknown target file")
        
        if self.output.startswith('file://'):
            output = self.output[7:]  
        if not is_directory(os.path.dirname(output)):
            return (None, "output dir not exist")
        task_path = os.path.dirname(target)
        task_id = os.path.basename(task_path)
        
        trim_duration = None
        if self.trim_time_start is not None and self.trim_time_end is not None:
            trim_duration = self.trim_time_end - self.trim_time_start
            if trim_duration <= 0:
                trim_duration = None
            
        
        config = FaceSwapConfig()
        config.face_detect_model = self.face_detect_model  #yoloface only
        if self.face_detect_score < 0 and self.face_detect_score > 1:
            config.face_detect_score = self.face_detect_score
        if self.watermark:
            config.watermark = self.watermark
        if self.face_analyse_order in face_analyser_orders:
            config.face_order = self.face_analyse_order
        
        if 'box' in self.face_mask_types:
            config.face_mask_box = True
        if 'occlusion' in self.face_mask_types:
            config.face_mask_occlusion = True
        if self.face_mask_blur >0 and self.face_mask_blur <= 1:
            config.face_mask_blur = self.face_mask_blur
        
        task = TaskInfo(
            uid             = 'tg_6412449819',
            task_id         = task_id,
            task_type       = TaskType.FaceSwap,  # 假设1代表换脸任务
            task_state      = TaskState.InQueue,  # 初始状态
            priority        = 0,
            credit          = 1,
            start_time      = int(time.time()),
            update_time     = int(time.time()),
            video           = video,
            duration        = None,
            trim_duration   = trim_duration,
            format          = 'mp4' if video else 'jpg',
            obj_keys=ObjectKeys(
                source=source,
                target=target,
                output=output
            )
        )
        task._task_path = task_path
        task._faceswap_config = config
        return (task, 'OK')
    
class FaceSwapResponse(BaseModel):
    output: Optional[str] = None
    detail: Optional[str] = None
    
