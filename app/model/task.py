
import os
from typing import Optional, List
from pydantic import BaseModel, PrivateAttr
from datetime import datetime
from app.base.config import config

class TaskType:
    Unknown         = 0                 # 原TaskNone（避免与Python关键字None冲突）
    Dummy           = 1                 # 原TaskDummy
    FaceSwap        = 3                 # 原TaskFaceSwap
    MinComfyTask    = 1000              # 范围标记保持原名
    Upscale         = 1000              # 原TaskUpscale
    Rmbg            = 1001              # 原TaskRMBG（RFC规范缩写通常全大写，这里按驼峰处理）
    Anime           = 1002              # 原TaskAnime
    LivePortrait    = 1003              # 原TaskLivePortrait
    Txt2Img         = 1004              # 原TaskTxt2Img
    FaceRestore     = 1005              # 原TaskFaceRestore
    RedrawBg        = 1006              # 原TaskRedrawBG（BG按驼峰处理为Bg）
    FaceSwap2       = 1007              # 原TaskFaceSwap
    MaxComfyTask    = 1007              # 范围标记保持原名

def is_comfy_task(t) -> bool:
    if t >= TaskType.MinComfyTask and t <= TaskType.MaxComfyTask:
        return True
    return False 

def get_task_type_name(t) -> str:
    if t == TaskType.Dummy: 
        return 'dummy'
    elif t == TaskType.Upscale: 
        return 'unscale'
    elif t == TaskType.Rmbg: 
        return 'rmbg'
    elif t == TaskType.Anime: 
        return 'anime'
    elif t == TaskType.LivePortrait: 
        return 'liveportrait'
    elif t == TaskType.Txt2Img: 
        return 'txt2img'
    elif t == TaskType.FaceRestore: 
        return 'facerestore'
    elif t == TaskType.FaceSwap2: 
        return 'faceswap2'
    return 'none'
    
class TaskState:
    Unknown     = 0              # 原TaskStateNone（避免与Python关键字None冲突）
    InQueue     = 1              # 原TaskStateInQueue
    InProgress  = 2              # 原TaskStateInProgress
    Success     = 3              # 原TaskStateSuccess
    Fail        = 4              # 原TaskStateFail
    Cancel      = 5              # 原TaskStateCancel
    Deleted     = 6              # 原TaskStateDeleted
    ReplyFail   = 7              # 原TaskStateReplyFail

class Priority:
    Free = 0
    Vip = 1
    
    
class LivePortraitInfo(BaseModel):
    type: str


class UpscaleTaskInfo(BaseModel):
    scale: int


class LoRAInfo(BaseModel):
    name: str
    strength: float


class Txt2ImgInfo(BaseModel):
    width: int
    height: int
    checkpoint: Optional[str] = None
    lora: Optional[LoRAInfo] = None
    positive_prompt: str
    negative_prompt: str
    face_detailer: Optional[bool] = None


class RedrawBGInfo(BaseModel):
    checkpoint: Optional[str] = None
    positive_prompt: str
    negative_prompt: str
    denoise: float


class ObjectKeys(BaseModel):
    source: Optional[str] = None
    target: Optional[str] = None
    output: Optional[str] = None


class FaceSwapConfig(BaseModel):
    face_detect_model: str = 'yoloface'
    face_detect_score: float = '0.5'
    face_enhance_blend: float = 0.7
    face_mask_box: bool = True
    face_mask_occlusion: bool = False
    face_mask_blur: float = 0.3
    face_order: str = 'left-right'
    watermark: bool = False
    
class TaskInfo(BaseModel):
    uid: str
    task_id: str
    task_type: int
    task_state: int
    priority: int
    credit: int
    start_time: int
    update_time: int
    video: bool
    duration: Optional[int] = None
    trim_duration: Optional[int] = None
    format: str
    obj_keys: ObjectKeys
    upscale: Optional[UpscaleTaskInfo] = None
    live_portrait: Optional[LivePortraitInfo] = None
    txt2img: Optional[Txt2ImgInfo] = None
    watermark: Optional[bool] = False
    _task_path: str = PrivateAttr(default=None)
    _faceswap_config: Optional[FaceSwapConfig] = None
    
    def get_task_path(self):
        if self._task_path is None:
            today = datetime.today().strftime("%Y%m%d")
            self._task_path = os.path.join(config.get("deep.task_path"), today, self.task_id)
        return self._task_path