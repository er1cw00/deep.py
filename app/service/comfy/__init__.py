import os
import sys
import json
import importlib.util
import torch
from app.model.task import TaskState, TaskType 
from app.base.logger import logger
from app.base.config import config
from .utils import add_tbox_path_to_sys_path, add_comfy_path_to_sys_path

class Comfy:
    def __init__(self):        
        self._swapper = None
        self._rmbg = None
        self._liveportrait = None
        self._restore = None  
        
        self.comfy_path = None
        self.tbox_path = None
        
    def init(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.version.hip is not None and torch.version.hip != '':
            self.device = "rocm"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.comfy_path = config.get('comfy_path')
        self.tbox_path = os.path.join(self.comfy_path, "custom_nodes/ComfyUI-tbox/src")
        if not os.path.exists(self.comfy_path):
            logger.error(f"ComfyUI path not found: {self.comfy_path}")
            return False
        if not os.path.exists(self.tbox_path):
            logger.error(f"ComfyUI tbox path not found: {self.tbox_path}")
            return False
        
        logger.info(f"ComfyUI Path: {self.comfy_path}")
        logger.info(f"ComfyUI tbox Path: {self.tbox_path}")
        logger.info(f"Device: {self.device}")
        # Add paths to sys.path
        add_comfy_path_to_sys_path(self.comfy_path)
        add_tbox_path_to_sys_path(self.tbox_path)
        
        import folder_paths
        self.model_path = folder_paths.models_dir
        logger.info(f"Model Path: {self.model_path}")
        
        
    def faceswap(self, task):
        from .faceswap import FaceSwapper
        if self._swapper == None:
            self._swapper = FaceSwapper('inswapper_128', self.model_path, self.device)
        return self._swapper.process(task)
    
    def liveportrait(self, task):
        print(f'task =  {task.json()}')
        
        from .liveportrait import LivePortrait
        if self._liveportrait == None:
            self._liveportrait = LivePortrait(self.model_path, self.device)
        return self._liveportrait.process(task)
    
    def rmbg(self, task):
        from .rmbg import RMBG
        if self._rmbg == None:
            
            self._rmbg = RMBG(self.model_path, self.device)
        return self._rmbg.process(task)
    
    def restore(self, task):
        from .facerestore import FaceRestore
        if self._restore == None:
            self._restore = FaceRestore(self.model_path, self.device)
        return self._restore.process(task)
    


comfy = Comfy()


# if __name__ == '__main__':
#     config.init("../../deep.yaml")
#     comfy.init()
#     task = {
#         'task_id': '0eeb9e938dbfaf1a5914ef5d6ef27496',
#         'task_path': '/Users/wadahana/Desktop/test',
#         'type': TaskType.LivePortrait,
#         'state': TaskState.InProgress,        
#     }
    