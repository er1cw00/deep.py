import os
import sys
import json
import torch
import subprocess
import importlib.util
from loguru import logger
from app.model.task import TaskState, TaskType 
from app.base.config import config
from app.base.error import Error
from .utils import add_tbox_path_to_sys_path, add_comfy_path_to_sys_path

class Deep:
    def __init__(self):        
        self._swapper = None
        self._rmbg = None
        self._restore = None  
        
        self.comfy_path = None
        self.tbox_path = None
        self.ckpt_list = {}
        self.lora_list = {}
        
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
        
        self.load_ckpt()
        
        import folder_paths
        self.model_path = folder_paths.models_dir
        logger.info(f"Model Path: {self.model_path}")
        
    def load_ckpt(self):
        path = os.path.join(config.get('model_json'))
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.ckpt_map = {item['name']: item for item in data.get('checkpoint', [])}
        self.lora_map = {item['name']: item for item in data.get('lora', [])}
        
        # print("Checkpoint 字典:")
        # print(json.dumps(self.ckpt_map, indent=2, ensure_ascii=False))

        # print("\nLoRA 字典:")
        # print(json.dumps(self.lora_map, indent=2, ensure_ascii=False))

    def check_ckpt(self, name):
        default = self.ckpt_map.get('default')
        item = self.ckpt_map.get(name, default)
        return item['model']
    
    def check_lora(self, name):
        lora = self.lora_map.get(name, self.lora_map.get('FilmVelvia3'))
        return lora['model'], lora['clip_skip']
    
    def reset(self, task_type):
        if task_type == TaskType.Rmbg:
           self._restore = None
        elif task_type == TaskType.FaceRestore:
            self._rmbg = None
        else:
            self._rmbg = None
            self._restore = None

    def faceswap(self, task):
        from .faceswap import FaceSwapper
        self.reset(TaskType.FaceSwap)
        if self._swapper == None:
            self._swapper = FaceSwapper('inswapper_128', self.model_path, self.device)
        return self._swapper.process(task)
    
    def liveportrait(self, task):
        task_path = task.get_task_path()
        output_path = os.path.join(task_path, 'output.mp4')
        liveportrait_path = os.path.join(os.path.dirname(__file__), "comfy/live_portrait.py")
        device = 'CPU'
        if self.device == 'cuda':
            device = 'CUDA'
        elif self.device == 'mps':
            device = 'CoreML'
        commands = [
                'python', liveportrait_path, 
                '-c', self.comfy_path,
                '-p', task_path,
                '-d', device
            ]
        err = self.do_exec(commands=commands)
        if err != Error.OK:
            return err
        
        if os.path.isfile(output_path) == True:
            return output_path, Error.OK
        return '',  Error.FileNotFound
    
    def rmbg(self, task):
        from .rmbg import RMBG
        self.reset(TaskType.Rmbg)
        if self._rmbg == None:
            self._rmbg = RMBG(self.model_path, self.device)
        return self._rmbg.process(task)
    
    def restore(self, task):
        from .facerestore import FaceRestore
        self.reset(TaskType.FaceRestore)
        if self._restore == None:
            self._restore = FaceRestore(self.model_path, self.device)
        return self._restore.process(task)
    
    def do_exec(self, commands):
        try:
            result = subprocess.run(commands, capture_output=True, text=True, check=True)
            logger.info(f"do_exec return: {result.returncode}, stderr: {result.stderr.strip()}")
            return Error.OK
        except subprocess.CalledProcessError as e:
            logger.error(f"do_exec exception return: {e.returncode}, error: {e.stderr.strip()}")
            return Error.SubprocessFail
   
    def txt2img(self, task):
        print(f'task =  {task.json()}') 
        task_path = task.get_task_path()
        output_path = os.path.join(task_path, 'output.jpg')
        txt2img_path = os.path.join(os.path.dirname(__file__), "comfy/txt2img.py")
        ckpt_model = self.check_ckpt(task.txt2img.checkpoint)
        print(f'ckpt_model: {ckpt_model}')
        #lora_model, clip_skip = self.check_lora(task.txt2img.lora)
        negative_prompt = 'watermark,nsfw,' + task.txt2img.negative_prompt
        commands = [
                'python', txt2img_path, 
                '-c', self.comfy_path,
                '-o', output_path,
                '-W', str(task.txt2img.width),
                '-H', str(task.txt2img.height),
                '-m', ckpt_model,
                '-P', task.txt2img.positive_prompt,
                '-N', negative_prompt,
                '-F', str(task.txt2img.face_detailer == True)
            ]
        logger.debug(f'commands: {commands}')
        err = self.do_exec(commands=commands)
        if err != Error.OK:
            return err
        
        if os.path.isfile(output_path) == True:
            return output_path, Error.OK
        return '',  Error.FileNotFound
    
    def anime(self, task):
        print(f'task =  {task.json()}')
        task_path = task.get_task_path()
        target_path = os.path.join(task_path, 'target.jpg')
        output_path = os.path.join(task_path, 'output.jpg')
        anime_path = os.path.join(os.path.dirname(__file__), "comfy/anime.py")
        device = 'CPU'
        if self.device == 'cuda':
            device = 'CUDA'
        elif self.device == 'mps':
            device = 'CoreML'
        
        commands = [
                'python', anime_path, 
                '-c', self.comfy_path,
                '-i', target_path,
                '-o', output_path,
                '-W', '1024',
                '-H', '1024',
                '-d', device
            ]
        err = self.do_exec(commands=commands)
        if err != Error.OK:
            return err
        
        if os.path.isfile(output_path) == True:
            return output_path, Error.OK
        return '',  Error.FileNotFound
    
deep = Deep()


    
# export PYTHONPATH=$PWD python -m app.service.deep.faceswap   
#  anime  task =  {"uid":"0e4aa699aeabf6b599b3eab95c8c32fd","task_id":"b272db2e63c66b816ed033b0f7af599f","task_type":1002,"task_state":1,"priority":1,"credit":1,"start_time":1744399660,"update_time":28800,"video":false,"duration":null,"format":"jpg","obj_keys":{"source":null,"target":"r2://website/dev/task/b272db2e63c66b816ed033b0f7af599f/target","output":"r2://website/dev/task/b272db2e63c66b816ed033b0f7af599f/output.jpg"},"upscale":null,"live_portrait":null,"txt2img":null}
