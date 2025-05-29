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

from app.service.deep.rmbg import RMBG
from app.service.deep.faceswap import FaceSwapper#, FaceMaskConfig
#from app.service.deep.utils import add_tbox_path_to_sys_path, add_comfy_path_to_sys_path

class Deep:
    def __init__(self):        
        self._swapper = None
        self._rmbg = None
        
        self.model_path = None
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

        self.model_path = config.get('model_path')
        if not os.path.isdir(self.model_path):
            logger.error(f"models path not exist: {self.model_path}")
            return False
        
        logger.info(f"model path: {self.model_path}")
        logger.info(f"device: {self.device}")

        self.load_ckpt()
        
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
    
    def do_faceswap(self, task):
        if self._swapper == None:

            model_path = os.path.join(self.model_path, 'facefusion')
            self._swapper = FaceSwapper(model_path=model_path,
                                device=self.device, 
                                mask_bbox=True,
                                mask_occlusion=True,
                                show_progress=False)
            
        return self._swapper.process(task)
    
    def faceswap(self, task):
        return self.do_faceswap(task)
    
    def restore(self, task):
        return self.do_faceswap(task)
    
    def rmbg(self, task):
        if self._rmbg == None:
            model_path = os.path.join(self.model_path, 'bria')
            self._rmbg = RMBG(model_path=model_path, device=self.device)
        return self._rmbg.process(task)
    
    def liveportrait(self, task):
        task_path = task.get_task_path()
        output_path = os.path.join(task_path, 'output.mp4')
        liveportrait_path = os.path.join(os.path.dirname(__file__), "comfy/live_portrait.py")
        commands = [
                'python', liveportrait_path, 
                '-d', self.device,
                '-m', self.model_path,
                '-p', task_path
            ]
        err = self.do_exec(commands=commands)
        if err != Error.OK:
            return err
        
        if os.path.isfile(output_path) == True:
            return output_path, Error.OK
        return '',  Error.FileNotFound
    

    
   
    
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
        positive_prompt = '' + task.txt2img.positive_prompt
        negative_prompt = 'watermark,nsfw,' + task.txt2img.negative_prompt
        commands = [
                'python', txt2img_path, 
                '-d', self.device,
                '-m', self.model_path,
                '-c', ckpt_model,
                '-o', output_path,
                '-W', str(task.txt2img.width),
                '-H', str(task.txt2img.height),
                '-P', positive_prompt,
                '-N', negative_prompt
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
        anime_path = os.path.join(os.path.dirname(__file__), "comfy/anime2.py")
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
