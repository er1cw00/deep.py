import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize


from bria.v1 import BriaV1
from app.base.error import Error

class RMBG:
    def __init__(self, model_path, device):
        self.device = device
        self.model_path = os.path.join(model_path, 'bria/rmbg-1.4.pth')
        
        self.net = BriaV1()
        self.net.load(self.model_path, self.device)

    def process(self, task):
        if task.video:
            return self.process_video(task)
        else:
            return self.process_image(task)  
        
    def process_image(self, task):
        task_path = task.get_task_path()
        target_path = os.path.join(task_path, 'target.jpg')
        output_path = os.path.join(task_path, 'output.png')

        target = cv2.imread(target_path)
        
        [h, w, c] = target.shape
        
        frame = cv2.resize(target, (1024, 1024), interpolation=cv2.INTER_LINEAR).astype(np.uint8)    
        tensor = torch.tensor(frame, dtype=torch.float32).permute(2,0,1)
        tensor = torch.unsqueeze(tensor,0)
        tensor = torch.divide(tensor,255.0)
        tensor = normalize(tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
        tensor = tensor.to(self.device)
        
        result = self.net(tensor)
        result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
        v_max = torch.max(result)
        v_min = torch.min(result)
        result = (result-v_min)/(v_max-v_min)    
        mask = (result*255).cpu().data.numpy().astype(np.uint8)
        
        output = cv2.cvtColor(np.array(target), cv2.COLOR_RGB2RGBA) 
        mask = mask
        output[:, :, 3] = mask
        
        cv2.imwrite(output_path, output)
        return output_path, Error.OK