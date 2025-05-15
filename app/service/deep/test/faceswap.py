
import os
import sys
from tqdm import tqdm
from app.base.error import Error
from app.model.task import TaskState, TaskType, TaskInfo, get_task_type_name
from app.service.deep.faceswap import FaceSwapper, FaceMaskConfig

def get_task_info(task_id, task_type, video):
    task_path = f'/Users/wadahana/workspace/AI/tbox.ai/deep.py/task/20250505/{task_id}' 
    task_info = {
        'uid': '1234567890',
        'task_id': task_id,
        'task_type': task_type,
        'task_state': TaskState.InProgress,    
        'priority': 1,
        'credit': 3,
        'start_time': 1745104229,
        'update_time': 1745104300,
        'format': 'mp4' if video else 'jpg',
        'video': video,
        'obj_keys': {} 
    }
    task = TaskInfo(**task_info)
    task._task_path = task_path
    return task

def test(swapper, task_id, task_type, video):
    name = get_task_type_name(task_type)
    task = get_task_info(task_id=task_id, task_type=task_type, video=video)
    output, err = swapper.process(task=task)
    if err != Error.OK:
        print(f'test {name} fail, err: {err}')
        return
    if os.path.isfile(output) == False:
        print(f'test {name} fail, {output} not exist')
        return
    print(f'test {name} success, output: {output}')
    return

model_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion'

mask_config = FaceMaskConfig(
    bbox=True,
    bbox_blur=0.3,
    occlusion=True,
)

swapper = FaceSwapper(model_path=model_path,
                      device="mps", 
                      mask_config=mask_config,
                      show_progress=False)


test(swapper=swapper, task_id='2e3df096b2f8db491d32df0ce2092555', task_type=TaskType.FaceSwap, video=False)
test(swapper=swapper, task_id='029ae54703795997925aabbfdacab41b', task_type=TaskType.FaceSwap, video=True)
    
    
# test(swapper=swapper, task_id='e99e58e983130db43bcac9fa0948e27d', task_type=TaskType.FaceRestore, video=False)
# test(swapper=swapper, task_id='b95771bfcf7c0f3788776f6315a6de2e', task_type=TaskType.FaceRestore, video=True)
    

