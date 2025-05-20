
import os
import sys
import time
from tqdm import tqdm
from app.base.error import Error
from app.model.task import TaskState, TaskType, TaskInfo, get_task_type_name
from app.service.deep.faceswap import FaceSwapper, FaceMaskConfig
from app.deepfake.test.file import get_test_files, TEST_DIR
from app.deepfake.utils import Timer

def get_task_info(task_id, task_type, video):
#    task_path = os.path.join(TEST_DIR, f'{task_id}')
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
    t = Timer()
    t.tic()
    output, err = swapper.process(task=task)
    t.toc()
    if err != Error.OK:
        print(f'test {name} fail, err: {err}')
        return
    if os.path.isfile(output) == False:
        print(f'test {name} fail, {output} not exist')
        return
    print(f'test {name} success, output: {output}')
    t.show('faceswap')
    return

model_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models/facefusion'
#model_path = "/home/eric/workspace/AI/sd/ComfyUI/models/facefusion"

mask_config = FaceMaskConfig(
    bbox=True,
    bbox_blur=0.3,
    occlusion=True,
)

swapper = FaceSwapper(model_path=model_path,
                      device="cuda", 
                      mask_config=mask_config,
                      show_progress=True)



#test(swapper=swapper, task_id='53dd886693270e1811a465740f7a266a', task_type=TaskType.FaceSwap, video=True)


# test(swapper=swapper, task_id='2e3df096b2f8db491d32df0ce2092555', task_type=TaskType.FaceSwap, video=False)
# test(swapper=swapper, task_id='029ae54703795997925aabbfdacab41b', task_type=TaskType.FaceSwap, video=True)
    
    
# test(swapper=swapper, task_id='e99e58e983130db43bcac9fa0948e27d', task_type=TaskType.FaceRestore, video=False)
# test(swapper=swapper, task_id='b95771bfcf7c0f3788776f6315a6de2e', task_type=TaskType.FaceRestore, video=True)
    
    
#photo_list, video_list = get_test_files()

# print("Photo directories:")
# for path in photo_list:
#     subdir = os.path.basename(os.path.normpath(path))
#     print(f'photo task: {subdir}')
#     test(swapper, task_id=subdir,task_type=TaskType.FaceSwap,video=False)

# print("\nVideo directories:")
# for path in video_list:
#     subdir = os.path.basename(os.path.normpath(path))
#     print(f'video task: {subdir}')
#     test(swapper, task_id=subdir,task_type=TaskType.FaceSwap,video=True)
#     time.sleep(20)

# test(swapper=swapper, task_id='0b782a3795a49d30044efb1cbfacb30f', task_type=TaskType.FaceSwap, video=False)
# time.sleep(15)
# test(swapper=swapper, task_id='0b782a3795a49d30044efb1cbfacb30f', task_type=TaskType.FaceSwap, video=False)
# time.sleep(20)
# test(swapper=swapper, task_id='1e42b87f42559936a9447be1bce59165', task_type=TaskType.FaceSwap, video=True)


# 32b7fb67fca93cd460bd7334bf845e39 blink
# 53dd886693270e1811a465740f7a266a no audio file
# 87f97cd804122945562134319fa5d6ea face detect
# fbb6081fa3544ba51e4058b71660cfe3 face detect

