
import os
import sys
import subprocess
from tqdm import tqdm
from app.base.error import Error
from app.model.task import TaskState, TaskType, TaskInfo, get_task_type_name


def do_exec(commands):
    try:
        result = subprocess.run(commands, capture_output=False, text=True, check=True)
        print(f"do_exec return: {result.returncode}, stderr: {result.stderr.strip()}")
        return Error.OK
    except subprocess.CalledProcessError as e:
        print(f"do_exec exception return: {e.returncode}, error: {e.stderr.strip()}")
        return Error.SubprocessFail
    
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

model_path = '/Users/wadahana/workspace/AI/sd/ComfyUI/models'
task = get_task_info(task_id='e99e58e983130db43bcac9fa0948e27d', task_type=TaskType.LivePortrait, video=False)
task_path = task.get_task_path()
output_path = os.path.join(task_path, 'output.mp4')
liveportrait_path = os.path.join(os.path.dirname(__file__), "../comfy/live_portrait.py")

commands = [
    'python', liveportrait_path, 
    '-m', model_path,
    '-p', task_path,
    '-d', 'mps'
]
err = do_exec(commands=commands)
if err != Error.OK:
    print(f'test live_portait fail, err: {err}')
    
if os.path.isfile(output_path) == False:
    print(f'test live_portait fail, output {output_path} not exist')

print(f'test live_portait success')