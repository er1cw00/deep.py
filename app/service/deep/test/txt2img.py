import os
import sys
import subprocess
from tqdm import tqdm
from app.base.error import Error
from app.model.task import TaskState, TaskType, TaskInfo, get_task_type_name


def do_exec(commands):
    try:
        result = subprocess.run(commands, capture_output=False, text=True, check=True)
        print(f"do_exec return: {result.returncode}")# stderr: {result.stderr.strip()}")
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
task = get_task_info(task_id='c109d814cbcba7d8655683e5b20f6b25', task_type=TaskType.Txt2Img, video=False)
task_path = task.get_task_path()

output_path = os.path.join(task_path, 'output.jpg')
txt2img_path = os.path.join(os.path.dirname(__file__), "../comfy/txt2img2.py")
checkpoint = 'MoyouV2.safetensors'
print(f'checkpoint: {checkpoint}')
positive_prompt = 'A photorealistic image of a young woman with long flowing hair, wearing a swimsuit, standing on a sunny beach, by the ocean, soft lighting, natural shadows, 4K, ultra-detailed, realistic skin texture, DSLR photo, shallow depth of field, cinematic'
negative_prompt = 'watermark,nsfw'
commands = [
        'python', txt2img_path, 
        '-m', model_path,
        '-d', "mps",
        '-o', output_path,
        '-W', str(768),
        '-H', str(1024),
        '-c', checkpoint,
        '-P', positive_prompt,
        '-N', negative_prompt
    ]
commands_str = ' '.join(commands)
print(f'commands: {commands_str}')

err = do_exec(commands=commands)
if err != Error.OK:
    print(f'test live_portait fail, err: {err}')
    sys.exit(1)
    
if os.path.isfile(output_path) == False:
    print(f'test txt2img fail, output {output_path} not exist')
    sys.exit(1)

print(f'test txt2img success')