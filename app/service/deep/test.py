


if __name__ == '__main__':
    import os
    from loguru import logger
    from app.base.logger import logger_init
    from app.base.config import config
    from app.model.task import TaskState, TaskType, TaskInfo 
    from .utils import add_tbox_path_to_sys_path, add_comfy_path_to_sys_path
    
    config.init("./deep.yaml")
    logger_init()
#    deep.init()

    comfy_path = config.get('comfy_path')
    tbox_path = os.path.join(comfy_path, "custom_nodes/ComfyUI-tbox/src")
    add_comfy_path_to_sys_path(comfy_path)
    add_tbox_path_to_sys_path(tbox_path)
    
    import folder_paths
    model_path = folder_paths.models_dir
    logger.info(f"Model Path: {model_path}")
       
    
    current_path = os.getcwd() #os.path.dirname(os.path.abspath(__file__))
    #task_path = os.path.join(current_path, "test/ABCDEFGHIJKL")
    #os.mkdir(task_path, exist_ok=True)
    task_path = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250423/71cc479c0d8ef0c05cce86b24e0a9bf9"
    task_info = {
        'uid': '1234567890',
        'task_id': '71cc479c0d8ef0c05cce86b24e0a9bf9',
        'task_type': TaskType.FaceSwap,
        'task_state': TaskState.InProgress,    
        'priority': 1,
        'credit': 1,
        'start_time': 1745104229,
        'update_time': 1745104300,
        'format': 'jpg',
        'video': False,
        'obj_keys': {} 
    }
    
    task = TaskInfo(**task_info)
    task._task_path = task_path
    
    from .faceswap import FaceSwapper
    swapper = FaceSwapper('inswapper_128', model_path, 'mps')
    swapper.process_image(task)

