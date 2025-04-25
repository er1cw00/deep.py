


if __name__ == '__main__':
    import os
    from loguru import logger
    from app.base.logger import logger_init
    from app.base.config import config
    from app.model.task import TaskState, TaskType, TaskInfo 
    from app.service.deep import Deep
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
   
    task_path = "/Users/wadahana/workspace/AI/tbox.ai/data/deep/task/20250425/abfbb5cfe35cabbfd209ce312c8f9750" 
    task_info = {
        'uid': '1234567890',
        'task_id': 'abfbb5cfe35cabbfd209ce312c8f9750',
        'task_type': TaskType.LivePortrait,
        'task_state': TaskState.InProgress,    
        'priority': 1,
        'credit': 1,
        'start_time': 1745104229,
        'update_time': 1745104300,
        'format': 'mp4',
        'video': False,
        'obj_keys': {} 
    }
    
    task = TaskInfo(**task_info)
    task._task_path = task_path
    
    deep = Deep()
    deep.init()
    deep.load_ckpt()
    deep.liveportrait(task)
    
    # faceswap 20250423/e6fb79adb157a4d9e28167b06e603752
    # from .faceswap import FaceSwapper
    # swapper = FaceSwapper('inswapper_128', model_path, 'mps')
    # swapper.process_image(task)

