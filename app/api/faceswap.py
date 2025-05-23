import os
import json
import time
import traceback
from loguru import logger
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.base.error import Error
from app.model.task import TaskInfo, TaskType, TaskState
from app.model.faceswap import FaceSwapResponse, FaceSwapRequest
from app.service.task import ts
from app.service.deep import deep

router = APIRouter()

@router.get("/internal/ping") #include_in_schema=False
async def ping():
    resp = JSONResponse(
        status_code=200,
        content={"code": 0, "message":"OK", "timestamp":int(time.time())}
    )
    return resp

@router.post("/facefusion/frame_process", response_model=FaceSwapResponse)
async def frame_process(req: FaceSwapRequest):
    print('req:' + json.dumps(req.model_dump()))
    task, message = req.toTaskInfo()
    if task is None:
        return JSONResponse(
            status_code=422,
            content={"output": None, "detail": message or "request invalid"}
        )
    output = ""
    task.task_state = TaskState.InProgress
    try:
        if task.task_type == TaskType.FaceSwap:
            output, err = deep.faceswap(task)
        if err == Error.OK:
            task.task_state = TaskState.Success
        else:
            logger.error(f'frame_process >> task({task.task_id}) fail, err: {err}')
            task.task_state = TaskState.Fail
            
    except Exception as e:
        err = Error.Unknown
        task.task_state = TaskState.Fail
        logger.error(f'frame_process >> task({task.task_id}) exception: {e}')
        traceback.print_exc()
        
    logger.debug(f"frame_process >> update task: {task.task_id}, state:{task.task_state}, output: {output} err: {err}")
    if task.task_state == TaskState.Success:
        if os.path.isfile(output) == True:
            return FaceSwapResponse(
                output=output,
                detail="success"
            )
        else:
            message = 'output file not exist'
    else:
        message = err or "unknown error"
        
    return JSONResponse(
            status_code=422,
            content={"output": None, "detail": str(message) if message else "unknown error"}
        )