import os
import asyncio
import requests
import traceback
from loguru import logger
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import ValidationError
from app.base.config import config
from app.base.error import Error
from app.model.schema import GetDeepTaskResponse, UpdateDeepTaskRequest
from app.model.task import TaskType, TaskState
from app.service.task import ts
from app.service.deep import deep

class Scheduler:
    def __init__(self):
        logger.info(f'scheduler init >> ')
    
    def dispatch_task(self, task):
        output = ''
        task.task_state == TaskState.InProgress
        try:
            err = ts.prepare_files(task)
            if err != Error.OK:
                logger.error(f'dispatch_task >> prepare files for task({task.task_id}) fail, err: {err}')
            else:
                if task.task_type == TaskType.FaceSwap:
                    output, err = deep.faceswap(task)
                elif task.task_type == TaskType.Rmbg:
                    output, err = deep.rmbg(task)
                elif task.task_type == TaskType.FaceRestore:
                    output, err = deep.restore(task)
                elif task.task_type == TaskType.LivePortrait:
                    output, err = deep.liveportrait(task)
                elif task.task_type == TaskType.Anime:
                    output, err = deep.anime(task)
                elif task.task_type == TaskType.Txt2Img:
                    output, err = deep.txt2img(task)

            if err == Error.OK:
                task.task_state = TaskState.Success
            else:
                logger.error(f'dispatch_task >> task({task.task_id}) fail, err: {err}')
                task.task_state = TaskState.Fail  
        except Exception as e:
            err = Error.Unknown
            task.task_state = TaskState.Fail
            logger.error(f'dispatch_task >> task({task.task_id}) exception: {e}')
            traceback.print_exc()
            
        logger.debug(f"dispatch_task >> update task: {task.task_id}, output: {output} err: {err}")
        ts.update_task(task, output)
        
       
    async def scheule_task(self):
        """异步后台任务：定期拉取数据"""
        interval = config.get('interval', 5)
        logger.info(f'scheule_task >>> interval: {interval}')
        await asyncio.sleep(5)
        while True:
            #logger.debug("scheule_task run....")
            try:
                task = ts.get_task()
                if task != None:
                    self.dispatch_task(task)
            except Exception as e:
                logger.warning(f"Error fetching task: {e}")
                traceback.print_exc()
            #logger.debug("scheule_task run >>")
            await asyncio.sleep(interval)
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动异步后台任务"""
    if config.get('mode', 'none') == 'pull':
        scheduler = Scheduler()
        asyncio.create_task(scheduler.scheule_task())
    yield