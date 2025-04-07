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
from app.service.comfy import comfy

class Scheduler:
    def __init__(self):
        logger.info(f'scheduler init >> ')
    
    def dispatch_task(self, task):
        task.task_state == TaskState.InProgress
        err = ts.prepare_files(task)
        if err != Error.OK:
            logger.error(f'dispatch_task >> prepare files for task({task.task_id}) fail, err: {err}')
        if task.task_type == TaskType.FaceSwap:
            output, result = comfy.faceswap(task)
        
        if result != Error.OK:
            task.task_state = TaskState.Fail  
        else:
            task.task_state = TaskState.Success
            
        ts.update_task(task, output)
               
        logger.debug(f"dispatch_task >> task: {task.task_id}, err: {err}")
        
       # self.update_task(task)
            
    

       
    async def scheule_task(self):
        """异步后台任务：定期拉取数据"""
        logger.info('scheule_task >>>')
        await asyncio.sleep(5)
        while True:
            logger.debug("scheule_task run....")
            try:
                task = ts.get_task()
                if task != None:
                    self.dispatch_task(task)
            except Exception as e:
                logger.warning(f"Error fetching task: {e}")
                traceback.print_exc()
            await asyncio.sleep(5)
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动异步后台任务"""
    
    scheduler = Scheduler()
    asyncio.create_task(scheduler.scheule_task())
    yield