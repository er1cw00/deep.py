import os
import requests
from pydantic import ValidationError
from app.model.task import TaskInfo, TaskType, TaskState
from app.model.schema import BaseResponse, GetDeepTaskResponse, UpdateDeepTaskRequest
from app.base.config import config
from app.base.error import Error
from app.base.logger import logger
from app.service.s3 import s3

def ensure_dir(directory: str):
    try:
        os.makedirs(directory, exist_ok=True)  # 存在不会报错
        return True  # 成功返回 True
    except Exception as e:
        print(f"mkdir fail: {e}")
        return False  # 失败返回 False

class TaskService:
    def __init__(self):
        pass
    def init(self):
        self.env        = config.get('env')
        self.api_base   = config.get('api_base')
        self.api_key    = config.get('api_key')
        self.node_name  = config.get('name')
        self.path       = config.get("task_path")
        logger.info(f"TaskService start >>")
    
        
    def get_task(self):
        count = 0 
        if self.env == 'pro':
            count = config.get_proxy_count()
        if count == 0:
            resp = self.do_get_task(proxy=None)
            if resp != None and resp.code == 0:
                return resp.task
        for i in range(count):
            proxy = config.get_proxy(i)
            resp = self.do_get_task(proxy)
            if resp != None and resp.code == 0:
                return resp.task
        return None
    
    def prepare_files(self, task):
        logger.debug(f"prepare_files >>")
        task_path = task.get_task_path()
        logger.debug(f"prepare_files >> task_path: {task_path}")

        if ensure_dir(task_path) == False:
            return Error.FileNotFound
        if os.path.isdir(task_path) == False:
            return Error.FileNotFound
        
        if task.obj_keys.source != None:
            _, err = self.do_fetch_file(url=task.obj_keys.source, name='source', task_path=task_path)
            if err != Error.OK:
                return err
        if task.obj_keys.target != None:
            _, err = self.do_fetch_file(url=task.obj_keys.target, name='target', task_path=task_path)
            if err != Error.OK:
                return err
        return Error.OK
    
    def update_task(self, task, output):
        req = UpdateDeepTaskRequest(
            node        = self.node_name,
            uid         = task.uid,
            task_id     = task.task_id,
            task_state  = TaskState.Fail
        )
        if task.task_state == TaskState.Success:
            if os.path.isfile(output) == True:
                obj_key = task.obj_keys.output
                if obj_key.startswith("r2://") or obj_key.startswith("s3://"):
                    obj_key = obj_key[5:]
                err = s3.put_object(output, obj_key)
                if err == Error.OK:
                    req.task_state  = TaskState.Success
                else:
                    logger.error(f"update_task >> put obj task({task.task_id}) result fail, err: {err}")
        count = 0 
        err = Error.OK
        if self.env == 'pro':
            count = config.get_proxy_count()
                
        if count == 0:
            resp, err = self.do_update_task(req, proxy=None)
            if err != Error.OK:
                logger.error(f"update_task >> update task({task.task_id}) fail, err: {err}")
                return err
            
        for i in range(count):
            proxy = config.get_proxy(i)
            resp, err = self.do_update_task(req, proxy)
            if err == Error.OK: 
                return Error.OK
            else: 
                logger.warning(f"update_task >> update task({task.task_id}) fail, err: {err}")
        

    def do_get_task(self, proxy=None):
        url = f'{self.api_base}/task'
        params = {
            "taskType": "both",
            "node": self.node_name
        }
        headers = {'Authorization': f'Basic api-{self.api_key}'}
        proxies = None
        if proxy != None:
            proxies = {
                "http": proxy,
                "https": proxy
            }
        try:
            resp = requests.get(url, params=params, headers=headers, proxies=proxies)
            if resp.status_code == 200 :
                # 解析返回的 JSON 并转换为 Pydantic 对象
                j = resp.json()
                task_resp = GetDeepTaskResponse.model_validate(j)
                return task_resp
            else:
                # 处理请求失败的情况
                logger.error(f"fetch_task >> {resp.reason} {resp.status_code} - {resp.text}")
            return None
        except ValidationError as e:
            logger.error(f"unmarsh task resp fail >> {e}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"request http error exception >> {e}")
        except requests.exceptions.Timeout as e:
            logger.error(f"request timeout exception >> {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"request exception >> {e}")
        except Exception as e:
            logger.error(f"unknown exception >> {e}")
        return None
    
    def do_fetch_file(self, url, name, task_path) -> tuple[str, Error]:
        logger.debug(f"do_fetch_file >> url: {url}, name: {name}, task_path: {task_path}")
        if url.startswith("http://") or url.startswith("https://"):
            return '', Error.FileNotFound
        elif url.startswith("r2://") or url.startswith("s3://"):
            return s3.get_object(url[5:], task_path, name)
        elif url.startswith("file://"):
            return url[7:], Error.OK
    
    def do_update_task(self, req, proxy=None):

        url = f'{self.api_base}/task'
        headers = {'Authorization': f'Basic api-{self.api_key}'}
        proxies = None
        if proxy != None:
            proxies = {
                "http": proxy,
                "https": proxy
            }
        try:
            response = requests.post(url, json=req.dict(), headers=headers, proxies=proxies)
            if response.status_code == 200 :
                # 解析返回的 JSON 并转换为 Pydantic 对象
                j = response.json()
                resp = BaseResponse.model_validate(j)
                return resp, Error.OK
            else:
                logger.error(f"fetch_task >> {resp.reason} {resp.status_code} - {resp.text}")
                err = Error.NetworkError
        except ValidationError as e:
            logger.error(f"unmarsh task resp fail >> {e}")
            err = Error.UnknownResponse
        except requests.exceptions.HTTPError as e:
            logger.error(f"request http error exception >> {e}")
            err = Error.NetworkError
        except requests.exceptions.Timeout as e:
            logger.error(f"request timeout exception >> {e}")
            err = Error.NetworkError
        except requests.exceptions.RequestException as e:
            logger.error(f"request exception >> {e}")
            err = Error.NetworkError
        except Exception as e:
            logger.error(f"unknown exception >> {e}")
            err = Error.Unknown
        return None, err
    
ts = TaskService()