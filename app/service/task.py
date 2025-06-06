import os
import requests
import ffmpeg
import traceback
from PIL import Image
from loguru import logger
from pydantic import ValidationError
from app.model.task import TaskInfo, TaskType, TaskState
from app.model.schema import BaseResponse, GetDeepTaskResponse, UpdateDeepTaskRequest
from app.base.media import get_mime_type_from_filepath, get_postfix_from_mime_type
from app.base.config import config
from app.base.error import Error
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
        self.api_key    = 'api-'+config.get('app_key')
        self.env        = config.get('env')
        self.node_name  = config.get('name')
        self.api_base   = config.get('deep.api_base')
        self.path       = config.get('deep.task_path')
        self.types      = config.get('deep.task_type')
        task_types = {'comfy', 'faceswap'}
        self.types = [t for t in self.types if t in task_types] # 过滤只保留 'comfy' 和 'faceswap'
        
        logger.info(f"TaskService start >> env({self.env}), node({self.node_name}), types({self.types})")

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
                
    def generate_task_types(self):
        if not self.types:  # 如果 types 为空
            return ''
        elif len(self.types) == 1:  # 如果 types 只有一个字符串
            return f'taskType={self.types[0]}'
        else:  # 如果 types 有两个字符串
            return '&'.join(f'taskType={t}' for t in self.types)  # 用 '&' 连接
        
    def do_get_task(self, proxy=None):
        types = self.generate_task_types()
        url = f'{self.api_base}/task?node={self.node_name}'
        if types != '':
            url = url + '&' + types
        logger.debug(f'url: {url}')
        
        headers = {'Authorization': f'Basic {self.api_key}'}
        proxies = None
        if proxy != None:
            proxies = {
                "http": proxy,
                "https": proxy
            }
        try:
            resp = requests.get(url, headers=headers, proxies=proxies)
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
            return self.download_file(url, task_path, name)
        elif url.startswith("r2://") or url.startswith("s3://"):
            return s3.get_object(url[5:], task_path, name)
        elif url.startswith("file://"):
            return url[7:], Error.OK
    
    def do_update_task(self, req, proxy=None):

        url = f'{self.api_base}/task'
        headers = {'Authorization': f'Basic {self.api_key}'}
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


    def download_file(self, url, task_path, name):
        tmp_file, content_type, err = self.do_download_file(url, task_path, name)
        if err == Error.NetworkError:
            logger.error(f'download ({url}) network error')
            return '', err
        
        format, video, err = get_postfix_from_mime_type(content_type)
        if err != Error.OK:
            logger.error(f'download ({url}) unsupport file type({content_type})')
            return '', err
        
        if video == False:
            dst_file = os.path.join(task_path, f'{name}.jpg')
            if format == 'jpg':
                os.rename(tmp_file, dst_file)
            elif format == 'png':
                with Image.open(tmp_file) as img:
                    img = img.convert("RGB")
                    img.save(dst_file, "JPEG")
                os.remove(tmp_file)
            else:
                print(f'download ({url}) unknown file type({content_type})')
                return '', Error.UnsupportFile
        else:
            dst_file = os.path.join(task_path, f'{name}.mp4')
            if format == 'mp4':
                os.rename(tmp_file, dst_file)
            elif format == 'mov':
                ffmpeg.input(tmp_file).output(dst_file, vcodec="copy", acodec="copy").run(quiet=True, overwrite_output=True)
                os.remove(tmp_file)
            else:
                print(f'download ({url}) unknown file type({content_type})')
                return '', Error.UnsupportFile                    
            
        if os.path.isfile(dst_file): 
            return dst_file, Error.OK
            
        return '', Error.FileNotFound
    
    def do_download_file(self, url, task_path, name, proxies=None ):
        try:
            headers = {'Authorization': f'Basic {self.api_key}'}
            response = requests.get(url, headers=headers, stream=True, timeout=15, proxies=proxies)
            
            status = response.status_code
            if status == 404:
                return '', '', Error.FileNotFound
            elif status == 401:
                return '', '', Error.Unauthorized
            elif 500 <= status < 600:
                return '', '', Error.ServerError
            
            # 获取 Content-Type 并映射到扩展名
            content_type = response.headers.get("Content-Type", "").lower()
            content_length = response.headers.get("Content-Length")
            content_length = int(content_length) if content_length is not None else 0
            if content_length == 0:
                return '', '', Error.UnsupportFile
            
            temp_file = os.path.join(task_path, f'{name}.tmp')
            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return temp_file, content_type, Error.OK 
        
        except requests.exceptions.RequestException as e:
            logger.error(f"request exception >> {e}")
            err = Error.NetworkError
        except requests.exceptions.ConnectionError:
            logger.error(f"connect exception >> {e}")
            err = Error.NetworkError
        except Exception as e:
            logger.error(f"exception >> {e}")
            err = Error.Unknown
            
        return '', '', err     
        
        
ts = TaskService()