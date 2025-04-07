from pydantic import BaseModel
from typing import Optional
from .task import TaskInfo

class BaseResponse(BaseModel):
    code: int
    message: str
    
class KeepaliveResponse(BaseResponse):
    timestamp: int

class VersionResponse(BaseResponse):
    env: str
    build_time: str
    commit: str
    branch: str

class GetDeepTaskResponse(BaseResponse):
    task: Optional[TaskInfo] = None
    

class UpdateDeepTaskRequest(BaseModel):
    node: str
    uid: str
    task_id: str
    task_state: int
