from pydantic import BaseModel
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
    task: TaskInfo
    
