import time
from fastapi import APIRouter, Depends
from app.model.schema import KeepaliveResponse

router = APIRouter()

#@router.get("/internal/ping", response_model=KeepaliveResponse) #include_in_schema=False
@router.get("/keepalive", response_model=KeepaliveResponse)
async def keepalive():
    resp = KeepaliveResponse(
        code=0,
        message="OK",
        timestamp=int(time.time()) 
    )
    return resp
