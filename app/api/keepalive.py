import time
from fastapi import APIRouter, Depends
from app.model.schema import KeepaliveResponse

router = APIRouter()

@router.get("/keepalive", response_model=KeepaliveResponse)
async def keepalive():
    resp = KeepaliveResponse(
        code=200,
        message="OK",
        timestamp=int(time.time()) 
    )
    return resp