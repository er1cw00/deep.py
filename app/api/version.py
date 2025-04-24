import time
from fastapi import APIRouter, Depends
from app.model.schema  import VersionResponse
from app.version import VERSION, BUILD_TIME, GIT_BRANCH, GIT_COMMIT

router = APIRouter()

@router.get("/version", response_model=VersionResponse)
async def version():
    resp = VersionResponse(
        code=200,
        message="OK",
        env='dev',
        version=VERSION,
        branch=GIT_BRANCH,
        commit=GIT_COMMIT,
        build_time=BUILD_TIME,
    )
    return resp