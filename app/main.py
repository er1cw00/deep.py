import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

import argparse
import uvicorn
import asyncio
from loguru import logger
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.base.logger import logger_init
from app.base.config import config
from app.service.schedule import lifespan
from app.service.s3 import s3
from app.service.task import ts
from app.service.deep import deep
from app.routes import routes_init


## init config
parser = argparse.ArgumentParser(description="conv.yaml")
parser.add_argument("--config", type=str, default="conv.yaml")
args = parser.parse_args()

config.init(args.config)

# 初始化日志
logger_init()

s3.init()
ts.init()
deep.init()

# FastAPI 初始化
app = FastAPI(lifespan=lifespan) 
routes_init(app)

server_host = config.get('server.host')
server_port = config.get('server.port')

logger.info(f'Server running at http://{server_host}:{server_port}')

uvicorn.run(app, host=server_host, port=server_port,loop="asyncio", timeout_keep_alive=30, log_level="info", access_log=False)