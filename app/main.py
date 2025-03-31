import os
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
from app.routes import routes_init


os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

## init config
parser = argparse.ArgumentParser(description="conv.yaml")
parser.add_argument("--config", type=str, default="conv.yaml")
args = parser.parse_args()

config.init(args.config)


#comfy_root = config.get('deep.comfy_path')

## init logger
logger_init()



#print(f'Comfy root path: {comfy_root}')

# init_comfy(comfy_root)


s3.init()


# FastAPI 初始化
app = FastAPI(lifespan=lifespan) 
routes_init(app)

server_host = config.get('server.host')
server_port = config.get('server.port')

logger.info(f'Server running at http://{server_host}:{server_port}')

uvicorn.run(app, host=server_host, port=server_port,loop="asyncio", log_config=None, log_level="info")