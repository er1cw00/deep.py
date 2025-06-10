import os
import sys
import logging
from loguru import logger
from app.base.config import config
from datetime import datetime

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # 获取 Loguru 的日志级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # 捕获日志的原始位置（避免所有日志显示为来自此处的调用）
        frame, depth = logging.currentframe(), 6
        while frame and depth > 0:
            frame = frame.f_back
            depth -= 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    #current_time = datetime.now().strftime("%Y%m%d0000")
    #log_file = os.path.join(log_path, f"logview-{current_time}.log")
def logger_init():
    log_path = config.get('log.path')
    log_level = config.get('log.level').upper()
    env = config.get('env')
            
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
        print(f"创建日志目录: {log_path}")
    

    log_file = os.path.join(log_path, "logview-{time:YYYYMMDDHHmmss}.log")

    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        print(f"Invalid log level '{log_level}', fallback to 'INFO'")
        log_level = 'INFO'
    print(f'log_level: {log_level}')
    logger.remove()
    log_format = "{time:HH:mm:ss.SSS} <level>{level:<4.4s}</level> {file}:{line} {message}"
    print(f'log_file: {log_file}')
    # 配置 Loguru
    logger.add(
        log_file,
        format=log_format,
        #rotation="10 minutes",
        rotation="00:00",               # 每日轮换（按需修改）
        level=log_level,                # 动态设置日志级别
        enqueue=True,                   # 多进程安全
        encoding="utf-8",               # 编码格式
        backtrace=True,                 # 记录异常堆栈（可选）
        diagnose=True                   # 显示变量值（可选）
    )
    if env != 'pro':
        logger.add(
            sink=sys.stdout,
            level=log_level,
            format=log_format,
            enqueue=True
        )
    
    # **重定向 FastAPI、Uvicorn 和 `boto3` 日志**
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)
    
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "boto3", "botocore", "urllib3"):
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
        logging.getLogger(logger_name).propagate = False  # 防止日志重复

    logger.info("Logger initialized successfully!")
