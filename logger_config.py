import logging
import logging.handlers
import os
from datetime import datetime

def setup_logger(name="CRC25", log_dir="log"):
    """
    设置日志记录器
    - 200MB文件滚动
    - 记录时间、线程、调用位置
    """
    # 创建log目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 日志文件名加上年月日时分秒
    now_str = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"{name}_{now_str}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=200 * 1024 * 1024,  # 200MB
        backupCount=10,
        encoding='utf-8'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    
    # 创建格式化器：时间 进程 线程 调用位置
    formatter = logging.Formatter(
        '%(asctime)s - [%(process)d][%(thread)d] - %(levelname)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置格式化器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 全局logger实例
logger = setup_logger()
