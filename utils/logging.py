import os
import sys

from constants import merged_config_path
from loguru import logger as log

FORMAT = (
    "<green>{time:MM-DD HH:mm:ss.S}</green> | "
    "<level>{level.icon}</level> | "
    "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

def set_logger(level, file_name=None):
    log.remove()
    log.add(sys.stdout, format=FORMAT, level=level)

    levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    for level_name in levels:
        log.level(level_name, icon=level_name[0])

    # Also log to file
    service_name = os.getenv("SERVICE_NAME", "host")
    if file_name is None:
        log_file_path = os.path.join(os.path.abspath(os.path.dirname(merged_config_path)), f"{service_name}.log")
    else:
        file_name = os.path.splitext(os.path.basename(file_name))[0]
        log_file_path = os.path.join(os.path.abspath(os.path.dirname(merged_config_path)), f"{service_name}_{file_name}.log")

    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
    with open(log_file_path, 'a'):
        os.utime(log_file_path, None)

    os.chmod(log_file_path, 0o777)
    log.add(log_file_path, level=level)
