import os
import random
from typing import List

class CrawlerConfig:
    # 获取项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    
    # 重试配置
    RETRY_COUNT: int = 3        # 每个URL最多重试次数
    RETRY_DELAY: int = 1        # 重试等待间隔（秒）
    
    # 并发配置
    CONCURRENCY_LIMIT: int = 50  # 并发连接数限制
    MAX_PROCESSES: int = 16      # 最大进程数
    
    # 请求配置
    REQUEST_TIMEOUT: int = 60    # 请求超时时间（秒）
    
    # 文件路径配置
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    
    INPUT_FILE: str = os.path.join(RAW_DIR, "detail_urls.json")
    OUTPUT_DIR: str = os.path.join(RAW_DIR)
    
    @staticmethod
    def get_user_agent(ua_list: List[str]) -> str:
        return random.choice(ua_list) 