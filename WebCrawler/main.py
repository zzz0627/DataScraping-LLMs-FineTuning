import json
import logging
from async_crawler import AsyncCrawler
from crawler_config import CrawlerConfig
from ua_info import ua_list

def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crawler.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 读取配置和URL
        config = CrawlerConfig()
        with open(config.INPUT_FILE, 'r', encoding='utf-8') as f:
            urls = json.load(f)
        
        logger.info(f"总共读取到{len(urls)}个URL")
        
        # 创建爬虫实例并开始爬取
        crawler = AsyncCrawler(config, ua_list)
        crawler.crawl(urls)
        
    except Exception as e:
        logger.error(f"程序运行错误: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 