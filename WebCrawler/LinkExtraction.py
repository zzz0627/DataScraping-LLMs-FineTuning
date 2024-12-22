import time
import json
import logging
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

class XYWYLinkCrawler:
    def __init__(self, start_page=1, end_page=5):
        self.start_page = start_page
        self.end_page = end_page
        self.all_detail_urls = []
        
        # 设置数据目录
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.raw_data_dir = os.path.join(self.project_root, 'data', 'raw')
        os.makedirs(self.raw_data_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.raw_data_dir, 'link_extraction.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 初始化Selenium，添加更多选项
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # 无头模式
        options.add_argument('--disable-gpu')
        # 添加以下选项来解决SSL问题
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # 禁用日志
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        self.driver = webdriver.Chrome(options=options)

    def parse_list_page(self, page: int):
        """解析列表页，获取详情页链接"""
        url = f"https://club.xywy.com/list_all_{page}.htm"
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.ksAll-list .fl a'))
            )

            links = self.driver.find_elements(By.CSS_SELECTOR, '.ksAll-list .fl a')
            detail_urls = [link.get_attribute('href') for link in links if 'wenda' in link.get_attribute('href')]
            return detail_urls

        except Exception as e:
            self.logger.error(f"解析列表页失败 {url}: {str(e)}")
            return []

    def save_links(self, filename="detail_urls.json"):
        """保存收集到的详情链接到raw目录"""
        if not self.all_detail_urls:
            self.logger.info("无链接可保存")
            return

        # 构建完整的保存路径
        save_path = os.path.join(self.raw_data_dir, filename)
        
        # 保存为JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_detail_urls, f, ensure_ascii=False, indent=2)

        self.logger.info(f"详情页链接已保存: {len(self.all_detail_urls)}条 -> {save_path}")

    def crawl(self):
        try:
            total_pages = self.end_page - self.start_page + 1
            self.logger.info(f"开始爬取链接，共 {total_pages} 页...")
            
            # 使用tqdm创建进度条
            with tqdm(total=total_pages, desc="爬取进度") as pbar:
                for page in range(self.start_page, self.end_page + 1):
                    detail_urls = self.parse_list_page(page)
                    self.all_detail_urls.extend(detail_urls)
                    pbar.update(1)
                    pbar.set_postfix({'链接数': len(detail_urls)})

            self.logger.info(f"爬取完成，总共获取 {len(self.all_detail_urls)} 个链接")
            self.save_links("detail_urls.json")

        except Exception as e:
            self.logger.error(f"链接爬取错误: {str(e)}")
        finally:
            self.driver.quit()


if __name__ == "__main__":
    # 抓取 1 ~ 1000 页的详情页链接
    crawler = XYWYLinkCrawler(start_page=1, end_page=3)
    crawler.crawl()