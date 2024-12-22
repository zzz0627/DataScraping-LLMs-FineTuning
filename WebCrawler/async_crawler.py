import asyncio
import aiohttp
import os
import json
import logging
import time
import ujson
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import List, Dict, Optional
from multiprocessing import Process, cpu_count
from aiohttp import ClientTimeout, TCPConnector
from crawler_config import CrawlerConfig

class AsyncCrawler:
    def __init__(self, config: CrawlerConfig, ua_list: List[str]):
        self.config = config
        self.user_agent = config.get_user_agent(ua_list)
        self.logger = logging.getLogger(__name__)
        
        # 确保数据目录存在
        os.makedirs(self.config.RAW_DIR, exist_ok=True)
        os.makedirs(self.config.PROCESSED_DIR, exist_ok=True)

    async def fetch_and_parse(self, url: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """抓取并解析单个URL，带重试逻辑与GB18030解码"""
        for attempt in range(self.config.RETRY_COUNT):
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        self.logger.warning(f"非200状态码: {url} status={response.status}, 尝试重试...({attempt+1}/{self.config.RETRY_COUNT})")
                        await asyncio.sleep(self.config.RETRY_DELAY)
                        continue
                        
                    content = await response.read()
                    try:
                        html = content.decode('gb18030', errors='replace')
                    except:
                        self.logger.error(f"编码解码失败: {url}")
                        return None

                    soup = BeautifulSoup(html, "lxml")
                    question = soup.select_one('.clearfix h1')
                    answer = soup.select_one('#xywy_2024')

                    if question and answer:
                        question_text = question.get_text(strip=True)
                        answer_text = answer.get_text(strip=True)
                        if len(question_text) > 1 and len(answer_text) > 10:
                            return {
                                'question': question_text,
                                'answer': answer_text,
                                'url': url
                            }
                    return None

            except (aiohttp.ClientError, aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                self.logger.warning(f"请求异常 {url}: {str(e)} 尝试重试({attempt+1}/{self.config.RETRY_COUNT})")
                await asyncio.sleep(self.config.RETRY_DELAY)
            except Exception as e:
                self.logger.error(f"抓取失败 {url}: {str(e)}")
                return None

        self.logger.error(f"抓取多次失败 {url}")
        return None

    async def process_urls(self, urls: List[str], output_file: str) -> List[Dict]:
        """处理一批URL并将结果写入output_file"""
        connector = TCPConnector(limit=self.config.CONCURRENCY_LIMIT)
        headers = {'User-Agent': self.user_agent}
        timeout = ClientTimeout(total=self.config.REQUEST_TIMEOUT)

        results = []
        async with aiohttp.ClientSession(connector=connector, headers=headers, timeout=timeout) as session:
            tasks = [asyncio.create_task(self.fetch_and_parse(url, session)) for url in urls]
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Crawling"):
                result = await f
                if result:
                    results.append(result)

        if results:
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(ujson.dumps(r, ensure_ascii=False) + '\n')

        return results

    def run_process(self, urls: List[str], output_file: str):
        """进程入口函数"""
        try:
            self.logger.info(f"子进程开始处理 {len(urls)} 个URL, 输出文件: {output_file}")
            asyncio.run(self.process_urls(urls, output_file))
            self.logger.info(f"子进程完成: {output_file}")
        except Exception as e:
            self.logger.error(f"进程错误: {str(e)}", exc_info=True)

    def crawl(self, urls: List[str]) -> None:
        """启动爬虫主流程"""
        try:
            os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

            num_processes = min(cpu_count(), self.config.MAX_PROCESSES)
            chunk_size = len(urls) // num_processes if num_processes > 0 else len(urls)

            processes = []
            partial_files = []

            for i in range(num_processes):
                start = i * chunk_size
                end = start + chunk_size if i < num_processes - 1 else len(urls)
                urls_chunk = urls[start:end]

                part_file = os.path.join(self.config.OUTPUT_DIR, f"part_{i}.jsonl")
                if os.path.exists(part_file):
                    os.remove(part_file)

                p = Process(target=self.run_process, args=(urls_chunk, part_file))
                p.start()
                processes.append(p)
                partial_files.append(part_file)

            for p in processes:
                p.join()

            self._merge_results(partial_files)

        except Exception as e:
            self.logger.error(f"爬虫运行错误: {str(e)}", exc_info=True)

    def _merge_results(self, partial_files: List[str]) -> None:
        """合并结果文件"""
        merged_data = []
        seen_urls = set()

        for pf in partial_files:
            if os.path.exists(pf):
                with open(pf, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = ujson.loads(line.strip())
                            if data['url'] not in seen_urls:
                                seen_urls.add(data['url'])
                                merged_data.append(data)
                        except Exception as e:
                            self.logger.error(f"解析行失败: {str(e)}")
                os.remove(pf)

        # 使用时间戳生成文件名
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        final_json = os.path.join(self.config.RAW_DIR, f'xywy_qa_{timestamp}.json')
        
        # 保存到raw目录
        with open(final_json, 'w', encoding='utf-8') as f:
            ujson.dump(merged_data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"原始数据已保存: {len(merged_data)}条 -> {final_json}")
  