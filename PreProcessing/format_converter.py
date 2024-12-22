import json
import os
import random
from typing import List, Dict
import argparse
import logging
from pathlib import Path


class QwenFormatConverter:
    def __init__(self, input_file: str):
        """初始化转换器"""
        self.setup_directories()
        self.input_file = input_file
        self.setup_logging()
        self.department_prompts = {
            '内科': '你是一位经验丰富的内科医生，专长于内科疾病的诊断和治疗。请以专业、严谨且富有同理心的态度回答患者的问题。',
            '外科': '你是一位资深外科医生，擅长各类外科手术和治疗。请以专业、谨慎且负责任的态度为患者解答问题。',
            '妇产科': '你是一位专业的妇产科医生，熟悉妇科疾病和孕产保健。请以专业、细心且体贴的方式回答患者的问题。',
            '皮肤科': '你是一位专业的皮肤科医生，擅长诊治各类皮肤疾病。请以专业、耐心的态度为患者分析病情并提供建议。',
            '五官科': '你是一位经验丰富的五官科医生，专注于耳鼻喉和眼科疾病。请以专业、细致的态度回答患者的问题。',
            '中医科': '你是一位资深的中医科医生，精通中医理论和治疗方法。请以专业、系统且通俗易懂的方式为患者解答问题。',
            '精神科': '你是一位专业的精神科医生，擅长心理和精神疾病的诊治。请以专业、理解且支持的态度回应患者的问题。',
            '心理科': '你是一位专业的心理科医生，擅长心理咨询和治疗。请以专业、共情且支持的方式帮助患者。',
            '传染科': '你是一位专业的传染科医生，熟悉各类传染病的诊断和防治。请以专业、负责的态度回答患者的问题。',
            '整形美容科': '你是一位专业的整形美容科医生，擅长医疗美容和整形手术。请以专业、谨慎的态度为患者提供建议。',
            '老年科': '你是一位经验丰富的老年科医生，专注于老年人疾病的诊治。请以专业、耐心且体贴的方式回答问题。',
            '男科': '你是一位专业的男科医生，擅长男性疾病的诊治。请以专业、谨慎且理解的态度回应患者的问题。',
            '生殖科': '你是一位专业的生殖科医生，擅长不孕不育等生殖问题的治疗。请以专业、细致的方式为患者解答问题。',
            '儿科': '你是一位富有经验的儿科医生，专注于儿童疾病的诊治。请以专业、耐心且关爱的态度回答问题。',
            '性病科': '你是一位专业的性病科医生，擅长性传播疾病的诊治。请以专业、谨慎且理解的态度回应患者的问题。'
        }

    def setup_directories(self):
        """设置目录结构"""
        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        
        # 设置数据目录
        self.data_dir = os.path.join(self.project_root, 'data')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.model_dir = os.path.join(self.data_dir, 'model')
        
        # 创建必要的目录
        os.makedirs(self.model_dir, exist_ok=True)

    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.processed_dir, 'format_converter.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> Dict:
        """加载分类后的数据"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"找不到输入文件: {self.input_file}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"输入文件格式错误: {self.input_file}")
            raise

    def is_valid_qa(self, question: str, answer: str) -> bool:
        """检查问答对的质量"""
        # 1. 基本长度检查
        if len(question.strip()) < 1 or len(answer.strip()) < 10:
            return False
        
        # 2. 内容质量检查
        low_quality_keywords = ['咨询电话', '热线', 'QQ群', '微信号']
        if any(keyword in answer for keyword in low_quality_keywords):
            return False
        
        return True

    def convert_to_alpaca_format(self, max_samples_per_dept: int = None) -> List[Dict]:
        """转换为通义千问对话格式"""
        data = self.load_data()
        dialogue_format_data = []
        total_original = sum(len(pairs) for pairs in data.values())
        total_converted = 0
        dept_stats = {}

        for dept, qa_pairs in data.items():
            original_count = len(qa_pairs)
            dept_stats[dept] = {'original': original_count}
            
            # 先进行数据质量过滤
            qa_pairs = [qa for qa in qa_pairs 
                       if self.is_valid_qa(qa['question'], qa['answer'])]
            dept_stats[dept]['filtered'] = len(qa_pairs)
            
            # 如果设置了每个科室的最大样本数，随机采样
            if max_samples_per_dept and len(qa_pairs) > max_samples_per_dept:
                qa_pairs = random.sample(qa_pairs, max_samples_per_dept)
            dept_stats[dept]['final'] = len(qa_pairs)
            
            for qa in qa_pairs:
                formatted_item = {
                    "id": f"{dept}_{total_converted}",
                    "conversations": [
                        {
                            "from": "user",
                            "value": qa['question']
                        },
                        {
                            "from": "assistant",
                            "value": qa['answer']
                        }
                    ]
                }
                dialogue_format_data.append(formatted_item)
                total_converted += 1

        # 打印详细的统计信息
        self.logger.info(f"\n=== 数据转换统计 ===")
        self.logger.info(f"原始数据总量: {total_original} 条")
        for dept, stats in dept_stats.items():
            filtered_msg = f"{dept}: {stats['original']} -> {stats['filtered']} (质量过滤)"
            if max_samples_per_dept and stats['filtered'] > max_samples_per_dept:
                filtered_msg += f" -> {stats['final']} (采样限制)"
            self.logger.info(filtered_msg)
        self.logger.info(f"最终数据量: {total_converted} 条")

        return dialogue_format_data

    def save_formatted_data(self, output_file: str, max_samples_per_dept: int = None):
        """保存转换后的数据为json格式"""
        try:
            formatted_data = self.convert_to_alpaca_format(max_samples_per_dept)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存为json格式
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"数据已保存至: {output_file}")
            
        except Exception as e:
            self.logger.error(f"保存数据时出错: {str(e)}")
            raise

def get_latest_classified_file(processed_dir: str) -> str:
    """获取最新的分类数据文件"""
    files = [f for f in os.listdir(processed_dir) 
             if f.startswith('classified_qa_') and f.endswith('.json')]
    
    if not files:
        raise FileNotFoundError("未找到分类数据文件")
        
    # 文件名中的时间戳排序
    latest_file = sorted(files, reverse=True)[0]
    return latest_file

def parse_args():
    parser = argparse.ArgumentParser(description='将分类后的医疗问答数据转换为Alpaca训练格式')
    parser.add_argument('--input', '-i', type=str,
                       help='输入文件路径（相对于processed目录，默认使用最新的分类数据文件）')
    parser.add_argument('--output', '-o', type=str,
                       help='输出文件名（将保存在model目录下，默认格式：alpaca_medical_qa_时间戳.jsonl）')
    parser.add_argument('--max-samples', '-m', type=int,
                       default=5000,
                       help='每个科室的最大样本数（默认：5000）')
    parser.add_argument('--seed', '-s', type=int,
                       default=42,
                       help='随机种子（默认：42）')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # 首先创建必要的目录结构
        project_root = os.path.dirname(os.path.dirname(__file__))
        processed_dir = os.path.join(project_root, 'data', 'processed')
        model_dir = os.path.join(project_root, 'data', 'model')
        
        # 获取输入文件路径
        if args.input:
            input_file = args.input
        else:
            # 自动获取最新的分类数据文件
            input_file = get_latest_classified_file(processed_dir)
        input_path = os.path.join(processed_dir, input_file)
        
        # 设置输出文件名
        if args.output:
            output_name = args.output
        else:
            # 使用输入文件名对应的输出名称
            timestamp = input_file.replace('classified_qa_', '').replace('.json', '')
            output_name = f'alpaca_medical_qa_{timestamp}.json'
        output_path = os.path.join(model_dir, output_name)
        
        # 设置随机种子
        random.seed(args.seed)
        
        # 创建转换器实例（使用完整的输入文件路径）
        converter = QwenFormatConverter(input_path)
        
        # 执行转换
        converter.logger.info(f"使用输入文件: {input_file}")
        converter.save_formatted_data(output_path, args.max_samples)
        
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()