import json
import torch
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel
from collections import defaultdict
from typing import Dict, List, Set
from tqdm import tqdm
import jieba
import re
import os
import time

class MedicalDataProcessor:
    def __init__(self, model_name='bert-base-chinese'):
        """初始化医疗数据处理器"""
        # 设置项目路径
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.raw_dir = os.path.join(self.project_root, 'data', 'raw')
        self.processed_dir = os.path.join(self.project_root, 'data', 'processed')
        self.model_dir = os.path.join(self.project_root, 'data', 'model')
        
        # 确保目录存在
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # 初始化科室关键词（保持原有的关键词字典不变）
        self.department_keywords = {
            '内科': [
                # 血管相关
                '心脏', '心慌', '心悸', '胸闷', '心血管', '高血压', '血压', '动脉', '血管',
                # 呼吸系统
                '肺', '呼吸', '咳嗽', '哮喘', '支气管', '胸痛', '呼吸困难',
                # 消化系统
                '胃', '肝', '消化', '腹痛', '胃痛', '肠', '食管', '胆', '消化不良',
                # 内分泌系统
                '糖尿病', '内分泌', '甲状腺', '代谢',
                # 神经系统
                '头晕', '头痛', '眩晕', '晕厥', '脑供血', '脑梗', '脑血管', '神经痛',
                '偏头痛', '脑中风', '脑出血', '记忆力'
            ],
            '外科': [
                # 骨科
                '骨折', '关节', '脊椎', '腰椎', '颈椎', '骨头', '韧带', '腱', '骨质疏松',
                '腿疼', '膝盖', '腰疼', '颈椎病', '椎间盘', '腰间盘',
                # 普外科
                '手术', '外伤', '创伤', '疝气', '阑尾炎', '创口', '伤口', '肿瘤',
                '疼痛', '出血', '炎症'
            ],
            '妇产科': [
                '月经', '经期', '闭经', '痛经', '经血', '白带',
                '怀孕', '孕期', '产后', '产前', '妊娠', '流产', '避孕',
                '宫颈', '子宫', '卵巢', '输卵管', '乳腺', '乳房',
                '妇科', '产科', '不孕', '更年期'
            ],
            '皮肤科': [
                '皮肤', '痘痘', '痤疮', '湿疹', '皮疹', '荨麻疹', '过敏', '瘙痒',
                '皮炎', '脱发', '毛发', '皮屑', '痒', '皮肤病', '斑', '疣',
                '皮肤瘙痒', '皮肤过敏', '皮肤感染', '皮肤发炎'
            ],
            '性病科': [
                '性病', '艾滋', 'HIV', '梅毒', '淋病', '生殖器', '尖锐湿疣',
                '包皮', '性传播', 'std', '传染', '生殖器疱疹', '龟头炎',
                '阴道炎', '前列腺炎'
            ],
            '儿科': [
                '儿童', '婴儿', '新生儿', '小儿', '幼儿', '宝宝', '婴幼儿',
                '小孩', '母乳', '奶粉', '发育', '儿童生长', '儿童发育',
                '儿童营养', '儿童免疫', '儿童疫苗', '儿童感冒', '儿童发烧',
                '儿童咳嗽', '儿童腹泻'
            ],
            '五官科': [
                # 眼科
                '眼睛', '视力', '近视', '远视', '白内障', '青光眼', '结膜炎',
                '眼压', '眼底', '眼球', '眼科',
                # 耳鼻喉科
                '耳鼻喉', '鼻炎', '咽喉', '中耳炎', '扁桃体', '鼻窦', '耳鸣',
                '鼻塞', '咽喉炎', '声带', '听力',
                # 口腔科
                '牙齿', '口腔', '牙龈', '蛀牙', '牙周', '口腔溃疡'
            ],
            '中医科': [
                '中医', '中药', '针灸', '推拿', '艾灸', '养生', '调理',
                '气血', '阴阳', '经络', '脉象', '虚火', '肝火', '肾虚',
                '阳虚', '阴虚', '痰湿', '气虚', '血虚', '心火', '肝阳'
            ],
            '精神科': [
                '精神', '抑郁', '焦虑', '失眠', '躁狂', '精神病', '幻觉',
                '妄想', '自杀', '强迫', '神经衰弱', '精神分裂', '双相情感',
                '恐惧症', '睡眠障碍', '情绪低落', '注意力', '多动症'
            ],
            '心理科': [
                '心理', '情绪', '压力', '心理咨询', '心理治疗', '心理障碍',
                '心理健康', '心理问题', '心理创伤', '心理疾病', '心理干预',
                '心理辅导', '心理调节', '心理状态'
            ],
            '传染科': [
                '传染', '病毒', '细菌', '感染', '发烧', '流感', '肺炎',
                '传染病', '发热', '疫苗', '免疫', '新冠', '乙肝', '丙肝',
                '艾滋病', '结核', '带状疱疹', '水痘', '麻疹', '流行性感冒'
            ],
            '整形美容科': [
                '整形', '美容', '整容', '美白', '丰胸', '隆鼻', '整形手术',
                '美容手术', '瘦脸', '抗衰', '除皱', '眼袋', '双眼皮',
                '注射', '填充', '吸脂', '隆胸', '面部提升', '激光美容'
            ],
            '老年科': [
                '老年', '养老', '老人', '高龄', '老年人', '老年病',
                '老年痴呆', '帕金森', '阿尔茨海默', '老年护理', '老年保健',
                '老年营养', '老年康复', '老年综合征'
            ],
            '男科': [
                '前列腺', '阳痿', '早泄', '男性', '勃起', '精子', '男科',
                '睾丸', '包皮', '性功能', '前列腺炎', '精索静脉曲张',
                '男性不育', '男性更年期', '前列腺增生'
            ],
            '生殖科': [
                '不孕', '生殖', '试管婴儿', '人工受精', '卵子', '精子',
                '受精', '胚胎', '助孕', '生殖系统', '生殖功能', '卵',
                '月经', '子宫', '卵巢', '输卵管', '宫腔', '胎儿'
            ],
            '其他科': []
        }
        
        # 加载最新的原始数据
        self.data = self._load_latest_data()
        self._init_medical_dict()
        self.keyword_weights = self._init_keyword_weights()
        self.exclusion_rules = self._init_exclusion_rules()
        self.department_embeddings = self._precompute_department_embeddings()

    def _load_latest_data(self) -> List[Dict]:
        """加载最新的JSON数据文件"""
        # 查找最新的问答数据文件（以xywy_qa_开头的json文件）
        json_files = [f for f in os.listdir(self.raw_dir) 
                     if f.startswith('xywy_qa_') and f.endswith('.json')]
        
        if not json_files:
            raise FileNotFoundError("No QA data files found in raw data directory")
        
        # 获取最新的数据文件
        latest_file = max(json_files, key=lambda x: os.path.getctime(os.path.join(self.raw_dir, x)))
        data_path = os.path.join(self.raw_dir, latest_file)
        print(f"Loading QA data from: {data_path}")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 验证数据格式
                if isinstance(data, list) and all(isinstance(item, dict) and 'question' in item and 'answer' in item for item in data):
                    return data
                else:
                    raise ValueError("Invalid data format: expecting list of QA pairs")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def save_classified_data(self, classified_data: Dict[str, List[Dict]]):
        """保存分类后的数据"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON格式
        json_path = os.path.join(self.processed_dir, f'classified_qa_{timestamp}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(classified_data, f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats_path = os.path.join(self.processed_dir, f'classification_stats_{timestamp}.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("分类统计信息:\n")
            for dept, qa_list in classified_data.items():
                f.write(f"{dept}: {len(qa_list)}条\n")
        
        print(f"分类数据已保存至: {json_path}")
        print(f"统计信息已保存至: {stats_path}")

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """获取文本的BERT嵌入"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', 
                                  truncation=True, max_length=512,
                                  padding=True).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :]

    def _precompute_department_embeddings(self) -> Dict[str, torch.Tensor]:
        """预计算每个科室关键词的嵌入"""
        embeddings = {}
        for dept, keywords in self.department_keywords.items():
            dept_text = ' '.join(keywords)
            embeddings[dept] = self._get_text_embedding(dept_text)
        return embeddings

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """使用BERT计算文本相似度"""
        emb1 = self._get_text_embedding(text1)
        emb2 = self._get_text_embedding(text2)
        similarity = F.cosine_similarity(emb1, emb2)
        return similarity.item()

    def _classify_text(self, text: str) -> str:
        """使用BERT进行文本分类"""
        processed_text = self._preprocess_text(text.lower())
        text_embedding = self._get_text_embedding(processed_text)
        
        scores = {}
        for dept, dept_embedding in self.department_embeddings.items():
            similarity = F.cosine_similarity(text_embedding, dept_embedding)
            keyword_score = self._calculate_keyword_score(
                processed_text, dept, 
                self.department_keywords[dept]
            )
            scores[dept] = (similarity.item() * 0.7 + keyword_score * 0.3)
        
        scores = self._apply_exclusion_rules(scores)
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return '其他科'

    def classify_qa_pairs(self) -> Dict[str, List[Dict]]:
        """批量处理问答对分类"""
        if not self.data:
            raise ValueError("No data loaded")
        
        classified_data = defaultdict(list)
        
        print(f"开始处理 {len(self.data)} 条问答对...")
        batch_size = 32
        for i in tqdm(range(0, len(self.data), batch_size)):
            batch = self.data[i:i + batch_size]
            batch_texts = [
                f"{qa.get('question', '')} {qa.get('question', '')} {qa.get('answer', '')[:200]}"
                for qa in batch
            ]
            
            for text, qa_pair in zip(batch_texts, batch):
                department = self._classify_text(text)
                classified_data[department].append(qa_pair)
        
        return dict(classified_data)

    def _init_medical_dict(self):
        """初始化医学词典"""
        for keywords in self.department_keywords.values():
            for keyword in keywords:
                jieba.add_word(keyword)
    
    def _init_keyword_weights(self) -> Dict[str, Dict[str, float]]:
        """初始化关键词权重"""
        weights = {}
        for dept, keywords in self.department_keywords.items():
            weights[dept] = {}
            for keyword in keywords:
                weight = 1.0
                if len(keyword) > 2:
                    weight *= 1.5
                if '病' in keyword or '炎' in keyword:
                    weight *= 1.3
                weights[dept][keyword] = weight
        return weights
    
    def _init_exclusion_rules(self) -> Dict[str, Set[str]]:
        """初始化互斥规则"""
        return {
            '儿科': {'老年科'},
            '男科': {'妇产科'},
            '精神科': {'心理科'},
            '内科': {'外科'},
        }

    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _calculate_keyword_score(self, text: str, dept: str, keywords: List[str]) -> float:
        """计算加权的关键词匹配得分"""
        score = 0
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword in text_lower:
                base_score = text_lower.count(keyword) * self.keyword_weights[dept][keyword]
                question_end = text_lower.find('?') if '?' in text_lower else len(text_lower)//2
                keyword_pos = text_lower.find(keyword)
                if keyword_pos < question_end:
                    base_score *= 1.5
                
                context_start = max(0, keyword_pos - 10)
                context_end = min(len(text_lower), keyword_pos + len(keyword) + 10)
                context = text_lower[context_start:context_end]
                
                for related_keyword in keywords:
                    if related_keyword != keyword and related_keyword in context:
                        base_score *= 1.2
                
                score += base_score
        
        return score

    def _apply_exclusion_rules(self, scores: Dict[str, float]) -> Dict[str, float]:
        """应用互斥规则"""
        result = scores.copy()
        max_dept = max(scores.items(), key=lambda x: x[1])[0]
        
        if max_dept in self.exclusion_rules:
            for excluded_dept in self.exclusion_rules[max_dept]:
                result[excluded_dept] *= 0.5
        
        return result

def main():
    processor = MedicalDataProcessor()
    print("正在处理数据...")
    classified_data = processor.classify_qa_pairs()
    processor.save_classified_data(classified_data)

if __name__ == "__main__":
    main()