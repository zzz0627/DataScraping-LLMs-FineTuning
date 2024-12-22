import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import logging
from pathlib import Path


class MedicalDataVisualizer:
    def __init__(self):
        """初始化可视化器"""
        self.setup_directories()
        self.setup_logging()

    def setup_directories(self):
        """设置目录结构"""
        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        
        # 设置数据目录
        self.data_dir = os.path.join(self.project_root, 'data')
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.vis_dir = os.path.join(self.processed_dir, 'visualizations')
        
        # 创建可视化结果目录
        os.makedirs(self.vis_dir, exist_ok=True)

    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.processed_dir, 'visualizer.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, file_path: str) -> dict:
        """加载分类后的数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"找不到输入文件: {file_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"输入文件格式错误: {file_path}")
            raise

    def create_visualization(self, data: dict, output_path: str):
        """创建数据可视化"""
        # 准备数据
        stats = {dept: len(qa_pairs) for dept, qa_pairs in data.items()}
        total = sum(stats.values())
        
        # 创建DataFrame并排序
        df = pd.DataFrame({
            'department': list(stats.keys()),
            'count': list(stats.values())
        }).sort_values('count', ascending=False)
        
        df['percentage'] = df['count'] / total * 100
        
        # 创建子图布局
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=('各科室问答数量分布', '各科室问答占比'),
            column_widths=[0.6, 0.4]
        )
        
        # 添加柱状图
        fig.add_trace(
            go.Bar(
                x=df['count'],
                y=df['department'],
                orientation='h',
                marker=dict(
                    color=df['count'],
                    colorscale='Viridis'
                ),
                text=df['count'],
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # 添加环形图
        fig.add_trace(
            go.Pie(
                labels=df['department'],
                values=df['count'],
                hole=0.4,
                textinfo='label+percent',
                textposition='outside',
                pull=[0.1 if i < 5 else 0 for i in range(len(df))],
                marker=dict(
                    colors=px.colors.qualitative.Set3
                )
            ),
            row=1, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text="医疗问答数据分布",
            showlegend=False,
            width=1500,
            height=800,
            template='plotly_white',
            font=dict(size=12)
        )
        
        # 更新x轴和y轴的标签
        fig.update_xaxes(title_text="问答对数量", row=1, col=1)
        fig.update_yaxes(title_text="科室", row=1, col=1)
        
        # 只保存HTML格式
        try:
            fig.write_html(output_path)
            self.logger.info(f"可视化结果已保存: {output_path}")
            
            # 打印简要统计信息
            self.logger.info(f"总问答对数量: {total}")
            
        except Exception as e:
            self.logger.error(f"保存可视化结果时出错: {str(e)}")
            raise

def get_latest_classified_file(processed_dir: str) -> str:
    """获取最新的分类数据文件"""
    files = [f for f in os.listdir(processed_dir) 
             if f.startswith('classified_qa_') and f.endswith('.json')]
    
    if not files:
        raise FileNotFoundError("未找到分类数据文件")
        
    # 按文件名中的时间戳排序
    latest_file = sorted(files, reverse=True)[0]
    return latest_file

def parse_args():
    parser = argparse.ArgumentParser(description='医疗问答数据可视化工具')
    parser.add_argument('--input', '-i', type=str,
                       help='输入文件路径（相对于processed目录，默认使用最新的分类数据文件）')
    parser.add_argument('--output', '-o', type=str,
                       help='输出文件名（将保存在visualizations目录下，默认使用与输入文件对应的名称）')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        visualizer = MedicalDataVisualizer()
        
        # 获取输入文件路径
        if args.input:
            input_file = args.input
        else:
            # 自动获取最新的分类数据文件
            input_file = get_latest_classified_file(visualizer.processed_dir)
        input_path = os.path.join(visualizer.processed_dir, input_file)
        
        # 设置输出文件名
        if args.output:
            output_name = args.output
        else:
            # 使用输入文件名对应的输出名称
            timestamp = input_file.replace('classified_qa_', '').replace('.json', '')
            output_name = f'medical_qa_distribution_{timestamp}.html'
        output_path = os.path.join(visualizer.vis_dir, output_name)
        
        # 执行可视化
        visualizer.logger.info(f"使用输入文件: {input_file}")
        data = visualizer.load_data(input_path)
        visualizer.create_visualization(data, output_path)
        
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()