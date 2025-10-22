"""
Ragas 评估器
用于评估 RAG 系统的检索和生成质量
"""

import os
import logging
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
    answer_similarity,
)
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
import pandas as pd

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """Ragas 评估器"""
    
    def __init__(
        self, 
        llm_model: str = "qwen-turbo",
        embedding_model: str = "text-embedding-v3",
        api_key: str = None
    ):
        """
        初始化评估器
        
        Args:
            llm_model: LLM 模型名称
            embedding_model: Embedding 模型名称
            api_key: API 密钥
        """
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
        
        self.llm = Tongyi(model=llm_model)
        self.embeddings = DashScopeEmbeddings(model=embedding_model)
        
        # 定义评估指标
        self.metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness,
            answer_similarity,
        ]
        
        logger.info(f"初始化 Ragas 评估器: LLM={llm_model}, Embedding={embedding_model}")
    
    def evaluate_batch(
        self, 
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> pd.DataFrame:
        """
        批量评估
        
        Args:
            questions: 问题列表
            answers: 答案列表
            contexts: 上下文列表（每个问题对应多个上下文）
            ground_truths: 标准答案列表
            
        Returns:
            DataFrame: 评估结果
        """
        logger.info(f"开始批量评估 {len(questions)} 个问题...")
        
        # 准备数据集
        dataset = Dataset.from_dict({
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        })
        
        try:
            # 执行评估
            result = evaluate(
                dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False
            )

            # 转换为 DataFrame
            df_result = result.to_pandas()
            logger.info("批量评估完成")
            
            return df_result
            
        except Exception as e:
            logger.error(f"批量评估失败: {e}")
            raise
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """
        单个问题评估
        
        Args:
            question: 问题
            answer: 答案
            contexts: 上下文列表
            ground_truth: 标准答案
            
        Returns:
            Dict: 评估指标
        """
        logger.info(f"评估单个问题: {question[:50]}...")
        
        # 准备单个数据集
        dataset = Dataset.from_dict({
            'question': [question],
            'answer': [answer],
            'contexts': [contexts],
            'ground_truth': [ground_truth]
        })
        
        try:
            # 执行评估
            result = evaluate(
                dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False
            )
            
            # 转换为字典
            metrics_dict = result.to_pandas().iloc[0].to_dict()
            logger.info("单个评估完成")
            
            return metrics_dict
            
        except Exception as e:
            logger.error(f"单个评估失败: {e}")
            raise
    
    def evaluate_with_metadata(
        self,
        test_data: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        带元数据的评估
        
        Args:
            test_data: 测试数据列表，每条包含:
                {
                    'question': str,
                    'answer': str,
                    'contexts': List[str],
                    'ground_truth': str,
                    'search_type': str,
                    'response_time': float,
                    ...其他元数据
                }
        
        Returns:
            DataFrame: 包含评估指标和元数据的结果
        """
        logger.info(f"开始评估 {len(test_data)} 条测试数据...")
        
        # 提取评估所需数据
        questions = [d['question'] for d in test_data]
        answers = [d['answer'] for d in test_data]
        contexts = [d['contexts'] for d in test_data]
        ground_truths = [d['ground_truth'] for d in test_data]
        
        # 执行评估
        df_metrics = self.evaluate_batch(questions, answers, contexts, ground_truths)
        
        # 添加 question 列（如果不存在）
        if 'question' not in df_metrics.columns:
            df_metrics['question'] = questions
        
        # 添加其他元数据
        for key in test_data[0].keys():
            if key not in ['question', 'answer', 'contexts', 'ground_truth']:
                df_metrics[key] = [d.get(key) for d in test_data]
        
        # 重新排列列顺序，把重要列放在前面
        # 优先顺序：question > 元数据 > 指标
        priority_cols = ['question', 'search_type', 'response_time']
        
        # 只选择存在的优先列
        existing_priority_cols = [col for col in priority_cols if col in df_metrics.columns]
        
        # 其他列（指标列）
        other_cols = [col for col in df_metrics.columns if col not in existing_priority_cols]
        
        # 合并列顺序
        ordered_cols = existing_priority_cols + other_cols
        df_metrics = df_metrics[ordered_cols]
        
        logger.info("评估完成")
        return df_metrics
    
    def calculate_summary_stats(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        """
        计算汇总统计
        
        Args:
            df_results: 评估结果 DataFrame
            
        Returns:
            Dict: 汇总统计
        """
        metric_columns = [
            'context_precision',
            'context_recall',
            'faithfulness',
            'answer_relevancy',
            'answer_correctness',
            'answer_similarity'
        ]
        
        # 过滤出存在的指标列
        available_metrics = [col for col in metric_columns if col in df_results.columns]
        
        summary = {}
        for metric in available_metrics:
            summary[metric] = {
                'mean': df_results[metric].mean(),
                'std': df_results[metric].std(),
                'min': df_results[metric].min(),
                'max': df_results[metric].max(),
                'median': df_results[metric].median()
            }
        
        # 计算整体得分（所有指标的平均值）
        if available_metrics:
            summary['overall_score'] = {
                'mean': df_results[available_metrics].mean().mean(),
                'std': df_results[available_metrics].mean().std()
            }
        
        return summary
    
    def compare_search_types(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """
        对比不同检索方式的效果
        
        Args:
            df_results: 评估结果 DataFrame（包含 search_type 列）
            
        Returns:
            DataFrame: 按检索类型分组的统计结果
        """
        if 'search_type' not in df_results.columns:
            logger.warning("结果中没有 search_type 列，无法进行对比")
            return pd.DataFrame()
        
        metric_columns = [
            'context_precision',
            'context_recall',
            'faithfulness',
            'answer_relevancy',
            'answer_correctness',
            'answer_similarity',
            'response_time'
        ]
        
        # 过滤出存在的指标列
        available_metrics = [col for col in metric_columns if col in df_results.columns]
        
        # 按检索类型分组统计
        comparison = df_results.groupby('search_type')[available_metrics].agg(['mean', 'std'])
        
        return comparison


# 示例用法
if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    
    # 设置 API Key
    os.environ["DASHSCOPE_API_KEY"] = "sk-0ef222e3c3d14e8d895aec2f2a16b4aa"
    
    # 创建评估器
    evaluator = RagasEvaluator()
    
    # 测试数据
    test_data = [
        {
            'question': '机器学习的主要应用领域有哪些？',
            'answer': '机器学习应用于金融风控、医疗诊断、推荐系统和图像识别。',
            'contexts': [
                "机器学习在金融风控应用领域中用于风险预测。",
                "机器学习在医疗诊断应用领域中可以帮助识别疾病模式。"
            ],
            'ground_truth': '机器学习通过数据训练模型并应用于金融风控、医疗、推荐系统和图像识别等领域。',
            'search_type': 'vector',
            'response_time': 0.5
        }
    ]
    
    # 执行评估
    results = evaluator.evaluate_with_metadata(test_data)
    print("\n评估结果:")
    print(results)
    
    # 计算汇总统计
    summary = evaluator.calculate_summary_stats(results)
    print("\n汇总统计:")
    for metric, stats in summary.items():
        print(f"{metric}: {stats}")

