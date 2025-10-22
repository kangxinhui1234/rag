"""
测试执行器
协调测试集加载、Spring AI 调用和 Ragas 评估
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from spring_ai_client import SpringAIClient, RagResponse
from ragas_evaluator import RagasEvaluator

logger = logging.getLogger(__name__)


class TestRunner:
    """测试执行器"""
    
    def __init__(
        self,
        spring_ai_url: str,
        api_key: str = None,
        timeout: int = 30
    ):
        """
        初始化测试执行器
        
        Args:
            spring_ai_url: Spring AI 服务 URL
            api_key: DashScope API Key
            timeout: 请求超时时间
        """
        self.spring_client = SpringAIClient(spring_ai_url, timeout)
        self.ragas_evaluator = RagasEvaluator(api_key=api_key)
        
        logger.info(f"测试执行器初始化完成: {spring_ai_url}")
    
    def load_testset(self, testset_path: str) -> List[Dict[str, Any]]:
        """
        加载测试集
        
        Args:
            testset_path: 测试集文件路径
            
        Returns:
            List[Dict]: 测试数据列表
        """
        logger.info(f"加载测试集: {testset_path}")
        
        with open(testset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持两种格式
        if isinstance(data, list):
            testset = data
        elif isinstance(data, dict) and 'testset' in data:
            testset = data['testset']
        else:
            raise ValueError("不支持的测试集格式")
        
        logger.info(f"加载了 {len(testset)} 条测试数据")
        return testset
    
    def run_single_test(
        self,
        question: str,
        ground_truth: str,
        ground_truth_contexts: List[str],
        search_type: str = "hybrid",
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        运行单个测试
        
        Args:
            question: 问题
            ground_truth: 标准答案
            ground_truth_contexts: 标准上下文
            search_type: 检索类型
            top_k: 返回文档数量
            **kwargs: 其他参数（如 vector_weight, bm25_weight）
            
        Returns:
            Dict: 测试结果
        """
        try:
            # 调用 Spring AI
            rag_response = self.spring_client.rag_query(
                question=question,
                search_type=search_type,
                top_k=top_k
            )
            
            # 组装结果
            result = {
                'question': question,
                'answer': rag_response.answer,
                'contexts': [rag_response.contexts],
                'ground_truth': ground_truth,
                'ground_truth_contexts': ground_truth_contexts,
                'search_type': search_type,
                'response_time': rag_response.response_time,
                'status': 'success'
            }
            
            # 添加额外参数
            result.update(kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"测试失败: {question[:50]}... - {e}")
            return {
                'question': question,
                'answer': '',
                'contexts': [],
                'ground_truth': ground_truth,
                'ground_truth_contexts': ground_truth_contexts,
                'search_type': search_type,
                'response_time': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_test_suite(
        self,
        testset: List[Dict[str, Any]],
        search_configs: List[Dict[str, Any]],
        delay_between_requests: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        运行测试套件
        
        Args:
            testset: 测试集
            search_configs: 检索配置列表，例如:
                [
                    {'name': 'vector', 'type': 'vector'},
                    {'name': 'bm25', 'type': 'bm25'},
                    {'name': 'hybrid_0.7_0.3', 'type': 'hybrid', 'vector_weight': 0.7, 'bm25_weight': 0.3}
                ]
            delay_between_requests: 请求之间的延迟（秒）
            
        Returns:
            List[Dict]: 所有测试结果
        """
        all_results = []
        
        total_tests = len(testset) * len(search_configs)
        logger.info(f"开始测试: {len(testset)} 个问题 × {len(search_configs)} 种配置 = {total_tests} 次测试")
        
        with tqdm(total=total_tests, desc="运行测试") as pbar:
            for config in search_configs:
                config_name = config.get('name', config['type'])
                logger.info(f"\n测试配置: {config_name}")
                
                for test_case in testset:
                    question = test_case.get('question', '')
                    ground_truth = test_case.get('ground_truth', '')
                    ground_truth_contexts = test_case.get('ground_truth_contexts', [])
                    
                    # 运行测试
                    result = self.run_single_test(
                        question=question,
                        ground_truth=ground_truth,
                        ground_truth_contexts=ground_truth_contexts,
                        search_type=config['type'],
                        top_k=config.get('top_k', 5),
                        config_name=config_name,
                        **{k: v for k, v in config.items() if k not in ['name', 'type', 'top_k']}
                    )
                    
                    # 添加测试用例的元数据
                    if 'metadata' in test_case:
                        result['test_metadata'] = test_case['metadata']
                    
                    all_results.append(result)
                    pbar.update(1)
                    
                    # 延迟避免请求过快
                    if delay_between_requests > 0:
                        time.sleep(delay_between_requests)
        
        logger.info(f"\n测试完成: 成功 {sum(1 for r in all_results if r['status'] == 'success')} / {len(all_results)}")
        
        return all_results
    
    def evaluate_results(
        self,
        test_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        评估测试结果
        
        Args:
            test_results: 测试结果列表
            
        Returns:
            DataFrame: 包含评估指标的结果
        """
        logger.info("开始评估测试结果...")
        
        # 过滤成功的测试
        successful_results = [r for r in test_results if r['status'] == 'success']
        
        if not successful_results:
            logger.error("没有成功的测试结果可供评估")
            return pd.DataFrame()
        
        # 使用 Ragas 评估
        df_evaluated = self.ragas_evaluator.evaluate_with_metadata(successful_results)
        
        return df_evaluated
    
    def run_full_test(
        self,
        testset_path: str,
        search_configs: List[Dict[str, Any]],
        output_dir: str = "results",
        save_raw_results: bool = True
    ) -> Dict[str, Any]:
        """
        运行完整测试流程
        
        Args:
            testset_path: 测试集路径
            search_configs: 检索配置列表
            output_dir: 输出目录
            save_raw_results: 是否保存原始结果
            
        Returns:
            Dict: 测试报告
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 加载测试集
        testset = self.load_testset(testset_path)
        
        # 2. 运行测试
        test_results = self.run_test_suite(testset, search_configs)
        
        # 保存原始结果
        if save_raw_results:
            raw_results_file = output_path / "raw_test_results.json"
            with open(raw_results_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            logger.info(f"原始结果已保存: {raw_results_file}")
        
        # 3. 评估结果
        df_evaluated = self.evaluate_results(test_results)
        
        # 保存评估结果
        csv_file = output_path / "evaluation_results.csv"
        df_evaluated.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"评估结果已保存: {csv_file}")
        
        json_file = output_path / "evaluation_results.json"
        df_evaluated.to_json(json_file, orient='records', force_ascii=False, indent=2)
        logger.info(f"评估结果已保存: {json_file}")
        
        # 4. 生成汇总统计
        summary_stats = self.ragas_evaluator.calculate_summary_stats(df_evaluated)
        
        # 5. 对比不同检索方式
        comparison = self.ragas_evaluator.compare_search_types(df_evaluated)
        
        # 保存对比结果
        comparison_file = output_path / "search_type_comparison.csv"
        comparison.to_csv(comparison_file, encoding='utf-8-sig')
        logger.info(f"对比结果已保存: {comparison_file}")
        
        # 6. 组装报告
        # 将 comparison DataFrame 转换为 JSON 可序列化的格式
        if not comparison.empty:
            # 重置索引，将多级列转换为字符串
            comparison_dict = {}
            for search_type in comparison.index:
                comparison_dict[search_type] = {}
                for col in comparison.columns:
                    # col 是 tuple，如 ('context_precision', 'mean')
                    if isinstance(col, tuple):
                        metric_name = f"{col[0]}_{col[1]}"  # 例如: "context_precision_mean"
                    else:
                        metric_name = str(col)
                    comparison_dict[search_type][metric_name] = float(comparison.loc[search_type, col])
        else:
            comparison_dict = {}
        
        report = {
            'test_summary': {
                'total_tests': len(test_results),
                'successful_tests': sum(1 for r in test_results if r['status'] == 'success'),
                'failed_tests': sum(1 for r in test_results if r['status'] == 'failed'),
                'test_configs': len(search_configs),
                'questions': len(testset)
            },
            'summary_statistics': summary_stats,
            'search_type_comparison': comparison_dict,
            'output_files': {
                'raw_results': str(output_path / "raw_test_results.json") if save_raw_results else None,
                'evaluation_csv': str(csv_file),
                'evaluation_json': str(json_file),
                'comparison': str(comparison_file)
            }
        }
        
        # 保存报告
        report_file = output_path / "test_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"测试报告已保存: {report_file}")
        
        return report
    
    def close(self):
        """关闭资源"""
        self.spring_client.close()


# 示例用法
if __name__ == "__main__":
    import os
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置
    SPRING_AI_URL = "http://localhost:8080"
    DASHSCOPE_API_KEY = "sk-0ef222e3c3d14e8d895aec2f2a16b4aa"
    TESTSET_PATH = "data/testsets/testset.json"
    
    # 检索配置
    search_configs = [
        {'name': 'vector_only', 'type': 'vector'},
        {'name': 'bm25_only', 'type': 'bm25'},
        {'name': 'hybrid_0.7_0.3', 'type': 'hybrid', 'vector_weight': 0.7, 'bm25_weight': 0.3},
    ]
    
    # 创建测试执行器
    runner = TestRunner(SPRING_AI_URL, api_key=DASHSCOPE_API_KEY)
    
    # 运行测试
    report = runner.run_full_test(
        testset_path=TESTSET_PATH,
        search_configs=search_configs,
        output_dir="results"
    )
    
    # 打印报告摘要
    print("\n" + "="*60)
    print("测试报告摘要")
    print("="*60)
    print(f"总测试数: {report['test_summary']['total_tests']}")
    print(f"成功: {report['test_summary']['successful_tests']}")
    print(f"失败: {report['test_summary']['failed_tests']}")
    print("\n整体得分:")
    if 'overall_score' in report['summary_statistics']:
        print(f"  平均分: {report['summary_statistics']['overall_score']['mean']:.3f}")
    
    runner.close()






