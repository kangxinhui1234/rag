"""
报告生成器
生成可视化报告和HTML报告
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, results_dir: str):
        """
        初始化报告生成器
        
        Args:
            results_dir: 结果目录
        """
        self.results_dir = Path(results_dir)
        
        # 加载评估结果
        self.df_results = None
        self.comparison = None
        self.report = None
        
        self._load_results()
    
    def _load_results(self):
        """加载结果文件"""
        try:
            # 加载评估结果
            csv_file = self.results_dir / "evaluation_results.csv"
            if csv_file.exists():
                self.df_results = pd.read_csv(csv_file)
                logger.info(f"加载评估结果: {len(self.df_results)} 条")
            
            # 加载对比结果
            comparison_file = self.results_dir / "search_type_comparison.csv"
            if comparison_file.exists():
                self.comparison = pd.read_csv(comparison_file, header=[0, 1], index_col=0)
                logger.info(f"加载对比结果: {len(self.comparison)} 种配置")
            
            # 加载测试报告
            report_file = self.results_dir / "test_report.json"
            if report_file.exists():
                with open(report_file, 'r', encoding='utf-8') as f:
                    self.report = json.load(f)
                logger.info("加载测试报告")
        
        except Exception as e:
            logger.error(f"加载结果文件失败: {e}")
    
    def generate_visualizations(self) -> List[str]:
        """
        生成可视化图表
        
        Returns:
            List[str]: 生成的图表文件路径列表
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import seaborn as sns
            sns.set_style("whitegrid")
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
            plt.rcParams['axes.unicode_minus'] = False
        except ImportError:
            logger.error("需要安装 matplotlib 和 seaborn: pip install matplotlib seaborn")
            return []
        
        generated_files = []
        
        if self.df_results is None or len(self.df_results) == 0:
            logger.warning("没有评估结果可供可视化")
            return generated_files
        
        try:
            # 1. 指标对比雷达图
            radar_file = self._generate_radar_chart(plt, sns)
            if radar_file:
                generated_files.append(radar_file)
            
            # 2. 指标箱线图
            box_file = self._generate_box_plot(plt, sns)
            if box_file:
                generated_files.append(box_file)
            
            # 3. 响应时间对比
            time_file = self._generate_response_time_chart(plt, sns)
            if time_file:
                generated_files.append(box_file)
            
            # 4. 指标热力图
            heatmap_file = self._generate_heatmap(plt, sns)
            if heatmap_file:
                generated_files.append(heatmap_file)
            
            logger.info(f"生成了 {len(generated_files)} 个可视化图表")
            
        except Exception as e:
            logger.error(f"生成可视化图表失败: {e}")
        
        return generated_files
    
    def _generate_radar_chart(self, plt, sns) -> str:
        """生成雷达图"""
        if 'search_type' not in self.df_results.columns:
            return None
        
        try:
            import numpy as np
            
            metrics = ['context_precision', 'context_recall', 'faithfulness', 
                      'answer_relevancy', 'answer_correctness', 'answer_similarity']
            available_metrics = [m for m in metrics if m in self.df_results.columns]
            
            if len(available_metrics) < 3:
                return None
            
            # 计算每种检索类型的平均值
            grouped = self.df_results.groupby('search_type')[available_metrics].mean()
            
            # 雷达图
            angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            for search_type in grouped.index:
                values = grouped.loc[search_type].tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=search_type)
                ax.fill(angles, values, alpha=0.15)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(available_metrics)
            ax.set_ylim(0, 1)
            ax.set_title('检索方式指标对比 (雷达图)', size=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
            
            file_path = self.results_dir / "radar_chart.png"
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"雷达图已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"生成雷达图失败: {e}")
            return None
    
    def _generate_box_plot(self, plt, sns) -> str:
        """生成箱线图"""
        if 'search_type' not in self.df_results.columns:
            return None
        
        try:
            metrics = ['context_precision', 'context_recall', 'faithfulness', 
                      'answer_relevancy', 'answer_correctness', 'answer_similarity']
            available_metrics = [m for m in metrics if m in self.df_results.columns]
            
            if len(available_metrics) < 2:
                return None
            
            # 准备数据
            df_melted = self.df_results.melt(
                id_vars=['search_type'],
                value_vars=available_metrics,
                var_name='metric',
                value_name='score'
            )
            
            # 绘制箱线图
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.boxplot(data=df_melted, x='metric', y='score', hue='search_type', ax=ax)
            
            ax.set_title('各指标得分分布 (箱线图)', size=16)
            ax.set_xlabel('指标', size=12)
            ax.set_ylabel('得分', size=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(title='检索方式', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            file_path = self.results_dir / "box_plot.png"
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"箱线图已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"生成箱线图失败: {e}")
            return None
    
    def _generate_response_time_chart(self, plt, sns) -> str:
        """生成响应时间对比图"""
        if 'response_time' not in self.df_results.columns or 'search_type' not in self.df_results.columns:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            grouped = self.df_results.groupby('search_type')['response_time'].agg(['mean', 'std'])
            grouped.plot(kind='bar', y='mean', yerr='std', ax=ax, legend=False)
            
            ax.set_title('平均响应时间对比', size=16)
            ax.set_xlabel('检索方式', size=12)
            ax.set_ylabel('响应时间 (秒)', size=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            file_path = self.results_dir / "response_time.png"
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"响应时间图已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"生成响应时间图失败: {e}")
            return None
    
    def _generate_heatmap(self, plt, sns) -> str:
        """生成热力图"""
        if self.comparison is None or len(self.comparison) == 0:
            return None
        
        try:
            # 提取 mean 值
            mean_data = self.comparison.xs('mean', level=1, axis=1)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(mean_data.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                       cbar_kws={'label': '平均得分'})
            
            ax.set_title('检索方式 × 指标热力图', size=16)
            ax.set_xlabel('检索方式', size=12)
            ax.set_ylabel('指标', size=12)
            
            file_path = self.results_dir / "heatmap.png"
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"热力图已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"生成热力图失败: {e}")
            return None
    
    def generate_html_report(self) -> str:
        """
        生成 HTML 报告
        
        Returns:
            str: HTML 文件路径
        """
        if self.df_results is None or self.report is None:
            logger.warning("缺少必要数据，无法生成 HTML 报告")
            return None
        
        try:
            html_content = self._build_html_content()
            
            file_path = self.results_dir / "report.html"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML 报告已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"生成 HTML 报告失败: {e}")
            return None
    
    def _build_html_content(self) -> str:
        """构建 HTML 内容"""
        summary = self.report.get('test_summary', {})
        stats = self.report.get('summary_statistics', {})
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 测试报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-left: 4px solid #4CAF50; padding-left: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }}
        .summary-card h3 {{ margin: 0; font-size: 14px; opacity: 0.9; }}
        .summary-card p {{ margin: 10px 0 0 0; font-size: 32px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric-good {{ color: #4CAF50; font-weight: bold; }}
        .metric-medium {{ color: #FF9800; font-weight: bold; }}
        .metric-poor {{ color: #f44336; font-weight: bold; }}
        .chart {{ margin: 20px 0; text-align: center; }}
        .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #888; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 RAG 系统测试报告</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>总测试数</h3>
                <p>{summary.get('total_tests', 0)}</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>成功测试</h3>
                <p>{summary.get('successful_tests', 0)}</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>测试问题数</h3>
                <p>{summary.get('questions', 0)}</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <h3>检索配置数</h3>
                <p>{summary.get('test_configs', 0)}</p>
            </div>
        </div>
        
        <h2>整体指标统计</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>平均值</th>
                <th>标准差</th>
                <th>最小值</th>
                <th>最大值</th>
                <th>中位数</th>
            </tr>
"""
        
        for metric, values in stats.items():
            if metric != 'overall_score' and isinstance(values, dict):
                mean_val = values.get('mean', 0)
                metric_class = 'metric-good' if mean_val >= 0.7 else ('metric-medium' if mean_val >= 0.5 else 'metric-poor')
                
                html += f"""
            <tr>
                <td>{metric}</td>
                <td class="{metric_class}">{mean_val:.3f}</td>
                <td>{values.get('std', 0):.3f}</td>
                <td>{values.get('min', 0):.3f}</td>
                <td>{values.get('max', 0):.3f}</td>
                <td>{values.get('median', 0):.3f}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>可视化图表</h2>
"""
        
        # 添加图表（检查是否存在）
        chart_files = {
            'radar_chart.png': '指标对比雷达图',
            'box_plot.png': '指标分布箱线图',
            'heatmap.png': '检索方式对比热力图',
            'response_time.png': '响应时间对比'}
        
        has_charts = False
        for img_file, title in chart_files.items():
            if (self.results_dir / img_file).exists():
                has_charts = True
                html += f"""
        <div class="chart">
            <h3 style="color: #555; margin-bottom: 10px;">{title}</h3>
            <img src="{img_file}" alt="{title}">
        </div>
"""
        
        if not has_charts:
            html += """
        <div style="padding: 20px; background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; color: #856404;">
            <p>⚠️ 暂无可视化图表。图表生成需要安装 matplotlib 和 seaborn:</p>
            <code style="background-color: #f8f9fa; padding: 5px 10px; border-radius: 3px; display: inline-block; margin-top: 5px;">
                pip install matplotlib seaborn
            </code>
        </div>
"""
        
        html += """
        <div class="footer">
            <p>RAG 测试系统 | 生成时间: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html


# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 假设有结果目录
    results_dir = "results/20240101_120000"
    
    generator = ReportGenerator(results_dir)
    
    # 生成可视化
    viz_files = generator.generate_visualizations()
    print(f"生成了 {len(viz_files)} 个可视化图表")
    
    # 生成 HTML 报告
    html_file = generator.generate_html_report()
    print(f"HTML 报告: {html_file}")






