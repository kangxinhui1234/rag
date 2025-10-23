"""
RAG 测试主入口
使用配置文件运行完整的测试流程
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from test_runner import TestRunner
from report_generator import ReportGenerator


def setup_logging(config: dict):
    """设置日志"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('log_file', 'logs/rag_test.log')
    
    # 创建日志目录
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='RAG 测试系统')
    parser.add_argument(
        '--config',
        type=str,
        default='../config/test_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--testset',
        type=str,
        help='测试集路径（覆盖配置文件）'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='输出目录（覆盖配置文件）'
    )
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='跳过报告生成'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    logger = setup_logging(config)
    
    logger.info("="*60)
    logger.info("RAG 测试系统启动")
    logger.info("="*60)
    
    # 获取配置
    spring_ai_config = config['spring_ai']
    test_config = config['test']
    ragas_config = config['ragas']
    output_config = config['output']
    
    # 命令行参数覆盖
    testset_path = args.testset or test_config['testset_path']
    output_dir = args.output or output_config['results_dir']
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"配置文件: {args.config}")
    logger.info(f"测试集: {testset_path}")
    logger.info(f"输出目录: {output_dir}")
    
    # 检查测试集是否存在
    if not Path(testset_path).exists():
        logger.error(f"测试集不存在: {testset_path}")
        logger.info("请先运行 test_loader.py 生成测试集")
        return 1

    # 获取 DashScope API 密钥（用于 RAGAS 评估）
    # 优先级：ragas.api_key > 环境变量 DASHSCOPE_API_KEY
    api_key = spring_ai_config.get('api_key')
    if not api_key:
        api_key = os.environ.get('DASHSCOPE_API_KEY')

    if not api_key:
        logger.error("未设置 DASHSCOPE_API_KEY")
        logger.info("请在配置文件 ragas.api_key 中设置，或设置环境变量 DASHSCOPE_API_KEY")
        return 1
    os.environ['DASHSCOPE_API_KEY'] = api_key
    # 构建检索配置
    search_configs = []
    for config_item in test_config['search_types']:
        if config_item.get('enabled', True):
            search_configs.append({
                'name': config_item['name'],
                'type': config_item['type'],
                'top_k': test_config.get('top_k', 5),
                **{k: v for k, v in config_item.items() 
                   if k not in ['name', 'type', 'enabled']}
            })
    
    logger.info(f"启用的检索配置: {[c['name'] for c in search_configs]}")
    
    # 创建测试执行器
    runner = TestRunner(
        spring_ai_url=spring_ai_config['base_url'],
        api_key=api_key,
        timeout=spring_ai_config.get('timeout', 30)
    )
    
    # 检查 Spring AI 服务是否可用
    if not runner.spring_client.health_check():
        logger.warning("⚠️  Spring AI 服务不可用，请确保服务已启动")
        logger.info(f"服务地址: {spring_ai_config['base_url']}")
        response = input("是否继续测试？(y/n): ")
        if response.lower() != 'y':
            return 1
    else:
        logger.info("✅ Spring AI 服务可用")
    
    try:
        # 运行完整测试
        logger.info("\n开始运行测试...")
        report = runner.run_full_test(
            testset_path=testset_path,
            search_configs=search_configs,
            output_dir=str(output_dir),
            save_raw_results=output_config.get('save_raw_responses', True)
        )
        
        # 打印测试摘要
        print("\n" + "="*60)
        print("测试完成！")
        print("="*60)
        print(f"总测试数: {report['test_summary']['total_tests']}")
        print(f"✅ 成功: {report['test_summary']['successful_tests']}")
        print(f"❌ 失败: {report['test_summary']['failed_tests']}")
        
        if 'overall_score' in report['summary_statistics']:
            print(f"\n整体得分: {report['summary_statistics']['overall_score']['mean']:.3f}")
        
        print(f"\n结果保存在: {output_dir}")
        print("="*60)
        
        # 生成可视化报告
        if not args.skip_report and output_config.get('generate_visualizations', True):
            logger.info("\n生成可视化报告...")
            
            try:
                report_gen = ReportGenerator(str(output_dir))
                # 生成可视化图表
                viz_files = report_gen.generate_visualizations()
                logger.info(f"可视化图表已生成: {len(viz_files)} 个")


                # 生成 HTML 报告
                if 'html' in output_config.get('save_format', []):
                    html_file = report_gen.generate_html_report()
                    logger.info(f"HTML 报告已生成: {html_file}")
                

                
            except ImportError as e:
                logger.warning(f"无法生成可视化报告（缺少依赖）: {e}")
            except Exception as e:
                logger.error(f"生成报告时出错: {e}")
        
        logger.info("\n测试流程完成！")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
        return 1
    except Exception as e:
        logger.error(f"\n测试失败: {e}", exc_info=True)
        return 1
    finally:
        runner.close()


if __name__ == "__main__":
    sys.exit(main())


