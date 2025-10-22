# RAG 测试系统项目总结

## 📂 完整文件列表

### 核心代码
```
src/
├── __init__.py                    # 模块初始化
├── spring_ai_client.py            # Spring AI HTTP 客户端（与 Java 服务交互）
├── ragas_evaluator.py             # Ragas 评估器（评估检索和生成质量）
├── test_runner.py                 # 测试执行器（协调测试流程）
└── report_generator.py            # 报告生成器（生成可视化报告）
```

### 主要脚本
```
test_loader.py                     # 从 PDF 生成测试集
run_full_test.py                   # 主测试入口（运行完整测试）
demo_usage.py                      # 使用演示脚本
```

### 配置和数据
```
config/
└── test_config.yaml               # 测试配置文件

data/
├── source_docs/                   # 原始文档目录
├── testsets/                      # 测试集目录
│   └── sample_testset.json        # 示例测试集
└── 000001_2022_ZGPA_*.pdf         # 示例 PDF 文档
```

### 文档
```
README.md                          # 完整文档
QUICKSTART.md                      # 5分钟快速开始
RAG_TEST_ARCHITECTURE.md           # 详细架构设计
PROJECT_SUMMARY.md                 # 本文件

docs/
└── spring_ai_setup.md             # Spring AI 配置指南
```

### 其他
```
requirements.txt                   # Python 依赖
.gitignore                         # Git 忽略文件
```

---

## 🎯 核心功能模块

### 1. 测试集生成模块 (`test_loader.py`)

**功能**: 从 PDF 文档自动生成测试问题和答案

**主要步骤**:
1. 加载 PDF 文档
2. 文本切分 (chunking)
3. 使用 Ragas 生成测试集
4. 保存为 JSON 格式

**输出**: `ragas_testset.json`

**关键代码**:
```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator.from_langchain(
    llm=llm,
    embedding_model=embeddings
)

testset = generator.generate_with_langchain_docs(
    documents=docs,
    testset_size=5
)
```

### 2. Spring AI 客户端 (`src/spring_ai_client.py`)

**功能**: 与 Spring AI 服务进行 HTTP 通信

**支持的接口**:
- `vector_search()` - 向量检索
- `bm25_search()` - BM25 检索
- `hybrid_search()` - 混合检索
- `rag_query()` - 完整 RAG（检索 + 生成）

**使用示例**:
```python
client = SpringAIClient("http://localhost:8080")

# 完整 RAG 查询
result = client.rag_query(
    question="2022年净利润是多少？",
    search_type="hybrid"
)
print(result.answer)
```

### 3. Ragas 评估器 (`src/ragas_evaluator.py`)

**功能**: 使用 Ragas 框架评估 RAG 系统质量

**评估指标**:
- Context Precision - 检索精确度
- Context Recall - 检索召回率
- Faithfulness - 答案忠实度
- Answer Relevancy - 答案相关性
- Answer Correctness - 答案正确性
- Answer Similarity - 答案相似度

**使用示例**:
```python
evaluator = RagasEvaluator()

metrics = evaluator.evaluate_single(
    question="问题",
    answer="答案",
    contexts=["上下文1", "上下文2"],
    ground_truth="标准答案"
)

print(f"Faithfulness: {metrics['faithfulness']:.3f}")
```

### 4. 测试执行器 (`src/test_runner.py`)

**功能**: 协调测试流程，整合所有组件

**主要功能**:
- 加载测试集
- 调用 Spring AI 服务
- 使用 Ragas 评估
- 生成测试报告

**使用示例**:
```python
runner = TestRunner("http://localhost:8080", api_key="xxx")

report = runner.run_full_test(
    testset_path="testset.json",
    search_configs=[
        {'name': 'vector', 'type': 'vector'},
        {'name': 'hybrid', 'type': 'hybrid'}
    ]
)
```

### 5. 报告生成器 (`src/report_generator.py`)

**功能**: 生成可视化报告和图表

**生成内容**:
- 雷达图 (各指标对比)
- 箱线图 (指标分布)
- 热力图 (配置 × 指标)
- 响应时间对比图
- HTML 报告

**使用示例**:
```python
generator = ReportGenerator("results/20240122_153045")

# 生成可视化
viz_files = generator.generate_visualizations()

# 生成 HTML 报告
html_file = generator.generate_html_report()
```

---

## 🔄 完整测试流程

```
┌─────────────────────┐
│ 1. 准备阶段          │
│  - 安装依赖          │
│  - 设置 API Key      │
│  - 准备 PDF 文档     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 2. 生成测试集        │
│  python             │
│  test_loader.py     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3. 配置测试参数      │
│  编辑                │
│  test_config.yaml   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 4. 启动 Spring AI   │
│  (Java 服务)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 5. 运行测试          │
│  python             │
│  run_full_test.py   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 6. 查看结果          │
│  - CSV 数据          │
│  - JSON 数据         │
│  - HTML 报告         │
│  - 可视化图表        │
└─────────────────────┘
```

---

## 🌐 系统架构

```
┌──────────────────────────────────────────────────────┐
│                 Python 测试环境                        │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ 测试集   │  │ 测试执行 │  │ 评估报告 │            │
│  │ 生成器   │→ │   器     │→ │ 生成器   │            │
│  └──────────┘  └────┬─────┘  └──────────┘            │
│                      │                                 │
└──────────────────────┼─────────────────────────────────┘
                       │ HTTP API
                       ▼
┌──────────────────────────────────────────────────────┐
│              Spring AI 生产环境                        │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ 向量检索 │  │ BM25检索 │  │ 混合检索 │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│       └─────────────┼─────────────┘                   │
│                     │                                  │
│                     ▼                                  │
│              ┌──────────┐                              │
│              │ LLM 生成 │                              │
│              └──────────┘                              │
│                     │                                  │
└─────────────────────┼──────────────────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │ Elasticsearch │
              │  (向量 + BM25)│
              └──────────────┘
```

---

## 📊 测试配置示例

### 基础配置
```yaml
spring_ai:
  base_url: "http://localhost:8080"

test:
  testset_path: "ragas_source_testset.json"
  top_k: 5
  
  search_types:
    - name: "vector_only"
      type: "vector"
      enabled: true
```

### 对比测试配置
```yaml
test:
  search_types:
    # 纯向量检索
    - name: "vector_only"
      type: "vector"
      enabled: true
    
    # 纯 BM25
    - name: "bm25_only"
      type: "bm25"
      enabled: true
    
    # 混合检索 - 不同权重
    - name: "hybrid_0.9_0.1"
      type: "hybrid"
      vector_weight: 0.9
      bm25_weight: 0.1
      enabled: true
    
    - name: "hybrid_0.7_0.3"
      type: "hybrid"
      vector_weight: 0.7
      bm25_weight: 0.3
      enabled: true
    
    - name: "hybrid_0.5_0.5"
      type: "hybrid"
      vector_weight: 0.5
      bm25_weight: 0.5
      enabled: true
```

---

## 🚀 快速命令参考

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置环境变量
export DASHSCOPE_API_KEY="your-key"

# 3. 生成测试集
python test_loader.py

# 4. 运行测试
python run_full_test.py

# 5. 查看演示
python demo_usage.py

# 6. 自定义测试
python run_full_test.py --config custom_config.yaml --testset my_test.json

# 7. 跳过报告生成
python run_full_test.py --skip-report
```

---

## 📈 输出结果说明

### 测试结果目录结构
```
results/20240122_153045/
├── raw_test_results.json          # 原始测试数据
├── evaluation_results.csv         # 评估结果 (表格)
├── evaluation_results.json        # 评估结果 (JSON)
├── search_type_comparison.csv     # 检索方式对比统计
├── test_report.json               # 完整测试报告
├── report.html                    # HTML 可视化报告 ⭐
├── radar_chart.png                # 雷达图
├── box_plot.png                   # 箱线图
├── heatmap.png                    # 热力图
└── response_time.png              # 响应时间对比
```

### 关键结果文件

**evaluation_results.csv**:
```csv
question,search_type,context_precision,faithfulness,answer_relevancy,response_time
问题1,vector,0.724,0.812,0.689,0.532
问题1,hybrid,0.789,0.851,0.742,0.615
...
```

**search_type_comparison.csv**:
```csv
search_type,context_precision_mean,faithfulness_mean,answer_relevancy_mean
vector,0.724,0.812,0.689
bm25,0.653,0.743,0.612
hybrid,0.789,0.851,0.742
```

---

## 💡 使用场景

### 场景 1: 评估检索策略
**目标**: 对比向量检索、BM25 和混合检索的效果

**步骤**:
1. 配置 3 种检索类型
2. 运行测试
3. 查看 `search_type_comparison.csv`
4. 选择最佳策略

### 场景 2: 优化混合权重
**目标**: 找到最佳的向量和 BM25 权重组合

**步骤**:
1. 配置多个不同权重的混合检索
2. 运行 A/B 测试
3. 对比各项指标
4. 选择最优权重

### 场景 3: 持续监控
**目标**: 定期评估系统性能，及时发现问题

**步骤**:
1. 创建定时任务 (cron)
2. 定期运行测试
3. 对比历史结果
4. 性能退化时告警

### 场景 4: 文档更新影响评估
**目标**: 知识库更新后评估影响

**步骤**:
1. 更新前运行测试（baseline）
2. 更新知识库
3. 更新后再次测试
4. 对比前后差异

---

## 🔧 扩展和定制

### 添加自定义指标
在 `src/ragas_evaluator.py` 中:
```python
from ragas.metrics import your_custom_metric

self.metrics = [
    context_precision,
    faithfulness,
    your_custom_metric  # 添加自定义指标
]
```

### 添加新的检索类型
在 Spring AI 端添加新接口，然后在配置中启用:
```yaml
search_types:
  - name: "custom_search"
    type: "custom"
    enabled: true
```

### 自定义报告
修改 `src/report_generator.py` 中的模板和图表生成逻辑。

---

## 📚 技术栈

**Python 端**:
- Ragas - RAG 评估框架
- LangChain - LLM 应用框架
- Pandas - 数据处理
- Matplotlib/Seaborn - 可视化
- Requests - HTTP 客户端

**Java 端** (Spring AI):
- Spring Boot - 应用框架
- Spring AI - AI 应用框架
- Elasticsearch - 搜索引擎
- 向量数据库 - 存储 embeddings

---

## ⚠️ 注意事项

1. **API 配额**: 注意 DashScope API 的调用配额
2. **测试时长**: 大规模测试可能需要较长时间
3. **数据安全**: 测试数据可能包含敏感信息，注意保护
4. **版本兼容**: 确保 Ragas、LangChain 版本兼容
5. **服务可用性**: 测试前确保 Spring AI 服务正常运行

---

## 🎓 学习资源

- [Ragas 官方文档](https://docs.ragas.io/)
- [LangChain 文档](https://python.langchain.com/)
- [Spring AI 文档](https://spring.io/projects/spring-ai)
- [Elasticsearch 向量搜索](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html)

---

**项目完成时间**: 2024-01-22  
**最后更新**: 2024-01-22

欢迎贡献和反馈！🌟






