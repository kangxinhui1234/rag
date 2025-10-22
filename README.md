# RAG 知识库测试系统

一个完整的 RAG (Retrieval-Augmented Generation) 系统测试框架，用于评估和对比不同检索策略（向量检索、BM25、混合检索）的效果。

## 🎯 功能特性

- ✅ **自动化测试集生成**: 使用 Ragas 从 PDF 文档自动生成测试问题和答案
- ✅ **多种检索策略**: 支持向量检索、BM25、混合检索（可配置权重）
- ✅ **全面的评估指标**: Context Precision、Context Recall、Faithfulness、Answer Relevancy 等
- ✅ **跨环境测试**: Python 测试框架 + Spring AI 生产环境
- ✅ **可视化报告**: 自动生成雷达图、箱线图、热力图等
- ✅ **灵活配置**: YAML 配置文件，支持自定义测试场景

## 📁 项目结构

```
rag/
├── config/
│   └── test_config.yaml          # 测试配置文件
├── data/
│   ├── source_docs/              # 原始文档 (PDF)
│   └── testsets/                 # 生成的测试集
│             └── ragas_source_testset.json # ragas生成的测试数据集
│             └── wait_test_testset.json # 清洗过后格式化好的测试集，可以直接输入评估系统
├── src/
│   ├── spring_ai_client.py       # Spring AI HTTP 客户端
│   ├── ragas_evaluator.py        # Ragas 评估器
│   ├── test_runner.py            # 测试执行器
│   └── report_generator.py       # 报告生成器
├── results/                       # 测试结果输出
├── test_loader.py                # 测试集生成脚本
├── run_full_test.py              # 主测试入口
├── RAG_TEST_ARCHITECTURE.md      # 架构设计文档
└── README.md                     # 本文件
```

## 🚀 快速开始

### 1. 环境准备

#### Python 环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 环境变量

```bash
# 设置 DashScope API Key (阿里云通义千问)
export DASHSCOPE_API_KEY="your-api-key-here"

# Windows PowerShell:
$env:DASHSCOPE_API_KEY="your-api-key-here"
```

### 2. 生成测试集

```bash
# 从 PDF 文档生成测试集
python test_loader.py
```

这将生成 `ragas_testset.json`，包含自动生成的问题、答案和上下文。

**测试集格式**:
```json
[
  {
    "question": "2022年平安银行的净利润是多少？",
    "ground_truth": "2022年平安银行实现净利润XXX亿元",
    "ground_truth_contexts": [
      "平安银行2022年财报显示...",
      "根据2022年年报..."
    ],
    "metadata": {
      "doc_source": "000001_2022_ZGPA.pdf",
      "difficulty": "simple"
    }
  }
]
```

### 3. 配置 Spring AI 服务

在你的 Spring AI 项目中添加测试接口（参考 `docs/spring_ai_setup.md`）:

```java
@RestController
@RequestMapping("/api/rag")
public class RagTestController {
    
    @PostMapping("/search/vector")
    public SearchResponse vectorSearch(@RequestBody SearchRequest request) {
        // 向量检索实现
    }
    
    @PostMapping("/search/bm25")
    public SearchResponse bm25Search(@RequestBody SearchRequest request) {
        // BM25 检索实现
    }
    
    @PostMapping("/qa")
    public RagResponse ragQuery(@RequestBody RagRequest request) {
        // 完整 RAG 实现
    }
}
```

**启动 Spring AI 服务**:
```bash
# 确保服务运行在 http://localhost:8080
mvn spring-boot:run
```

### 4. 配置测试参数

编辑 `config/test_config.yaml`:

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
    - name: "bm25_only"
      type: "bm25"
      enabled: true
    - name: "hybrid_0.7_0.3"
      type: "hybrid"
      vector_weight: 0.7
      bm25_weight: 0.3
      enabled: true
```

### 5. 运行测试

```bash
# 运行完整测试
python run_full_test.py

# 使用自定义配置
python run_full_test.py --config config/my_test_config.yaml

# 指定测试集
python run_full_test.py --testset data/testsets/custom_testset.json

# 跳过报告生成（仅运行测试）
python run_full_test.py --skip-report
```

### 6. 查看结果

测试完成后，结果保存在 `results/YYYYMMDD_HHMMSS/` 目录：

```
results/20240122_153045/
├── raw_test_results.json          # 原始测试结果
├── evaluation_results.csv         # 评估结果 (CSV)
├── evaluation_results.json        # 评估结果 (JSON)
├── search_type_comparison.csv     # 检索方式对比
├── test_report.json               # 测试报告
├── report.html                    # HTML 可视化报告 ⭐
├── radar_chart.png                # 雷达图
├── box_plot.png                   # 箱线图
├── heatmap.png                    # 热力图
└── response_time.png              # 响应时间对比
```

**打开 HTML 报告**:
```bash
# 在浏览器中打开
start results/20240122_153045/report.html  # Windows
open results/20240122_153045/report.html   # Mac
```

## 📊 评估指标说明

### 检索质量指标

- **Context Precision**: 检索结果的精确度（是否都相关）
- **Context Recall**: 检索结果的召回率（是否检索全了）

### 生成质量指标

- **Faithfulness**: 答案是否忠实于检索的上下文（避免幻觉）
- **Answer Relevancy**: 答案是否直接回答问题
- **Answer Correctness**: 答案与标准答案的匹配程度
- **Answer Similarity**: 答案与标准答案的语义相似度

### 系统性能指标

- **Response Time**: 平均响应时间

## 🔧 高级用法

### 自定义测试集

手动创建测试集 `data/testsets/my_testset.json`:

```json
{
  "testset": [
    {
      "question": "你的问题?",
      "ground_truth": "标准答案",
      "ground_truth_contexts": [
        "相关上下文1",
        "相关上下文2"
      ],
      "metadata": {
        "category": "财务",
        "difficulty": "hard"
      }
    }
  ]
}
```

### 对比不同配置

```yaml
test:
  search_types:
    # 测试不同的混合权重
    - name: "hybrid_0.9_0.1"
      type: "hybrid"
      vector_weight: 0.9
      bm25_weight: 0.1
      enabled: true
    
    - name: "hybrid_0.5_0.5"
      type: "hybrid"
      vector_weight: 0.5
      bm25_weight: 0.5
      enabled: true
    
    - name: "hybrid_0.1_0.9"
      type: "hybrid"
      vector_weight: 0.1
      bm25_weight: 0.9
      enabled: true
```

### 编程式使用

```python
from src.test_runner import TestRunner
from src.ragas_evaluator import RagasEvaluator

# 创建测试执行器
runner = TestRunner("http://localhost:8080", api_key="your-key")

# 加载测试集
testset = runner.load_testset("data/testsets/testset.json")

# 运行单个测试
result = runner.run_single_test(
    question="2022年净利润是多少？",
    ground_truth="XXX亿元",
    ground_truth_contexts=["..."],
    search_type="hybrid"
)

# 评估结果
evaluator = RagasEvaluator()
metrics = evaluator.evaluate_single(
    question=result['question'],
    answer=result['answer'],
    contexts=result['contexts'],
    ground_truth=result['ground_truth']
)

print(f"Faithfulness: {metrics['faithfulness']:.3f}")
```

## 🐛 故障排除

### 问题 1: `ValueError: Node has no summary_embedding`

**原因**: Ragas 版本问题或文档数量太少

**解决方案**:
```bash
# 升级 Ragas
pip install --upgrade ragas

# 或增加文档数量
docs_subset = docs[10:30]  # 在 test_loader.py 中
```

### 问题 2: Spring AI 连接失败

**检查**:
```bash
# 测试服务是否可用
curl http://localhost:8080/actuator/health

# 检查防火墙和端口
netstat -an | findstr 8080
```

### 问题 3: API Key 无效

**检查**:
```bash
# 验证环境变量
echo $DASHSCOPE_API_KEY  # Linux/Mac
echo %DASHSCOPE_API_KEY%  # Windows CMD
$env:DASHSCOPE_API_KEY    # Windows PowerShell
```

### 问题 4: 内存不足

**解决方案**:
- 减少测试集大小
- 减少 `top_k` 值
- 分批测试

## 📖 文档

- [架构设计](RAG_TEST_ARCHITECTURE.md) - 详细的架构和设计说明
- [Spring AI 接口配置](docs/spring_ai_setup.md) - Spring AI 端配置指南
- [测试最佳实践](docs/best_practices.md) - 测试建议和技巧

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙋 常见问题

**Q: 可以使用 OpenAI 而不是阿里云吗？**

A: 可以。修改 `src/ragas_evaluator.py`:
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

self.llm = ChatOpenAI(model="gpt-4")
self.embeddings = OpenAIEmbeddings()
```

**Q: 如何添加自定义指标？**

A: 在 `src/ragas_evaluator.py` 的 `metrics` 列表中添加你的指标。

**Q: 支持其他语言的文档吗？**

A: 支持。Ragas 和 LangChain 支持多语言文档。

**Q: 可以测试实时数据库吗？**

A: 可以。只需修改 Spring AI 端的数据源即可。

## 📧 联系方式

如有问题，请提交 Issue 或发送邮件。

---

**Happy Testing! 🚀**






