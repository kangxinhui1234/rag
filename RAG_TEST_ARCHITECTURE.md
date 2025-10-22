# RAG 知识库测试架构方案

## 📋 系统概述

### 现有系统
- **生产环境**: Spring AI + Elasticsearch (向量检索 + BM25)
- **测试环境**: Python + Ragas (测试集生成 + 指标评估)

### 测试目标
1. 评估检索质量（召回率、精确度）
2. 评估生成质量（忠实度、相关性）
3. 对比向量检索 vs BM25 vs 混合检索效果
4. 持续监控系统性能

---

## 🏗️ 测试架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        测试流程总览                               │
└─────────────────────────────────────────────────────────────────┘

Phase 1: 测试数据准备 (Python)
┌──────────────────────────────────────┐
│  1. 加载知识库文档                    │
│  2. 使用 Ragas 生成测试集             │
│     - 自动生成问题                    │
│     - 生成参考答案                    │
│     - 生成 ground_truth 上下文        │
│  3. 保存为标准格式 (JSON)             │
└──────────────────────────────────────┘
           ↓
Phase 2: RAG 系统调用 (Python ↔ Spring AI)
┌──────────────────────────────────────┐
│  4. 通过 HTTP API 调用 Spring AI      │
│     - 发送问题                        │
│     - 接收检索上下文                  │
│     - 接收生成答案                    │
│  5. 记录响应数据                      │
└──────────────────────────────────────┘
           ↓
Phase 3: 评估与分析 (Python)
┌──────────────────────────────────────┐
│  6. 使用 Ragas 计算指标               │
│     - Context Precision              │
│     - Context Recall                 │
│     - Faithfulness                   │
│     - Answer Relevancy               │
│  7. 生成评估报告                      │
│  8. 可视化分析结果                    │
└──────────────────────────────────────┘
```

---

## 📝 详细实施方案

### Step 1: 测试数据生成 (Python)

**输入**: 知识库原始文档 (PDF/TXT/etc.)
**输出**: `testset.json`

```json
{
  "testset": [
    {
      "question": "2022年平安银行的净利润是多少？",
      "ground_truth": "2022年平安银行实现净利润XXX亿元",
      "ground_truth_contexts": [
        "平安银行2022年财报显示...",
        "根据2022年年报..."
      ],
      "metadata": {
        "doc_source": "000001_2022_ZGPA.pdf",
        "difficulty": "simple",
        "type": "factual"
      }
    }
  ]
}
```

**关键代码**: `test_loader.py` (已完成)

---

### Step 2: Spring AI 接口定义

#### 2.1 需要在 Spring AI 中暴露的 API

```java
// Controller 示例
@RestController
@RequestMapping("/api/rag")
public class RagTestController {
    
    // 1. 向量检索接口
    @PostMapping("/search/vector")
    public SearchResponse vectorSearch(@RequestBody SearchRequest request) {
        // 使用 Elasticsearch 向量检索
        List<String> contexts = vectorSearchService.search(
            request.getQuestion(), 
            request.getTopK()
        );
        return new SearchResponse(contexts, "vector");
    }
    
    // 2. BM25 检索接口
    @PostMapping("/search/bm25")
    public SearchResponse bm25Search(@RequestBody SearchRequest request) {
        // 使用 Elasticsearch BM25
        List<String> contexts = bm25SearchService.search(
            request.getQuestion(), 
            request.getTopK()
        );
        return new SearchResponse(contexts, "bm25");
    }
    
    // 3. 混合检索接口
    @PostMapping("/search/hybrid")
    public SearchResponse hybridSearch(@RequestBody SearchRequest request) {
        // 向量 + BM25 混合
        List<String> contexts = hybridSearchService.search(
            request.getQuestion(), 
            request.getTopK(),
            request.getVectorWeight(),  // 默认 0.7
            request.getBm25Weight()      // 默认 0.3
        );
        return new SearchResponse(contexts, "hybrid");
    }
    
    // 4. 完整 RAG 接口 (检索 + 生成)
    @PostMapping("/qa")
    public RagResponse ragQuery(@RequestBody RagRequest request) {
        // 1. 检索
        List<String> contexts = retrievalService.search(
            request.getQuestion(),
            request.getSearchType()  // vector/bm25/hybrid
        );
        
        // 2. 生成答案
        String answer = generationService.generate(
            request.getQuestion(),
            contexts
        );
        
        return new RagResponse(
            request.getQuestion(),
            answer,
            contexts,
            System.currentTimeMillis() - startTime
        );
    }
}

// 请求/响应模型
class SearchRequest {
    private String question;
    private Integer topK = 5;
    private Double vectorWeight = 0.7;
    private Double bm25Weight = 0.3;
}

class RagRequest {
    private String question;
    private String searchType; // "vector", "bm25", "hybrid"
    private Integer topK = 5;
}

class RagResponse {
    private String question;
    private String answer;
    private List<String> contexts;
    private Long responseTime;
}
```

---

### Step 3: Python 测试框架设计

#### 3.1 目录结构

```
pythonProject6/
├── data/                          # 测试数据
│   ├── source_docs/              # 原始文档
│   └── testsets/                 # 生成的测试集
│       ├── testset_v1.json
│       └── testset_ground_truth.json
├── src/
│   ├── testset_generator.py      # 测试集生成器
│   ├── spring_ai_client.py       # Spring AI HTTP 客户端
│   ├── ragas_evaluator.py        # Ragas 评估器
│   └── test_runner.py            # 测试执行器
├── results/                       # 测试结果
│   ├── evaluation_results.csv
│   ├── comparison_report.html
│   └── metrics_visualization.png
├── config/
│   └── test_config.yaml          # 测试配置
└── run_full_test.py              # 主入口
```

---

## 🔄 完整测试流程

### 流程 1: 一次性完整测试

```python
# run_full_test.py
1. 加载测试集
2. For each question:
   a. 调用 Spring AI API (向量/BM25/混合)
   b. 获取 contexts 和 answer
   c. 记录结果
3. 使用 Ragas 批量评估
4. 生成对比报告
5. 输出结果和建议
```

### 流程 2: 持续监控测试

```python
# continuous_monitor.py
1. 定期执行测试 (每天/每周)
2. 记录历史指标
3. 检测性能退化
4. 自动告警
```

---

## 📊 评估指标体系

### 检索质量指标
- **Context Precision**: 检索结果的精确度
- **Context Recall**: 检索结果的召回率
- **MRR** (Mean Reciprocal Rank): 相关文档排序质量
- **Hit Rate@K**: 前K个结果命中率

### 生成质量指标
- **Faithfulness**: 答案对检索内容的忠实度
- **Answer Relevancy**: 答案相关性
- **Answer Correctness**: 答案正确性
- **Answer Similarity**: 答案与标准答案的相似度

### 系统性能指标
- **Response Time**: 响应时间
- **Throughput**: 吞吐量

---

## 🎯 对比测试方案

### A/B 测试设计

```
测试组：
1. 向量检索 Only
2. BM25 Only
3. 混合检索 (0.7 vector + 0.3 BM25)
4. 混合检索 (0.5 vector + 0.5 BM25)
5. 混合检索 (0.3 vector + 0.7 BM25)

评估维度：
- 各项 Ragas 指标
- 响应时间
- 不同问题类型的表现
```

---

## 📈 报告输出

### 1. CSV 详细结果
```csv
question,search_type,answer,contexts,context_precision,context_recall,faithfulness,answer_relevancy,response_time
```

### 2. HTML 可视化报告
- 指标对比雷达图
- 不同检索方式性能对比
- 问题难度分析
- 错误案例分析

### 3. 优化建议
基于测试结果自动生成优化建议

---

## 🛠️ 实施步骤

### Week 1: 基础设施搭建
- [ ] Spring AI 暴露测试 API
- [ ] Python 客户端开发
- [ ] 测试数据生成

### Week 2: 测试框架开发
- [ ] Ragas 评估集成
- [ ] 测试执行器开发
- [ ] 结果记录和存储

### Week 3: 分析和优化
- [ ] 报告生成
- [ ] 可视化开发
- [ ] 持续监控系统

---

## 🔍 最佳实践

1. **测试集多样性**: 包含不同难度、不同类型的问题
2. **版本控制**: 测试集和结果都要版本管理
3. **隔离测试环境**: 避免影响生产数据
4. **定期更新**: 随着知识库更新，测试集也要更新
5. **人工审核**: 定期人工审核测试结果，确保质量

---

## 📚 参考资源

- Ragas Documentation: https://docs.ragas.io/
- Elasticsearch Vector Search: https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html
- Spring AI: https://spring.io/projects/spring-ai






