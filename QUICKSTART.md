# 🚀 快速开始指南

5 分钟快速搭建 RAG 测试环境！

## 📦 Step 1: 安装依赖 (2 分钟)

```bash
# 克隆或进入项目目录
cd ragas-auto-test

# 安装 Python 依赖
pip install -r requirements.txt

# 设置 API Key
export DASHSCOPE_API_KEY="your-dashscope-api-key-here"
```

## 📝 Step 2: 生成测试集 (2 分钟)

```bash
# 编辑 test_loader.py，确认 PDF 路径正确
# 默认: data/000001_2022_ZGPA_2022_YEAR_2023-03-08.pdf

# 运行测试集生成
python test_loader.py
```

**输出**: `ragas_testset.json` (包含自动生成的测试数据)

**示例输出**:
```
Generated 10 chunks from PDF.
Merged 5 chunks from subset.
Applying HeadlinesExtractor: 100%|████| 4/4
Applying SummaryExtractor: 100%|████| 5/5
✅ 测试集已保存到 ragas_testset.json
生成了 3 条测试数据
```

## ⚙️ Step 3: 配置测试参数 (1 分钟)

编辑 `config/test_config.yaml`:

```yaml
spring_ai:
  base_url: "http://localhost:8080"  # ← 你的 Spring AI 服务地址

test:
  testset_path: "ragas_source_testset.json"  # ← 刚生成的测试集
  top_k: 5
```

## 🏃 Step 4: 运行测试 (根据测试集大小)

```bash
python run_full_test.py
```

**测试流程**:
```
1. 加载测试集 ✓
2. 调用 Spring AI (向量/BM25/混合检索) ...
3. 使用 Ragas 评估指标 ...
4. 生成报告和可视化 ✓
```

## 📊 Step 5: 查看结果

```bash
# 结果保存在 results/YYYYMMDD_HHMMSS/
ls results/

# 打开 HTML 报告
start results/20240122_153045/report.html
```

---

## 🎯 完整示例

### 场景: 对比 3 种检索方式

**配置** (`config/test_config.yaml`):
```yaml
test:
  search_types:
    - name: "vector_only"
      type: "vector"
      enabled: true
    - name: "bm25_only"
      type: "bm25"
      enabled: true
    - name: "hybrid_best"
      type: "hybrid"
      vector_weight: 0.7
      bm25_weight: 0.3
      enabled: true
```

**运行**:
```bash
python run_full_test.py
```

**结果示例**:
```
测试完成！
==================================================
总测试数: 9
✅ 成功: 9
❌ 失败: 0

整体得分: 0.756

结果保存在: results/20240122_153045
==================================================
```

**对比结果** (`results/.../search_type_comparison.csv`):
```
search_type      | context_precision | faithfulness | answer_relevancy
-----------------|-------------------|--------------|------------------
vector_only      | 0.724            | 0.812        | 0.689
bm25_only        | 0.653            | 0.743        | 0.612
hybrid_best      | 0.789            | 0.851        | 0.742  ← 最佳！
```

---

## 💡 使用技巧

### 技巧 1: 快速测试单个配置

```bash
# 临时禁用某些配置
# 在 config/test_config.yaml 中设置 enabled: false
```

### 技巧 2: 自定义测试集

创建 `my_test.json`:
```json
{
  "testset": [
    {
      "question": "你的问题?",
      "ground_truth": "答案",
      "ground_truth_contexts": ["上下文1", "上下文2"]
    }
  ]
}
```

运行:
```bash
python run_full_test.py --testset my_test.json
```

### 技巧 3: 只看结果不生成报告

```bash
python run_full_test.py --skip-report
```

### 技巧 4: 调试模式

编辑 `config/test_config.yaml`:
```yaml
logging:
  level: "DEBUG"  # 查看详细日志
```

---

## 🔧 常见问题

**Q: Spring AI 服务没启动怎么办？**

A: 测试会提示服务不可用，可以选择跳过或先启动服务。

**Q: 生成测试集失败？**

A: 检查:
1. PDF 文件是否存在
2. DASHSCOPE_API_KEY 是否设置
3. 网络连接是否正常

**Q: 想测试更多问题怎么办？**

A: 修改 `test_loader.py`:
```python
# 增加文档范围
docs_subset = docs[10:30]  # 原来是 [10:20]

# 增加测试集大小
testset_size=10  # 原来是 3
```

---

## 📈 下一步

1. **优化检索权重**: 尝试不同的 vector_weight 和 bm25_weight
2. **扩充测试集**: 添加更多领域的测试问题
3. **定期测试**: 设置 cron 任务定期运行测试
4. **A/B 测试**: 对比不同的 embedding 模型或 chunk 策略

---

## 🎓 进阶阅读

- [完整架构文档](RAG_TEST_ARCHITECTURE.md)
- [Spring AI 配置指南](docs/spring_ai_setup.md)
- [README](README.md)

---

**开始你的 RAG 测试之旅吧！** 🚀






