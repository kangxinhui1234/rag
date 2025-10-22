import os
from datasets import Dataset
from ragas import evaluate

# 设置阿里云API密钥（从环境变量获取）
os.environ["DASHSCOPE_API_KEY"] = ""  # 替换为你的密钥

# 主要评估指标分类
from ragas.metrics import (
    # 检索质量指标
    context_precision,  # 上下文精确度
    context_recall,  # 上下文召回率
    # 生成质量指标
    faithfulness,  # 答案忠实度
    answer_relevancy,  # 答案相关性
    answer_correctness,  # 答案正确性
    answer_similarity  # 答案相似度

)

def explain_metrics():
    """解释每个指标的含义"""
    metrics_explanation = {
        # user_input：用户的问题
        # retrieved_contexts：检索出来的上下文片段（一个列表）
        # reference：参考答案（ground truth）
        'context_precision': '检索的上下文是否精确相关（避免无关信息）', # 是否相关、排序位置， k位置前相关文档书/k位置前文档总数 ，所有的文档取平均值
        # Context Precision = ∑ (Precision@k × rel_k) / Total relevant documents

        'context_recall': '是否检索到了所有相关信息（避免信息遗漏）', # 答案是否都能在上下文找到支撑
        'faithfulness': '生成的答案是否忠实于提供的上下文（避免编造）',
        'answer_relevancy': '答案是否直接回答了问题', # 答案相关性
        'answer_correctness': '答案与真实答案的匹配程度', # 答案真实性
        'answer_similarity': '答案与真实答案的语义相似度', # 答案相似度
        'harmfulness': '答案是否包含有害内容',
        'ragas_score': '综合评分（多个指标的加权平均）'
    }

    print("🎯 Ragas评估指标说明:")
    for metric, desc in metrics_explanation.items():
        print(f"📍 {metric}: {desc}")


explain_metrics()
def simple_ragas_evaluation(test_data, sample_size=5):
    """
    最简单的RAG评估MVP
    """
    try:
        # 1. 读取数据
        print("📊 读取数据...")
       # df = pd.read_csv(csv_file_path)

        # 2. 准备Ragas数据集
        print("🔄 准备评估数据集...")

        # ✅ 构建 HuggingFace Dataset
        dataset = Dataset.from_dict(test_data)
        print(dataset)

        # 3. 导入模型（会自动使用阿里云）
        from langchain_community.llms import Tongyi
        from langchain_community.embeddings import DashScopeEmbeddings

        llm = Tongyi(model="qwen-turbo")
        embeddings = DashScopeEmbeddings(model="text-embedding-v3")
        # 存储每条问题结果
        all_results = []

        # 4. 执行评估
        print("🚀 开始评估...")
        for idx, sample in enumerate(dataset):
            contexts_combined = " ".join(sample['contexts'])

            single_dataset = Dataset.from_dict({
                'question': [sample['question']],
                'contexts': [sample['contexts']],
                'answer': [sample['answer']],
                'ground_truth': [sample['ground_truth']]
            })

            result = evaluate(
                single_dataset,
                metrics=[context_precision, context_recall, faithfulness, answer_relevancy,
                         answer_correctness, answer_similarity],
                embeddings=embeddings,
                llm=llm
            )

            # answer_relevancy,  # 答案相关性
            # answer_correctness,  # 答案正确性
            # answer_similarity  # 答案相似度
            row = result.to_pandas().iloc[0].to_dict()
            row['question'] = sample['question']
            all_results.append(row)

        # 5. 输出结果
        print("\n✅ 评估完成！")
        print("=" * 50)
        print("评估结果:")
        print(result)

        # 保存结果
        result_df = result.to_pandas()
        result_df.to_csv("mvp_evaluation_results.csv", index=False)
        print("\n💾 结果已保存到: mvp_evaluation_results.csv")

        return result

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return None


if __name__ == "__main__":
    # 使用方法
    # 完整测试数据集，10条独立问题，每条上下文包含多个文本块
    # 每个key都可以是一个数组，但是多个问题必须顺序一致
    #例子：
    # data_samples = {
    #     'question': ['第一届超级碗是什么时候举行的？', '谁赢得了最多的超级碗冠军？'],
    #     'answer': ['第一届超级碗于1967年1月15日举行', '赢得最多超级碗冠军的是新英格兰爱国者队'],
    #     'contexts': [['第一届 AFL-NFL 世界冠军赛是一场美式橄榄球比赛，于1967年1月15日在洛杉矶纪念体育馆举行'],
    #                  ['绿湾包装工队...位于威斯康星州绿湾市。', '包装工队参加...全国橄榄球联合会比赛']],
    #     'ground_truth': ['第一届超级碗于1967年1月15日举行', '新英格兰爱国者队赢得了创纪录的六次超级碗冠军']
    # }

    #  contexts上下文必须组成一个字符串文本块，不能传一个数组进来
    test_data = {
        'question': ['机器学习的主要应用领域有哪些？'],
        'contexts': [[
            "机器学习通过数据训练模型在金融风控应用领域中用于风险预测。"
            "机器学习在医疗诊断应用领域中可以帮助识别疾病模式。"
            "机器学习推荐系统通过用户行为数据进行个性化推荐。"
            "机器学习图像识别是机器学习的重要应用之一。"
        ]],
        'answer': ['机器学习应用于金融风控、医疗诊断、推荐系统和图像识别。'],
        'ground_truth': ['机器学习通过数据训练模型并应用于金融风控、医疗、推荐系统和图像识别等领域。']
    }
    simple_ragas_evaluation(test_data, sample_size=3)  # 先用3条测试
