import json
"""
Ragas测试集生成的文件结构不符合评估接口的入参要求，需要做格式转换
"""
def convert_ragas_format(source_file, target_file):
    """
    将Ragas测试集格式转换为评估所需的输入格式
    """
    # 读取源文件
    with open(source_file, 'r', encoding='utf-8') as f:
        ragas_data = json.load(f)

    # 构建目标格式
    eval_format = {
        "testset": []
    }

    for item in ragas_data:
        eval_sample = item["eval_sample"]

        # 构建测试集项
        test_item = {
            "question": eval_sample["user_input"],
            "ground_truth": eval_sample["reference"],
            "ground_truth_contexts": eval_sample["reference_contexts"] if eval_sample["reference_contexts"] else [],
            "metadata": {
                "synthesizer_name": item["synthesizer_name"],
                "difficulty": _determine_difficulty(item["synthesizer_name"]),
                "type": _determine_question_type(item["synthesizer_name"]),
                "category": "金融/银行业务"
            }
        }

        eval_format["testset"].append(test_item)

    # 写入目标文件
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(eval_format, f, ensure_ascii=False, indent=2)

    print(f"转换完成！结果已保存到 {target_file}")
    return eval_format


def _determine_difficulty(synthesizer_name):
    """根据synthesizer_name确定问题难度"""
    if "single_hop" in synthesizer_name:
        return "simple"
    elif "multi_hop" in synthesizer_name:
        return "complex"
    else:
        return "medium"


def _determine_question_type(synthesizer_name):
    """根据synthesizer_name确定问题类型"""
    if "specific" in synthesizer_name:
        return "factual"
    elif "abstract" in synthesizer_name:
        return "conceptual"
    else:
        return "factual"


# 使用示例
if __name__ == "__main__":
    source_file = "../data/testsets/ragas_source_testset.json"  # 输入文件
    target_file = "../data/testsets/wait_test_testset2.json"  # 输出文件

    # 执行转换
    converted_data = convert_ragas_format(source_file, target_file)

    # 打印转换后的数据预览
    print("\n转换后的数据预览：")
    print(json.dumps(converted_data, ensure_ascii=False, indent=2))