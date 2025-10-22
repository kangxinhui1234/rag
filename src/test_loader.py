# 正确导入方式
"""
数据测试集生成模块，直接运行即可生成测试问题集合

"""
import json
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from ragas.testset import TestsetGenerator

from src.file_transform import convert_ragas_format

os.environ["DASHSCOPE_API_KEY"] = "sk-0ef222e3c3d14e8d895aec2f2a16b4aa"  # 替换为你的密钥
loader = PyPDFLoader("../data/000001_2022_ZGPA_2022_YEAR_2023-03-08.pdf")
raw_docs = loader.load()  # 每页作为一个文档

# 2️⃣ 文本切分（chunking）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每块最大 500 字
    chunk_overlap=50 # 块之间重叠 50 字
)
docs = text_splitter.split_documents(raw_docs)  # 得到 chunked 文档列表

docs_subset = docs[10:20]  # 增加文档数量，避免数据太少

print(f"Generated {len(docs_subset)} chunks from PDF.")

merged_docs = []
for i in range(0, len(docs_subset), 2):
    merged_text = " ".join([d.page_content for d in docs_subset[i:i+2]])
    merged_docs.append(Document(page_content=merged_text))

print(f"Merged {len(merged_docs)} chunks from subset.")

# 3️⃣ 调用 Ragas 生成测试集
# 需要先创建 LangChain 模型对象
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings

llm = Tongyi(model="qwen-turbo")
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

generator = TestsetGenerator.from_langchain(
    llm=llm,
    embedding_model=embeddings,
)


# 5️⃣ 生成测试集
try:
    testset = generator.generate_with_langchain_docs(
        documents=merged_docs,
        testset_size=10,  # 减少数量，避免embedding问题
    )
    # 保存测试集
    source_file_path = '../data/testsets/ragas_source_testset.json'
    with open(source_file_path, "w", encoding="utf-8") as f:
        json.dump([sample.dict() for sample in testset], f, ensure_ascii=False, indent=2)
    
    print("测试集已保存到 ragas_source_testset.json")
    print(f"生成了 {len(testset)} 条测试数据")

    target_file = "../data/testsets/wait_test_testset2.json"  # 输出文件
    # 执行转换
    converted_data = convert_ragas_format(source_file_path, target_file)

except Exception as e:
    print(f"生成测试集时出错: {e}")
    print("\n可能的解决方案:")
    print("1. 增加文档数量 (docs_subset)")
    print("2. 减少 testset_size")
    print("3. 检查 DASHSCOPE_API_KEY 是否有效")
    print("4. 确保文档内容有足够的信息量")
    raise