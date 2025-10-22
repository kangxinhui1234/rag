"""
Spring AI HTTP 客户端
用于与 Spring AI 服务进行交互
"""

import requests
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class SearchRequest:
    """检索请求"""
    question: str
    top_k: int = 5
    vector_weight: float = 0.7
    bm25_weight: float = 0.3


@dataclass
class SearchResponse:
    """检索响应"""
    contexts: List[str]
    search_type: str
    response_time: float


@dataclass
class RagRequest:
    """RAG 请求"""
    question: str
    search_type: str = "hybrid"
    top_k: int = 5


@dataclass
class RagResponse:
    """RAG 响应"""
    question: str
    answer: str
    contexts: List[str]
    response_time: float
    search_type: str


class SpringAIClient:
    """Spring AI 客户端"""
    
    def __init__(self, base_url: str, timeout: int = 60):
        """
        初始化客户端
        
        Args:
            base_url: Spring AI 服务基础 URL
            timeout: 请求超时时间
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def vector_search(self, question: str, top_k: int = 5) -> SearchResponse:
        """
        向量检索
        
        Args:
            question: 问题
            top_k: 返回文档数量
            
        Returns:
            SearchResponse
        """
        return self._search(question, "vector", top_k)
    
    def bm25_search(self, question: str, top_k: int = 5) -> SearchResponse:
        """
        BM25 检索
        
        Args:
            question: 问题
            top_k: 返回文档数量
            
        Returns:
            SearchResponse
        """
        return self._search(question, "bm25", top_k)
    
    def hybrid_search(
        self, 
        question: str, 
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> SearchResponse:
        """
        混合检索
        
        Args:
            question: 问题
            top_k: 返回文档数量
            vector_weight: 向量权重
            bm25_weight: BM25 权重
            
        Returns:
            SearchResponse
        """
        url = f"{self.base_url}/api/rag/search/hybrid"
        payload = {
            "question": question,
            "topK": top_k,
            "vectorWeight": vector_weight,
            "bm25Weight": bm25_weight
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            response_time = time.time() - start_time
            
            return SearchResponse(
                contexts=data.get('contexts', []),
                search_type=f"hybrid_{vector_weight}_{bm25_weight}",
                response_time=response_time
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"混合检索失败: {e}")
            raise
    
    def rag_query(
        self, 
        question: str, 
        search_type: str = "hybrid",
        top_k: int = 5
    ) -> RagResponse:
        """
        完整 RAG 查询（检索 + 生成）
        
        Args:
            question: 问题
            search_type: 检索类型 (vector/bm25/hybrid)
            top_k: 返回文档数量
            
        Returns:
            RagResponse
        """
        url = f"{self.base_url}/api/rag/qa"
        payload = {
            "question": question,
            "searchType": search_type,
            "topK": top_k
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            response_time = time.time() - start_time
            
            return RagResponse(
                question=data.get('question', question),
                answer=data.get('answer', ''),
                contexts=data.get('contexts', []),
                response_time=response_time,
                search_type=search_type
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"RAG 查询失败: {e}")
            raise
    
    def _search(self, question: str, search_type: str, top_k: int) -> SearchResponse:
        """
        通用检索方法
        
        Args:
            question: 问题
            search_type: 检索类型
            top_k: 返回文档数量
            
        Returns:
            SearchResponse
        """
        url = f"{self.base_url}/api/rag/search/{search_type}"
        payload = {
            "question": question,
            "topK": top_k
        }
        
        start_time = time.time()
        try:
            response = self.session.post(
                url, 
                json=payload, 
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            response_time = time.time() - start_time
            
            return SearchResponse(
                contexts=data.get('contexts', []),
                search_type=search_type,
                response_time=response_time
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"{search_type} 检索失败: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            bool: 服务是否可用
        """
        try:
            url = f"{self.base_url}/actuator/health"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"健康检查失败: {e}")
            return False
    
    def close(self):
        """关闭会话"""
        self.session.close()


# 示例用法
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建客户端
    client = SpringAIClient("http://localhost:8080")
    
    # 测试健康检查
    if client.health_check():
        print("✅ Spring AI 服务可用")
    else:
        print("❌ Spring AI 服务不可用")
        exit(1)
    
    # 测试检索
    question = "2022年平安银行的净利润是多少？"
    
    # 向量检索
    vector_result = client.vector_search(question)
    print(f"\n向量检索结果 ({vector_result.response_time:.2f}s):")
    for i, ctx in enumerate(vector_result.contexts, 1):
        print(f"  {i}. {ctx[:100]}...")
    
    # BM25 检索
    bm25_result = client.bm25_search(question)
    print(f"\nBM25 检索结果 ({bm25_result.response_time:.2f}s):")
    for i, ctx in enumerate(bm25_result.contexts, 1):
        print(f"  {i}. {ctx[:100]}...")
    
    # 混合检索
    hybrid_result = client.hybrid_search(question)
    print(f"\n混合检索结果 ({hybrid_result.response_time:.2f}s):")
    for i, ctx in enumerate(hybrid_result.contexts, 1):
        print(f"  {i}. {ctx[:100]}...")
    
    # 完整 RAG
    rag_result = client.rag_query(question, search_type="hybrid")
    print(f"\nRAG 答案 ({rag_result.response_time:.2f}s):")
    print(f"  {rag_result.answer}")
    
    client.close()






