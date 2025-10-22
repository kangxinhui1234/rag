# Spring AI 接口配置指南

本指南介绍如何在你的 Spring AI 项目中添加测试接口。

## 📋 前提条件

- Spring Boot 3.x
- Spring AI
- Elasticsearch 8.x (已配置向量搜索和 BM25)
- Java 17+

## 🔧 添加依赖

在 `pom.xml` 中添加（如果还没有）:

```xml
<dependencies>
    <!-- Spring AI -->
    <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-core</artifactId>
        <version>0.8.0</version>
    </dependency>
    
    <!-- Elasticsearch -->
    <dependency>
        <groupId>co.elastic.clients</groupId>
        <artifactId>elasticsearch-java</artifactId>
        <version>8.11.0</version>
    </dependency>
    
    <!-- Lombok (可选，简化代码) -->
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <optional>true</optional>
    </dependency>
</dependencies>
```

## 📝 创建数据模型

### 1. 请求模型

```java
// src/main/java/com/yourpackage/model/SearchRequest.java
package com.yourpackage.model;

import lombok.Data;

@Data
public class SearchRequest {
    private String question;
    private Integer topK = 5;
    private Double vectorWeight = 0.7;
    private Double bm25Weight = 0.3;
}
```

```java
// src/main/java/com/yourpackage/model/RagRequest.java
package com.yourpackage.model;

import lombok.Data;

@Data
public class RagRequest {
    private String question;
    private String searchType = "hybrid"; // vector, bm25, hybrid
    private Integer topK = 5;
}
```

### 2. 响应模型

```java
// src/main/java/com/yourpackage/model/SearchResponse.java
package com.yourpackage.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import java.util.List;

@Data
@AllArgsConstructor
public class SearchResponse {
    private List<String> contexts;
    private String searchType;
    private Long responseTime; // 毫秒
}
```

```java
// src/main/java/com/yourpackage/model/RagResponse.java
package com.yourpackage.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import java.util.List;

@Data
@AllArgsConstructor
public class RagResponse {
    private String question;
    private String answer;
    private List<String> contexts;
    private Long responseTime; // 毫秒
    private String searchType;
}
```

## 🛠️ 实现服务层

### 1. 向量检索服务

```java
// src/main/java/com/yourpackage/service/VectorSearchService.java
package com.yourpackage.service;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.query_dsl.Query;
import co.elastic.clients.elasticsearch.core.SearchRequest;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.stereotype.Service;
import lombok.RequiredArgsConstructor;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class VectorSearchService {
    
    private final ElasticsearchClient esClient;
    private final EmbeddingClient embeddingClient;
    
    public List<String> search(String question, int topK) throws Exception {
        // 1. 生成问题的向量
        List<Double> queryVector = embeddingClient.embed(question);
        
        // 2. 构建 KNN 查询
        SearchRequest searchRequest = SearchRequest.of(s -> s
            .index("knowledge_base")  // 你的索引名称
            .size(topK)
            .knn(knn -> knn
                .field("embedding")  // 向量字段名
                .queryVector(queryVector.stream()
                    .map(Double::floatValue)
                    .collect(Collectors.toList()))
                .k(topK)
                .numCandidates(topK * 10)
            )
        );
        
        // 3. 执行查询
        SearchResponse<Document> response = esClient.search(
            searchRequest, 
            Document.class
        );
        
        // 4. 提取文档内容
        return response.hits().hits().stream()
            .map(hit -> hit.source().getContent())
            .collect(Collectors.toList());
    }
}
```

### 2. BM25 检索服务

```java
// src/main/java/com/yourpackage/service/BM25SearchService.java
package com.yourpackage.service;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.SearchRequest;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import org.springframework.stereotype.Service;
import lombok.RequiredArgsConstructor;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class BM25SearchService {
    
    private final ElasticsearchClient esClient;
    
    public List<String> search(String question, int topK) throws Exception {
        // 使用 match 查询（默认使用 BM25 算法）
        SearchRequest searchRequest = SearchRequest.of(s -> s
            .index("knowledge_base")
            .size(topK)
            .query(q -> q
                .match(m -> m
                    .field("content")  // 文本字段名
                    .query(question)
                )
            )
        );
        
        SearchResponse<Document> response = esClient.search(
            searchRequest, 
            Document.class
        );
        
        return response.hits().hits().stream()
            .map(hit -> hit.source().getContent())
            .collect(Collectors.toList());
    }
}
```

### 3. 混合检索服务

```java
// src/main/java/com/yourpackage/service/HybridSearchService.java
package com.yourpackage.service;

import org.springframework.stereotype.Service;
import lombok.RequiredArgsConstructor;

import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class HybridSearchService {
    
    private final VectorSearchService vectorSearchService;
    private final BM25SearchService bm25SearchService;
    
    public List<String> search(
        String question, 
        int topK,
        double vectorWeight,
        double bm25Weight
    ) throws Exception {
        
        // 1. 分别执行两种检索（多检索一些候选）
        int candidateSize = topK * 2;
        List<String> vectorResults = vectorSearchService.search(question, candidateSize);
        List<String> bm25Results = bm25SearchService.search(question, candidateSize);
        
        // 2. 使用 RRF (Reciprocal Rank Fusion) 融合
        Map<String, Double> scoreMap = new HashMap<>();
        
        // 向量检索结果打分
        for (int i = 0; i < vectorResults.size(); i++) {
            String doc = vectorResults.get(i);
            double score = vectorWeight / (i + 60);  // RRF 常数 k=60
            scoreMap.merge(doc, score, Double::sum);
        }
        
        // BM25 结果打分
        for (int i = 0; i < bm25Results.size(); i++) {
            String doc = bm25Results.get(i);
            double score = bm25Weight / (i + 60);
            scoreMap.merge(doc, score, Double::sum);
        }
        
        // 3. 按分数排序并返回 topK
        return scoreMap.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .limit(topK)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }
}
```

### 4. 生成服务

```java
// src/main/java/com/yourpackage/service/GenerationService.java
package com.yourpackage.service;

import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.stereotype.Service;
import lombok.RequiredArgsConstructor;

import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class GenerationService {
    
    private final ChatClient chatClient;
    
    private static final String RAG_PROMPT_TEMPLATE = """
        基于以下上下文回答问题。如果上下文中没有相关信息，请说"无法基于提供的信息回答"。
        
        上下文:
        {context}
        
        问题: {question}
        
        答案:
        """;
    
    public String generate(String question, List<String> contexts) {
        // 1. 组合上下文
        String contextStr = String.join("\n\n", contexts);
        
        // 2. 构建提示词
        PromptTemplate promptTemplate = new PromptTemplate(RAG_PROMPT_TEMPLATE);
        Prompt prompt = promptTemplate.create(Map.of(
            "context", contextStr,
            "question", question
        ));
        
        // 3. 调用 LLM
        return chatClient.call(prompt).getResult().getOutput().getContent();
    }
}
```

## 🌐 创建 Controller

```java
// src/main/java/com/yourpackage/controller/RagTestController.java
package com.yourpackage.controller;

import com.yourpackage.model.*;
import com.yourpackage.service.*;
import org.springframework.web.bind.annotation.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/rag")
@RequiredArgsConstructor
public class RagTestController {
    
    private final VectorSearchService vectorSearchService;
    private final BM25SearchService bm25SearchService;
    private final HybridSearchService hybridSearchService;
    private final GenerationService generationService;
    
    @PostMapping("/search/vector")
    public SearchResponse vectorSearch(@RequestBody SearchRequest request) {
        long startTime = System.currentTimeMillis();
        
        try {
            List<String> contexts = vectorSearchService.search(
                request.getQuestion(), 
                request.getTopK()
            );
            
            long responseTime = System.currentTimeMillis() - startTime;
            return new SearchResponse(contexts, "vector", responseTime);
            
        } catch (Exception e) {
            log.error("Vector search failed", e);
            throw new RuntimeException("Vector search failed: " + e.getMessage());
        }
    }
    
    @PostMapping("/search/bm25")
    public SearchResponse bm25Search(@RequestBody SearchRequest request) {
        long startTime = System.currentTimeMillis();
        
        try {
            List<String> contexts = bm25SearchService.search(
                request.getQuestion(), 
                request.getTopK()
            );
            
            long responseTime = System.currentTimeMillis() - startTime;
            return new SearchResponse(contexts, "bm25", responseTime);
            
        } catch (Exception e) {
            log.error("BM25 search failed", e);
            throw new RuntimeException("BM25 search failed: " + e.getMessage());
        }
    }
    
    @PostMapping("/search/hybrid")
    public SearchResponse hybridSearch(@RequestBody SearchRequest request) {
        long startTime = System.currentTimeMillis();
        
        try {
            List<String> contexts = hybridSearchService.search(
                request.getQuestion(), 
                request.getTopK(),
                request.getVectorWeight(),
                request.getBm25Weight()
            );
            
            long responseTime = System.currentTimeMillis() - startTime;
            return new SearchResponse(contexts, "hybrid", responseTime);
            
        } catch (Exception e) {
            log.error("Hybrid search failed", e);
            throw new RuntimeException("Hybrid search failed: " + e.getMessage());
        }
    }
    
    @PostMapping("/qa")
    public RagResponse ragQuery(@RequestBody RagRequest request) {
        long startTime = System.currentTimeMillis();
        
        try {
            // 1. 检索
            List<String> contexts = switch (request.getSearchType()) {
                case "vector" -> vectorSearchService.search(
                    request.getQuestion(), 
                    request.getTopK()
                );
                case "bm25" -> bm25SearchService.search(
                    request.getQuestion(), 
                    request.getTopK()
                );
                case "hybrid" -> hybridSearchService.search(
                    request.getQuestion(), 
                    request.getTopK(),
                    0.7, 0.3
                );
                default -> throw new IllegalArgumentException(
                    "Unknown search type: " + request.getSearchType()
                );
            };
            
            // 2. 生成答案
            String answer = generationService.generate(request.getQuestion(), contexts);
            
            long responseTime = System.currentTimeMillis() - startTime;
            
            return new RagResponse(
                request.getQuestion(),
                answer,
                contexts,
                responseTime,
                request.getSearchType()
            );
            
        } catch (Exception e) {
            log.error("RAG query failed", e);
            throw new RuntimeException("RAG query failed: " + e.getMessage());
        }
    }
}
```

## ⚙️ 配置文件

```yaml
# application.yml
spring:
  ai:
    openai:
      api-key: ${OPENAI_API_KEY}
      model: gpt-3.5-turbo
  
  elasticsearch:
    uris: http://localhost:9200
    username: ${ES_USERNAME:elastic}
    password: ${ES_PASSWORD:changeme}

# 自定义配置
app:
  knowledge-base:
    index-name: knowledge_base
    embedding-dimension: 1536
```

## 🚦 启动和测试

### 1. 启动应用

```bash
mvn spring-boot:run
```

### 2. 测试接口

```bash
# 测试向量检索
curl -X POST http://localhost:8080/api/rag/search/vector \
  -H "Content-Type: application/json" \
  -d '{"question":"2022年净利润是多少？","topK":5}'

# 测试完整 RAG
curl -X POST http://localhost:8080/api/rag/qa \
  -H "Content-Type: application/json" \
  -d '{"question":"2022年净利润是多少？","searchType":"hybrid","topK":5}'
```

## 📚 Document 实体类

```java
// src/main/java/com/yourpackage/entity/Document.java
package com.yourpackage.entity;

import lombok.Data;

@Data
public class Document {
    private String id;
    private String content;
    private List<Float> embedding;
    private Map<String, Object> metadata;
}
```

## ✅ 验证清单

- [ ] Elasticsearch 已启动并可访问
- [ ] 知识库索引已创建并包含数据
- [ ] 向量字段已正确配置（密集向量类型）
- [ ] Spring AI 应用已启动
- [ ] 所有 4 个测试接口可访问
- [ ] 测试查询返回预期结果

## 🔍 故障排除

### 问题 1: 向量检索返回空结果

**检查**:
```bash
# 验证索引中有向量数据
GET knowledge_base/_search
{
  "query": {
    "exists": {
      "field": "embedding"
    }
  }
}
```

### 问题 2: CORS 错误

添加 CORS 配置:
```java
@Configuration
public class WebConfig implements WebMvcConfigurer {
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
                .allowedOrigins("*")
                .allowedMethods("GET", "POST", "PUT", "DELETE");
    }
}
```

---

配置完成后，就可以使用 Python 测试框架进行测试了！






