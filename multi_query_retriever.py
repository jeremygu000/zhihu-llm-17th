# -*- coding: utf-8 -*-
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from MilvasRetriever import MilvusRetriever
from utils.knowledge_builder import KnowledgeBaseBuilder
from vector_stores.milvus_store import MilvusVectorStore
from langchain.callbacks.base import BaseCallbackHandler

import os

# === 环境变量 ===
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# === 初始化模型 ===
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)

# === 初始化 Milvus 向量库 ===
milvus_vs = MilvusVectorStore(
    collection_name="spdb_xian_rm_assessment_policy",
    connection_args={"host": "localhost", "port": "19530"},
)

# 如果你有本地保存的元数据，可以加载
kb = KnowledgeBaseBuilder(milvus_vs, embeddings)
kb.attach_existing_collection(meta_dir="./vector_db_milvus_meta")


# === 创建 MultiQueryRetriever ===
retriever = MultiQueryRetriever.from_llm(
    retriever=MilvusRetriever(milvus_vs, k=4), llm=llm
)

# === 测试查询 ===
query = "客户经理的考核标准是什么？"

from langchain.callbacks.base import BaseCallbackHandler


class MQPrintHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n[MultiQueryRetriever] LLM Prompt:")
        for p in prompts:
            print(p)

    def on_llm_end(self, response, **kwargs):
        try:
            text = response.generations[0][0].text
            print("\n[MultiQueryRetriever] LLM Output:")
            print(text)  # 里面通常包含改写的 queries
        except Exception:
            pass


results = retriever.invoke(query, config={"callbacks": [MQPrintHandler()]})

print("=== 多查询检索结果 ===")
for i, doc in enumerate(results, 1):
    print(f"[{i}] {doc.page_content[:50]}...\n")
