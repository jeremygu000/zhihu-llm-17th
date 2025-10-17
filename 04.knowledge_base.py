# -*- coding: utf-8 -*-
from __future__ import annotations
from langchain_community.embeddings import DashScopeEmbeddings  # 任选 Embeddings
from vector_store.faiss_store import FaissVectorStore
from meta_store.redis_store import RedisMetaStore
from kb_service.KnowledgeBaseService import KnowledgeBaseService
import os
from dotenv import load_dotenv

load_dotenv()

FAISS_DIR = "./faiss_index/policy_kb"
REDIS_URL = "redis://127.0.0.1:6379/0"
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY') 

embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=DASHSCOPE_API_KEY)

vs = FaissVectorStore(path=FAISS_DIR)
ms = RedisMetaStore(url=REDIS_URL, namespace="policy_kb")
kb = KnowledgeBaseService(vector_store=vs, meta_store=ms, embeddings=embeddings)

# 1) 从 PDF 构建
n = kb.build_from_pdf(
    pdf_path="./policy.pdf",
    global_meta={"book": "考核细则", "version": 1},
    chunk_size=1000,
    chunk_overlap=200,
    persist=True,
    text_mode="text",   # 若是扫描件可改 "ocr"
)
print("写入 chunks:", n)

# 2) 查询（返回 文档内容 + Redis 元数据）
for doc, meta in kb.similarity_search("投诉一次扣多少分？", k=3):
    print("== 命中 ==")
    print("内容：", doc.page_content[:120].replace("\n", " "))
    print("元数据：", meta)
