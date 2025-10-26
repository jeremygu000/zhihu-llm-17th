# -*- coding: utf-8 -*-
"""
RAG pipeline with Bidirectional Rewriting + Milvus + BGE-Rerank (online)

依赖：
- 你已有的：
  - vector_stores.milvus_store.MilvusVectorStore
  - utils.knowledge_builder.KnowledgeBaseBuilder
  - rerank_service.BGERerankService  （我们前面写过的 BGE 重排 Service）

- 本文件会用到：
  - bi_directional_rewriter.BidirectionalRewriter  （前一条消息给你的代码）
  - 环境变量：DASHSCOPE_API_KEY (for Query2Doc/Doc2Query)
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi

# 你已有的三件套
from BidirectionalRewriter import BidirectionalRewriter
from vector_stores.milvus_store import MilvusVectorStore
from utils.knowledge_builder import KnowledgeBaseBuilder
from rerank_service import BgeRerankService

# ----------------------------
# 配置
# ----------------------------
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


@dataclass
class PipelineConfig:
    milvus_collection: str = "spdb_xian_rm_assessment_policy"
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    meta_dir: str = "./vector_db_milvus_meta"  # 仅存放页码等元数据（非向量）
    search_k_each: int = 4  # 每条子查询在向量库取多少条
    max_subqueries: int = 4  # Doc2Query@Online 生成多少条
    final_top_k: int = 5  # Rerank 后保留多少条作答案上下文


CFG = PipelineConfig()

# ----------------------------
# 初始化组件
# ----------------------------
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

milvus_vs = MilvusVectorStore(
    collection_name=CFG.milvus_collection,
    connection_args={"host": CFG.milvus_host, "port": CFG.milvus_port},
)
kb = KnowledgeBaseBuilder(milvus_vs, embeddings)
# 如果你保存过页码/元数据，这里挂载；没有也可略过
kb.attach_existing_collection(meta_dir=CFG.meta_dir)

# 双向改写（Qwen）
rewriter = BidirectionalRewriter()  # 使用 bi_directional_rewriter.py 里的默认 LLMConfig

# BGE Rerank（你可以替换为 v2 / large 等）
reranker = BgeRerankService("BAAI/bge-reranker-base", use_fp16=True)

# （可选）回答模型
llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)


# ----------------------------
# 辅助：去重 & 截断
# ----------------------------
def _dedup_docs(docs):
    seen = set()
    out = []
    for d in docs:
        key = (getattr(d, "page_content", "") or "").strip()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


# ----------------------------
# 主流程：带双向改写的在线检索
# ----------------------------
def retrieve_with_bidirectional(query: str) -> List[Any]:
    """
    1) Query2Doc 扩写
    2) Doc2Query@Online 生成子查询
    3) 多路向量检索 → 合并去重
    4) BGE 重排 → 取 Top-K
    """
    # 1) Query2Doc：把短 query 扩写为“伪文档”
    pseudo_doc = rewriter.query2doc(query)
    print("[Query2Doc：]:", pseudo_doc)

    # 2) Doc2Query@Online：针对伪文档生成 3~4 个子查询
    subqueries = rewriter.doc2queries(pseudo_doc, n=CFG.max_subqueries)

    # 也把「原始查询」放进去，确保一致性
    subqueries = [query] + [sq for sq in subqueries if sq and sq.strip()]
    print("\n[子查询]：")
    for i, sq in enumerate(subqueries, 1):
        print(f"  {i}. {sq}")

    # 3) 每条子查询到 Milvus 检索 top-k
    all_docs = []
    for sq in subqueries:
        docs = milvus_vs.similarity_search(sq, k=CFG.search_k_each)
        all_docs.extend(docs)

    # 合并去重
    all_docs = _dedup_docs(all_docs)
    print(f"\n[合并后候选文档数] {len(all_docs)}")

    if not all_docs:
        return []

    # 4) 用 BGE-Rerank 按「原始用户 query」重排
    # BGE 输入是 (query, passage) 对；我们对所有候选 passage 打分
    scores = reranker.score_docs(query, all_docs)  # 返回与 all_docs 等长的分数列表

    # 排序并取 Top-K
    scored = list(zip(all_docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [d for d, s in scored[: CFG.final_top_k]]

    print(f"[Rerank 后保留] {len(top_docs)} 条")
    return top_docs


# ----------------------------
# （可选）把检索结果交给 LLM 进行回答
# ----------------------------
def answer_with_sources(query: str, top_docs: List[Any]) -> Dict[str, Any]:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        """
你是一位中文检索问答助手。仅依据给定文档回答；若找不到明确答案，请说“未在文档中找到明确答案”。

<context>
{context}
</context>

问题：{question}
请以简洁、准确、可引用原文措辞的中文作答。
""".strip()
    )
    chain = create_stuff_documents_chain(llm, prompt)
    resp = chain.invoke({"context": top_docs, "question": query})

    # 收集来源页码（如果你在 KnowledgeBaseBuilder 里维护了 page_info）
    seen_pages = set()
    sources = []
    for d in top_docs:
        text_key = (getattr(d, "page_content", "") or "").strip()
        page = getattr(kb, "page_info", {}).get(text_key, None)
        if page and page not in seen_pages:
            seen_pages.add(page)
            sources.append({"page": page, "snippet": text_key[:80] + "..."})

    return {"answer": resp, "sources": sources}


# ----------------------------
# Demo
# ----------------------------
if __name__ == "__main__":
    user_q = "客户经理被投诉一次扣多少分？"
    docs = retrieve_with_bidirectional(user_q)

    if not docs:
        print("未检索到相关内容。")
    else:
        final = answer_with_sources(user_q, docs)
        print("\n=== 回答 ===\n", final["answer"])
        print("\n=== 参考来源页码 ===")
        for s in final["sources"]:
            print(f"- 页码：{s['page']}  …… {s['snippet']}")
