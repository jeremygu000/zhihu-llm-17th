# ================== 检索 + 重排 + 生成回答 ==================

import os
from typing import List, Any, Dict, Optional
from dataclasses import dataclass

# 复用你已有的 BGE Rerank 与 LLM
from rerank_service import BgeRerankService
from langchain_community.llms import Tongyi
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import DashScopeEmbeddings

from utils.knowledge_builder import KnowledgeBaseBuilder
from vector_stores.milvus_store import MilvusVectorStore

from dotenv import load_dotenv

# ---------- 环境初始化 ----------
load_dotenv()
_api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
if not _api_key:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


@dataclass
class RetrievalConfig:
    k_each: int = 6  # 每路检索抓取的 top-k
    oversample: int = 3  # 不支持 expr 时的超采样倍数
    final_top_k: int = 5  # Rerank 后保留
    use_aug: bool = True  # 是否合并 Doc2Query 问句向量
    prefer_base: float = (
        0.0  # （可选）对原始文档加成/惩罚(>0奖励，<0惩罚)，单位是 rerank 分数的线性偏置
    )


RCFG = RetrievalConfig()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "spdb_xian_rm_assessment_policy")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=_api_key,
)

reranker = BgeRerankService(model_name="BAAI/bge-reranker-base", use_fp16=True)
milvus_vs = MilvusVectorStore(
    collection_name=COLLECTION_NAME,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    embeddings=embeddings,
)
# 2) 给 milvus_vs 注入
try:
    milvus_vs.set_embeddings(embeddings)  # 如果你在类里实现了这个方法
except AttributeError:
    # 如果没有 set_embeddings，就确保在 open_existing_collection 之前：
    milvus_vs._embeddings = embeddings  # 让 open_existing_collection 用到它
milvus_vs.open_existing_collection(load=True)

kb = KnowledgeBaseBuilder(milvus_vs, embeddings=embeddings)
kb.attach_existing_collection(meta_dir="./vector_db_milvus_meta")

answer_llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=_api_key)


def _dedup_docs(docs: List[Any]) -> List[Any]:
    seen, out = set(), []
    for d in docs:
        key = (getattr(d, "page_content", "") or "").strip()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _try_expr_search(
    vs: MilvusVectorStore, query: str, k: int, expr: Optional[str], oversample: int
) -> List[Any]:
    """
    优先用 Milvus expr（aug_type == 'doc2query' 等）过滤；
    如果你的 MilvusVectorStore 不支持 expr 参数，就退化为：超采样抓取 -> Python 过滤。
    """
    # 1) 直接尝试 expr 参数
    try:
        if expr:
            return vs.similarity_search(query, k=k, expr=expr)  # 若不支持会抛 TypeError
        else:
            return vs.similarity_search(query, k=k)
    except TypeError:
        pass

    # 2) 退化：超采样 + 过滤
    raw = vs.similarity_search(query, k=k * oversample)
    if not expr:
        return raw

    def match_expr(doc: Any) -> bool:
        md = dict(getattr(doc, "metadata", {}) or {})
        val = md.get("aug_type")
        # 仅支持最常见的两种表达式
        if expr.strip().replace(" ", "") == "aug_type=='doc2query'":
            return val == "doc2query"
        if expr.strip().replace(" ", "") == "aug_type!='doc2query'":
            return val != "doc2query"
        return True  # 实在不认识表达式就不过滤

    return [d for d in raw if match_expr(d)][:k]


def retrieve_augmented(
    query: str,
    vector_store: MilvusVectorStore,
    k_each: int = RCFG.k_each,
    oversample: int = RCFG.oversample,
    include_aug: bool = RCFG.use_aug,
) -> List[Any]:
    """
    两路检索：基础文档（base） + Doc2Query 问句（aug）
    - base expr: aug_type != 'doc2query'
    - aug  expr: aug_type == 'doc2query'
    - 若不支持 expr，则超采样+Python 过滤
    """
    # base 检索（原始文档向量）
    base_docs = _try_expr_search(
        vector_store,
        query,
        k_each,
        expr="aug_type != 'doc2query'",
        oversample=oversample,
    )
    # aug 检索（问句向量）
    aug_docs: List[Any] = []
    if include_aug:
        aug_docs = _try_expr_search(
            vector_store,
            query,
            k_each,
            expr="aug_type == 'doc2query'",
            oversample=oversample,
        )

    merged = _dedup_docs(base_docs + aug_docs)
    return merged


def rerank_and_select(
    query: str,
    docs: List[Any],
    top_k: int = RCFG.final_top_k,
    prefer_base: float = RCFG.prefer_base,
) -> List[Any]:
    if not docs:
        return []
    pairs = [(query, getattr(d, "page_content", "") or "") for d in docs]
    scores = reranker.score_pairs(pairs)

    # （可选）对原始文档加成/惩罚
    if prefer_base != 0.0:
        for i, d in enumerate(docs):
            aug_type = (getattr(d, "metadata", {}) or {}).get("aug_type")
            if aug_type != "doc2query":  # base 文档
                scores[i] += prefer_base

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, s in ranked[:top_k]]


def answer_with_sources(
    query: str, top_docs: List[Any], kb: KnowledgeBaseBuilder
) -> Dict[str, Any]:
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
    chain = create_stuff_documents_chain(answer_llm, prompt)
    answer = chain.invoke({"context": top_docs, "question": query})

    # 收集来源（若你在 KB 里维护了 page_info）
    seen_pages = set()
    sources = []
    for d in top_docs:
        text_key = (getattr(d, "page_content", "") or "").strip()
        page = getattr(kb, "page_info", {}).get(text_key, None)
        if page and page not in seen_pages:
            seen_pages.add(page)
            sources.append({"page": page, "snippet": text_key[:90] + "..."})
    return {"answer": answer, "sources": sources}


# ================== Demo：检索 + 回答 ==================
if __name__ == "__main__":
    # —— 如果你把这段加到 offline_doc2query_ingest.py 里，milvus_vs / embeddings / kb 已经在上文初始化过了 ——
    user_q = "客户经理被投诉一次扣多少分？"

    # 1) 召回（合并 base + doc2query）
    candidates = retrieve_augmented(
        user_q,
        vector_store=milvus_vs,
        k_each=6,
        oversample=3,
        include_aug=True,  # 设为 False 则仅用原始文档
    )
    print(f"[召回候选] {len(candidates)} 条")

    # 2) 重排（按原始 query 打分）
    top_docs = rerank_and_select(user_q, candidates, top_k=5, prefer_base=0.0)
    print(f"[Rerank 后 Top-K] {len(top_docs)} 条")

    # 3) 生成回答 + 来源
    if top_docs:
        final = answer_with_sources(user_q, top_docs, kb)
        print("\n=== 回答 ===\n", final["answer"])
        print("\n=== 参考来源页码 ===")
        for s in final["sources"]:
            print(f"- 页码：{s['page']} …… {s['snippet']}")
    else:
        print("未检索到相关内容。")
