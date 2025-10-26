# -*- coding: utf-8 -*-
"""
内存 BM25 + Milvus 向量召回 + BGE Rerank + LLM 生成答案（不做 BM25 持久化）

依赖：
- vector_stores/milvus_store.py -> MilvusVectorStore
- utils/knowledge_builder.py -> KnowledgeBaseBuilder
- rerank_service.py -> BgeRerankService
- bm25_service.py -> BM25Service
- 环境变量：DASHSCOPE_API_KEY、MILVUS_HOST/MILVUS_PORT/MILVUS_COLLECTION
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# LangChain
from langchain.schema import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 本地模块
from bm25_service import BM25Service
from vector_stores.milvus_store import MilvusVectorStore
from utils.knowledge_builder import KnowledgeBaseBuilder
from rerank_service import BgeRerankService


# --------------------------
# 配置
# --------------------------


@dataclass
class HybridConfig:
    k_sparse: int = 20  # BM25 抓取量
    k_dense: int = 20  # 向量召回抓取量
    alpha: float = 0.6  # 融合权重（BM25 比重）
    final_top_k: int = 6  # Rerank 后保留数量
    use_expr_filter: bool = True  # 是否启用双路过滤


# --------------------------
# 工具函数
# --------------------------


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-12:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin + 1e-12) for v in values]


def _milvus_try_search(
    vs: MilvusVectorStore, query: str, k: int, expr: Optional[str]
) -> List[Document]:
    """优先用 expr；不支持则回退。"""
    try:
        return (
            vs.similarity_search(query, k=k, expr=expr)
            if expr
            else vs.similarity_search(query, k=k)
        )
    except TypeError:
        return vs.similarity_search(query, k=k)


def _build_bm25_docs_from_kb(kb: KnowledgeBaseBuilder) -> List[Document]:
    """从 page_info 构造临时 Document 列表"""
    docs: List[Document] = []
    page_info = getattr(kb, "page_info", {}) or {}
    for chunk, page in page_info.items():
        docs.append(Document(page_content=str(chunk), metadata={"page": page}))
    return docs


# --------------------------
# 混合检索器
# --------------------------


class HybridRetriever:
    def __init__(
        self, bm25: Optional[BM25Service], vs: MilvusVectorStore, cfg: HybridConfig
    ):
        self.bm25 = bm25
        self.vs = vs
        self.cfg = cfg

    def retrieve(self, query: str) -> List[Document]:
        sparse_pairs, dense_pairs = [], []

        # 1) 稀疏召回
        if self.bm25:
            sparse_pairs = self.bm25.topk_with_scores(query, self.cfg.k_sparse)

        # 2) 稠密召回
        if self.cfg.use_expr_filter:
            base_docs = _milvus_try_search(
                self.vs, query, self.cfg.k_dense, "aug_type != 'doc2query'"
            )
            aug_docs = _milvus_try_search(
                self.vs, query, self.cfg.k_dense, "aug_type == 'doc2query'"
            )
            dense_docs = base_docs + aug_docs
        else:
            dense_docs = self.vs.similarity_search(query, k=self.cfg.k_dense)

        dense_pairs = [(d, 1.0) for d in dense_docs]

        # 3) 归一化 + 融合
        s_docs, s_scores = zip(*sparse_pairs) if sparse_pairs else ([], [])
        s_norm = _minmax_norm(list(s_scores)) if s_scores else []
        d_docs, d_scores = zip(*dense_pairs) if dense_pairs else ([], [])
        d_norm = _minmax_norm(list(d_scores)) if d_scores else []

        s_norm_pairs = list(zip(s_docs, s_norm))
        d_norm_pairs = list(zip(d_docs, d_norm))
        pool: Dict[str, Tuple[Document, float]] = {}

        def put(doc: Document, val: float):
            key = (doc.page_content or "").strip()
            if key and (key not in pool or val > pool[key][1]):
                pool[key] = (doc, val)

        for doc, sv in s_norm_pairs:
            put(doc, self.cfg.alpha * sv)
        for doc, dv in d_norm_pairs:
            base_val = pool.get((doc.page_content or "").strip(), (doc, 0.0))[1]
            fused = base_val + (1.0 - self.cfg.alpha) * dv
            put(doc, fused)

        fused = sorted(pool.values(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in fused]


# --------------------------
# 生成答案
# --------------------------


def answer_with_sources(
    query: str, top_docs: List[Document], kb: KnowledgeBaseBuilder, llm: Tongyi
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

    chain = create_stuff_documents_chain(llm, prompt)
    answer = chain.invoke({"context": top_docs, "question": query})

    seen, sources = set(), []
    page_info: Dict[str, int] = getattr(kb, "page_info", {}) or {}
    for d in top_docs:
        key = (d.page_content or "").strip()
        page = page_info.get(key)
        if page and page not in seen:
            seen.add(page)
            sources.append({"page": page, "snippet": key[:90] + "..."})
    return {"answer": answer, "sources": sources}


# --------------------------
# 主流程
# --------------------------

if __name__ == "__main__":
    load_dotenv()

    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    COLLECTION = os.getenv("MILVUS_COLLECTION", "spdb_xian_rm_assessment_policy")
    META_DIR = os.getenv("MILVUS_META_DIR", "./vector_db_milvus_meta")

    # 1) 向量库
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", dashscope_api_key=api_key
    )
    vs = MilvusVectorStore(
        collection_name=COLLECTION,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        embeddings=embeddings,
    )
    try:
        vs.set_embeddings(embeddings)
    except Exception:
        pass

    try:
        vs.load_local(META_DIR, embeddings)
    except Exception:
        vs.open_existing_collection(load=True)

    # 2) KB
    kb = KnowledgeBaseBuilder(vector_store=vs, embeddings=embeddings)
    try:
        kb.attach_existing_collection(meta_dir=META_DIR)
    except Exception:
        pass

    # 3) BM25 内存构建
    bm25_docs = _build_bm25_docs_from_kb(kb)
    if not bm25_docs:
        print("⚠️ 未从 KB.page_info 构造出 chunk 文档，BM25 将被禁用，仅用向量检索。")
        bm25_svc = None
    else:
        try:
            import jieba

            tokenizer = lambda t: list(jieba.cut(t))
        except Exception:
            tokenizer = None
        bm25_svc = BM25Service(tokenizer=tokenizer)
        bm25_svc.build_from_docs(bm25_docs)
        print(f"✅ BM25 内存索引已构建：{len(bm25_docs)} 个 chunk")

    # 4) 检索器 & 模型
    hcfg = HybridConfig()
    hybrid = HybridRetriever(bm25=bm25_svc, vs=vs, cfg=hcfg)
    reranker = BgeRerankService(model_name="BAAI/bge-reranker-base", use_fp16=True)
    llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=api_key)

    # 5) 测试查询
    user_q = "客户经理被投诉一次扣多少分？"

    candidates = hybrid.retrieve(user_q)
    print(f"[召回候选] {len(candidates)} 条")

    # 🔍 候选预览
    print("\n[召回候选预览]")
    for i, d in enumerate(candidates[:10], 1):
        txt = (d.page_content or "").replace("\n", " ")[:90]
        aug = (d.metadata or {}).get("aug_type")
        page = (d.metadata or {}).get("page") or "?"
        print(f"{i:02d}. page={page} aug={aug} | {txt}")

    if not candidates:
        print("未检索到候选内容。请检查集合加载或查询条件。")
        raise SystemExit

    # 6) Rerank
    ranked_docs, scores = reranker.rerank_docs(
        user_q, candidates, top_k=hcfg.final_top_k, return_scores=True
    )
    top_docs = ranked_docs
    print(f"[Rerank 后 Top-K] {len(top_docs)} 条")

    # 7) 生成答案
    if top_docs:
        result = answer_with_sources(user_q, top_docs, kb, llm)
        print("\n=== 回答 ===\n", result["answer"])
        print("\n=== 参考来源页码 ===")
        for s in result["sources"]:
            print(f"- 页码：{s['page']} …… {s['snippet']}")
    else:
        print("未检索到相关内容。")
