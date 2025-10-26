# -*- coding: utf-8 -*-
"""
内存 BM25 + Milvus 向量召回 + BGE Rerank + LLM 生成答案（不做 BM25 持久化）

依赖你已有：
- vector_stores/milvus_store.py -> MilvusVectorStore（已支持 set_embeddings / open_existing_collection / load_local）
- utils/knowledge_builder.py -> KnowledgeBaseBuilder（attach_existing_collection 会加载 page_info）
- rerank_service.py -> BgeRerankService（支持 rerank_docs(query, docs, top_k) 或 rerank(pairs, top_k)）
- 环境变量：DASHSCOPE_API_KEY、MILVUS_HOST/MILVUS_PORT/MILVUS_COLLECTION（可选）
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict

from dotenv import load_dotenv

# LangChain / Embeddings / LLM
from langchain.schema import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 你的本地模块
from vector_stores.milvus_store import MilvusVectorStore
from utils.knowledge_builder import KnowledgeBaseBuilder
from rerank_service import BgeRerankService

# 稀疏检索（BM25）
from rank_bm25 import BM25Okapi

# --------------------------
# 配置与数据结构
# --------------------------


@dataclass
class HybridConfig:
    # 稀疏（BM25）与稠密（Milvus）两路召回各自的抓取量
    k_sparse: int = 20
    k_dense: int = 20
    # 融合权重：alpha 越大，BM25 的权重越高
    alpha: float = 0.5
    # Rerank 最终保留数量
    final_top_k: int = 6
    # 是否启用基于元数据的双路（base/doc2query）表达式过滤（你的 Milvus 若不支持，会自动回退）
    use_expr_filter: bool = True


# --------------------------
# 简易 BM25 内存服务（不持久化）
# --------------------------


class BM25Service:
    def __init__(self, tokenizer=None):
        # tokenizer: Callable[[str], List[str]]
        # 若未提供，默认按空白切分
        self.tokenizer = tokenizer or (lambda t: t.split())
        self._docs: List[Document] = []
        self._tokens: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def build_from_docs(self, docs: List[Document]) -> None:
        self._docs = docs
        self._tokens = [self.tokenizer(d.page_content or "") for d in docs]
        self._bm25 = BM25Okapi(self._tokens)

    def topk_with_scores(self, query: str, k: int) -> List[Tuple[Document, float]]:
        if not self._bm25 or not self._docs:
            return []
        q_tokens = self.tokenizer(query)
        scores = self._bm25.get_scores(q_tokens)  # numpy array
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in order]


# --------------------------
# 工具函数
# --------------------------


def _ensure_documents(items):
    from langchain.schema import Document

    out = []
    for x in items:
        if isinstance(x, Document):
            out.append(x)
        elif isinstance(x, (list, tuple)) and x and isinstance(x[0], Document):
            # 兼容 [(Document, score)] / [[Document, score]]
            out.append(x[0])
        elif (
            isinstance(x, dict)
            and "document" in x
            and isinstance(x["document"], Document)
        ):
            # 兼容一些reranker返回dict
            out.append(x["document"])
        else:
            # 最兜底：把字符串变成 Document；其余忽略
            try:
                from langchain.schema import Document

                if isinstance(x, str) and x.strip():
                    out.append(Document(page_content=x))
            except Exception:
                pass
    return out


def _dedup_docs_keep_best(
    pairs: List[Tuple[Document, float]],
) -> List[Tuple[Document, float]]:
    """对 page_content 去重，保留得分最高者。"""
    best: Dict[str, Tuple[Document, float]] = {}
    for doc, score in pairs:
        key = (doc.page_content or "").strip()
        if not key:
            continue
        if key not in best or score > best[key][1]:
            best[key] = (doc, score)
    return list(best.values())


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-12:
        # 全相等：直接返回 1.0
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin + 1e-12) for v in values]


def _milvus_try_search(
    vs: MilvusVectorStore, query: str, k: int, expr: Optional[str]
) -> List[Document]:
    """优先用 expr；不支持则回退到普通 similarity_search。"""
    try:
        if expr:
            # 你的 MilvusVectorStore.similarity_search 若无 expr 参数会抛 TypeError
            return vs.similarity_search(query, k=k, expr=expr)  # type: ignore
        else:
            return vs.similarity_search(query, k=k)
    except TypeError:
        # 回退
        return vs.similarity_search(query, k=k)


def _build_bm25_docs_from_kb(kb: KnowledgeBaseBuilder) -> List[Document]:
    """
    如果你在 ingest 时保存了 page_info（chunk->page），这里可直接从中构建 Document 列表。
    这是纯内存临时对象，不落盘。
    """
    docs: List[Document] = []
    page_info = getattr(kb, "page_info", {}) or {}
    for chunk, page in page_info.items():
        docs.append(Document(page_content=str(chunk), metadata={"page": page}))
    return docs


# --------------------------
# Hybrid 检索器（BM25 + 向量）
# --------------------------


class HybridRetriever:
    def __init__(
        self,
        bm25: Optional[BM25Service],
        vs: MilvusVectorStore,
        cfg: HybridConfig,
    ):
        self.bm25 = bm25
        self.vs = vs
        self.cfg = cfg

    def retrieve(self, query: str) -> List[Document]:
        """
        1) 稀疏 BM25 召回（内存，不持久化）
        2) 稠密 Milvus 召回（可选带 expr）
        3) MinMax 归一 & 融合
        4) 去重 & 排序
        """
        sparse_pairs: List[Tuple[Document, float]] = []
        dense_pairs: List[Tuple[Document, float]] = []

        # 1) 稀疏召回
        if self.bm25:
            sparse_pairs = self.bm25.topk_with_scores(query, self.cfg.k_sparse)

        # 2) 稠密召回：两路（base / doc2query） or 单路（不支持 expr）
        dense_docs: List[Document] = []
        if self.cfg.use_expr_filter:
            # base：aug_type != 'doc2query'
            base_docs = _milvus_try_search(
                self.vs, query, self.cfg.k_dense, "aug_type != 'doc2query'"
            )
            # aug：aug_type == 'doc2query'
            aug_docs = _milvus_try_search(
                self.vs, query, self.cfg.k_dense, "aug_type == 'doc2query'"
            )
            dense_docs = base_docs + aug_docs
        else:
            dense_docs = self.vs.similarity_search(query, k=self.cfg.k_dense)

        # 用等权得分（Milvus 没直接返回分数时），先临时置为 1.0，再归一化时当成同一常数
        dense_pairs = [(d, 1.0) for d in dense_docs]

        # 3) 分数归一化 + 融合
        # 稀疏
        s_docs, s_scores = zip(*sparse_pairs) if sparse_pairs else ([], [])
        s_norm = _minmax_norm(list(s_scores)) if s_scores else []
        s_norm_pairs = list(zip(s_docs, s_norm))

        # 稠密
        d_docs, d_scores = zip(*dense_pairs) if dense_pairs else ([], [])
        # 这里如果你希望向量侧也有相似度，可改造成 similarity_search_with_score；当前就当全 1.0
        d_norm = _minmax_norm(list(d_scores)) if d_scores else []
        d_norm_pairs = list(zip(d_docs, d_norm))

        # 合并（alpha * sparse + (1-alpha) * dense）
        pool: Dict[str, Tuple[Document, float]] = {}

        def put(doc: Document, val: float):
            key = (doc.page_content or "").strip()
            if not key:
                return
            if key not in pool or val > pool[key][1]:
                pool[key] = (doc, val)

        for doc, sv in s_norm_pairs:
            put(doc, self.cfg.alpha * sv)
        for doc, dv in d_norm_pairs:
            base = pool.get((doc.page_content or "").strip(), (doc, 0.0))[1]
            fused = base + (1.0 - self.cfg.alpha) * dv
            put(doc, fused)

        fused_list = sorted(pool.values(), key=lambda x: x[1], reverse=True)
        fused_docs = [d for d, _ in fused_list]
        return fused_docs


# --------------------------
# 生成答案（Stuff）
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

    # 来源页码（来自你保存的 page_info）
    seen = set()
    sources = []
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

    # 环境变量
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    COLLECTION = os.getenv("MILVUS_COLLECTION", "spdb_xian_rm_assessment_policy")
    META_DIR = os.getenv(
        "MILVUS_META_DIR", "./vector_db_milvus_meta"
    )  # 你保存 page_info 的目录

    # 1) 模型 & 向量库
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1", dashscope_api_key=api_key
    )
    vs = MilvusVectorStore(
        collection_name=COLLECTION,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        embeddings=embeddings,  # 关键：把 embeddings 注入给 vs
    )
    # 如果你的类不支持 set_embeddings，可跳过；如果支持，保险起见再设置一次
    try:
        vs.set_embeddings(embeddings)
    except Exception:
        pass
    # 连接既有集合（只读检索）
    try:
        # 如果你之前用 save_local() 存了 meta，可以用 load_local 恢复（会给内部 LCMilvus 注入 embedding）
        vs.load_local(META_DIR, embeddings)
    except Exception:
        # 没有 meta 文件就走直接连接
        vs.open_existing_collection(load=True)

    # 2) 构建 KB 句柄（用于拿 page_info）
    kb = KnowledgeBaseBuilder(vector_store=vs, embeddings=embeddings)
    try:
        kb.attach_existing_collection(meta_dir=META_DIR)  # 会加载 page_info.pkl
    except Exception:
        # 没有也不报错，BM25 就会空
        pass

    # 3) 内存 BM25：从 page_info 构造文档
    bm25_docs: List[Document] = _build_bm25_docs_from_kb(kb)
    if not bm25_docs:
        print("⚠️ 未从 KB.page_info 构造出 chunk 文档，BM25 将被禁用，仅用向量检索。")
        bm25_svc = None
    else:
        # 可选中文分词：jieba
        tokenizer = None
        try:
            import jieba

            tokenizer = lambda t: list(jieba.cut(t))
        except Exception:
            tokenizer = None  # 回退空白切分
        bm25_svc = BM25Service(tokenizer=tokenizer)
        bm25_svc.build_from_docs(bm25_docs)
        print(f"✅ BM25 内存索引已构建：{len(bm25_docs)} 个 chunk")

    # 4) 混合召回器
    hcfg = HybridConfig(
        k_sparse=20, k_dense=20, alpha=0.6, final_top_k=6, use_expr_filter=True
    )
    hybrid = HybridRetriever(bm25=bm25_svc, vs=vs, cfg=hcfg)

    # 5) Reranker + LLM
    reranker = BgeRerankService(model_name="BAAI/bge-reranker-base", use_fp16=True)
    llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=api_key)

    # 6) 测试查询
    user_q = "客户经理被投诉一次扣多少分？"

    # 6.1 召回（BM25 + Milvus 融合）
    candidates = hybrid.retrieve(user_q)
    print(f"[召回候选] {len(candidates)} 条")

    # 6.2 Rerank（统一成只传 Document）
    raw_top = None
    if hasattr(reranker, "rerank_docs"):
        raw_top = reranker.rerank_docs(user_q, candidates, top_k=hcfg.final_top_k)
    else:
        pairs = [(user_q, (d.page_content or "")) for d in candidates]
        raw_top = reranker.rerank(pairs, top_k=hcfg.final_top_k)

    top_docs = _ensure_documents(raw_top)  # ⭐️新增：强制转 List[Document]
    print(f"[Rerank 后 Top-K] {len(top_docs)} 条")

    print(f"[Rerank 后 Top-K] {len(top_docs)} 条")

    # 6.3 生成答案 + 来源页码
    if top_docs:
        result = answer_with_sources(user_q, top_docs, kb, llm)
        print("\n=== 回答 ===\n", result["answer"])
        print("\n=== 参考来源页码 ===")
        for s in result["sources"]:
            print(f"- 页码：{s['page']} …… {s['snippet']}")
    else:
        print("未检索到相关内容。")
