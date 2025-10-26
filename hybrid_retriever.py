# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Any, Dict, Optional, Tuple
import numpy as np

from bm25_service import BM25Service
from vector_stores.milvus_store import MilvusVectorStore


def _safe_minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo + 1e-9)


def _dedup(docs: List[Any]) -> List[Any]:
    seen, out = set(), []
    for d in docs:
        k = (getattr(d, "page_content", "") or "").strip()
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


class HybridRetriever:
    """
    稀疏(BM25) + 稠密(Milvus) 融合：
    hybrid = alpha * bm25_norm + (1 - alpha) * dense_norm
    """

    def __init__(
        self,
        bm25: BM25Service,
        vs: MilvusVectorStore,
        *,
        k_sparse: int = 20,
        k_dense: int = 20,
        alpha: float = 0.55,  # 中文制度/规则类，BM25 通常更强
        use_expr: bool = False,  # 需要区分 base/doc2query 时可开
        expr_base: str = "aug_type != 'doc2query'",
        expr_aug: str = "aug_type == 'doc2query'",
    ):
        self.bm25 = bm25
        self.vs = vs
        self.k_sparse = k_sparse
        self.k_dense = k_dense
        self.alpha = alpha
        self.use_expr = use_expr
        self.expr_base = expr_base
        self.expr_aug = expr_aug

    # —— 稠密召回（若不支持 expr 参数，就走无 expr） ——
    def _dense_candidates(self, query: str) -> List[Any]:
        if not self.use_expr:
            return self.vs.similarity_search(query, k=self.k_dense)
        try:
            base = self.vs.similarity_search(query, k=self.k_dense, expr=self.expr_base)
            aug = self.vs.similarity_search(query, k=self.k_dense, expr=self.expr_aug)
            return _dedup(base + aug)
        except TypeError:
            # 你的 MilvusVectorStore 不支持 expr
            return self.vs.similarity_search(query, k=self.k_dense)

    def retrieve(self, query: str, top_k: int = 10) -> List[Any]:
        # 1) 稀疏分（与 bm25.docs 对齐）
        sparse_scores = np.array(self.bm25.scores(query))
        all_docs = self.bm25.docs

        # 2) 稠密候选 -> 映射到“全量 doc 索引”得分向量
        dense_docs = self._dense_candidates(query)

        # 若你的 Milvus 封装支持 with_score，可在这里改成真实分数：
        # e.g. pairs = self.vs.similarity_search_with_score(...)

        # 简洁近似：位置打分（越靠前越高）
        dense_score_map: Dict[str, float] = {}
        for rank, d in enumerate(dense_docs):
            key = (getattr(d, "page_content", "") or "").strip()
            dense_score_map[key] = 1.0 / (rank + 1)

        dense_scores = np.array(
            [
                dense_score_map.get((getattr(d, "page_content", "") or "").strip(), 0.0)
                for d in all_docs
            ]
        )

        # 3) 归一化 + 融合
        s_norm = _safe_minmax(sparse_scores)
        d_norm = _safe_minmax(dense_scores)
        hybrid = self.alpha * s_norm + (1.0 - self.alpha) * d_norm

        idx = np.argsort(hybrid)[::-1][:top_k]
        return [all_docs[i] for i in idx]
