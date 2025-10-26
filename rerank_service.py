# -*- coding: utf-8 -*-
"""
Unified Rerank Service
兼容 FlagReranker 与 FlagLLMReranker
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")

try:
    from FlagEmbedding import FlagReranker, FlagLLMReranker
except ImportError:
    raise ImportError("请先安装: pip install FlagEmbedding")


class BgeRerankService:
    """
    统一的 Rerank Service，可支持：
      - BAAI/bge-reranker-base / large
      - BAAI/bge-reranker-v2-gemma 等 LLM 模型
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        use_fp16: bool = True,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size

        # 根据模型名自动选择实现类
        if "gemma" in model_name or "llm" in model_name.lower():
            print(f"[Reranker] Using FlagLLMReranker: {model_name}")
            self._model = FlagLLMReranker(model_name, use_fp16=use_fp16)
        else:
            print(f"[Reranker] Using FlagReranker: {model_name}")
            self._model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: int = 3,
        return_scores: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        对候选文本进行重排序（基础文本接口）
        """
        if not docs:
            return [], []

        pairs = [(query, d) for d in docs]
        scores = self._model.compute_score(pairs, batch_size=self.batch_size)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]
        order = [i for i, _ in top]
        top_scores = [float(s) for _, s in top] if return_scores else []
        return order, top_scores

    def rerank_docs(
        self,
        query: str,
        docs: List,
        top_k: int = 3,
        return_scores: bool = True,
    ):
        """
        支持 LangChain Document 列表的高层接口
        """
        contents = [getattr(d, "page_content", "") or "" for d in docs]
        order, scores = self.rerank(query, contents, top_k, return_scores)
        ranked_docs = [docs[i] for i in order]
        return ranked_docs, scores

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """对 (query, passage) 对逐一打分，返回与输入同长度的分数列表"""
        if not pairs:
            return []
        scores = self._model.compute_score(pairs, batch_size=self.batch_size)
        # FlagEmbedding 可能返回 numpy 类型，这里统一成 float
        return [float(s) for s in scores]

    def score(self, query: str, docs: List[str]) -> List[float]:
        """对一组文本打分（不排序不截断），长度与 docs 相同"""
        if not docs:
            return []
        pairs = [(query, d) for d in docs]
        return self.score_pairs(pairs)

    def score_docs(self, query: str, docs: List) -> List[float]:
        """对 LangChain Document 列表打分（不排序不截断）"""
        contents = [getattr(d, "page_content", "") or "" for d in docs]
        return self.score(query, contents)
