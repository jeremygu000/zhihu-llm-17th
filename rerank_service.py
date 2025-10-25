# -*- coding: utf-8 -*-
"""
BGE-Rerank Service class
可本地直接调用，也可后续扩展为 FastAPI 接口版本。
"""
from __future__ import annotations
from typing import List, Tuple, Optional
from FlagEmbedding import FlagReranker


class BgeRerankService:
    """
    用于对候选文档进行重排序的 BGE rerank 服务。
    支持 batch 计算、得分输出，可独立注入到 RAG 或 QA pipeline。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        use_fp16: bool = True,
        batch_size: int = 16,
    ):
        """
        初始化 Reranker。

        参数:
            model_name: BGE Reranker 模型名称 (base / large)
            use_fp16: 是否启用 FP16 推理 (GPU 更快; CPU 会自动降级)
            batch_size: 批量计算大小
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self._model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: int = 3,
        return_scores: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        对候选文档进行重排。这是一个底层方法，适合在你只关心文本内容、不依赖 LangChain Document 对象时使用。

        参数:
            query: 查询字符串
            docs: 候选文本列表
            top_k: 返回的TopK数量
            return_scores: 是否返回相关性得分
        返回:
            (排序后索引列表, 对应得分列表)
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
        接受 langchain Document 对象列表，返回重排后的 Document 列表及分数。
        输入 / 输出都基于 LangChain Document 对象
        它是一个高层封装，内部会自动提取 page_content、按分数排序并返回新的文档列表。
        """
        contents = [getattr(d, "page_content", "") or "" for d in docs]
        order, scores = self.rerank(query, contents, top_k, return_scores)
        ranked_docs = [docs[i] for i in order]
        return ranked_docs, scores
