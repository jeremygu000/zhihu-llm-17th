# -*- coding: utf-8 -*-
"""
Unified Rerank Service (enhanced)
兼容 FlagReranker 与 FlagLLMReranker；更稳的批处理与返回、可选归一化/加权、调试输出。
"""
from __future__ import annotations
from typing import List, Tuple, Sequence, Optional, Union
import warnings
import math

warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")

try:
    from FlagEmbedding import FlagReranker, FlagLLMReranker
except ImportError:
    raise ImportError("请先安装: pip install FlagEmbedding")

try:
    from langchain.schema import Document  # 仅为类型提示与 metadata 访问
except Exception:
    Document = None  # 允许在非 LangChain 环境下使用（只传 str）

TextLike = Union[str, "Document"]


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
        self.batch_size = max(1, batch_size)

        # 根据模型名自动选择实现类（gemma / llm 走 LLMReranker）
        if "gemma" in model_name.lower() or "llm" in model_name.lower():
            print(f"[Reranker] Using FlagLLMReranker: {model_name}")
            self._model = FlagLLMReranker(model_name, use_fp16=use_fp16)
        else:
            print(f"[Reranker] Using FlagReranker: {model_name}")
            self._model = FlagReranker(model_name, use_fp16=use_fp16)

    # -----------------------------
    # 内部工具
    # -----------------------------
    @staticmethod
    def _to_text(d: TextLike) -> str:
        if hasattr(d, "page_content"):
            return getattr(d, "page_content", "") or ""
        return str(d or "")

    @staticmethod
    def _get_aug_type(d: TextLike) -> Optional[str]:
        """从 Document.metadata 读取 aug_type（用于加权偏置）；非 Document 返回 None。"""
        if hasattr(d, "metadata"):
            md = getattr(d, "metadata", None) or {}
            return md.get("aug_type")
        return None

    def _batched_compute_score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """稳健的分批打分：兼容单样本返回 float、批量返回 list 的差异。"""
        if not pairs:
            return []
        out: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i : i + self.batch_size]
            sc = self._model.compute_score(chunk, batch_size=self.batch_size)
            # 兼容返回 float / list / numpy
            if isinstance(sc, (float, int)):
                sc = [float(sc)]
            else:
                sc = [float(x) for x in sc]  # numpy -> python float
            out.extend(sc)
        return out

    # -----------------------------
    # 打分 API
    # -----------------------------
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """对 (query, passage) 对逐一打分，返回与输入同长度的分数列表"""
        return self._batched_compute_score(pairs)

    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        """对一组纯文本打分（不排序不截断）"""
        if not docs:
            return []
        pairs = [(query, d) for d in docs]
        return self._batched_compute_score(pairs)

    def score_docs(self, query: str, docs: Sequence[TextLike]) -> List[float]:
        """对一组 Document/str 混合打分（不排序不截断）"""
        pairs = [(query, self._to_text(d)) for d in docs]
        return self._batched_compute_score(pairs)

    # -----------------------------
    # 重排 API（纯文本）
    # -----------------------------
    def rerank(
        self,
        query: str,
        docs: Sequence[str],
        top_k: int = 3,
        return_scores: bool = True,
        *,
        normalize: Optional[str] = None,  # None | "minmax" | "softmax"
        prefer_base_bias: float = 0.0,  # 无效（纯文本无 metadata），保留同名参数便于接口统一
        debug: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """
        对候选文本进行重排序（基础文本接口）。
        normalize: 分数归一化方案；"softmax"利于阈值化,"minmax"利于线性融合。
        prefer_base_bias: 仅对 Document 生效，这里占位不影响。
        """
        if not docs:
            return [], []
        pairs = [(query, d) for d in docs]
        scores = self._batched_compute_score(pairs)  # 与 docs 对齐

        # 归一化（可选）
        scores = self._normalize(scores, normalize)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        k = min(top_k, len(ranked))
        top = ranked[:k]
        order = [i for i, _ in top]
        top_scores = [float(s) for _, s in top] if return_scores else []
        if debug:
            print("\n[Reranker Debug] ranked (text):")
            for r, (i, s) in enumerate(top, 1):
                prev = (docs[i] or "").replace("\n", " ")[:100]
                print(f"  #{r:02d} score={s:.4f} | {prev}")
        return order, top_scores

    # -----------------------------
    # 重排 API（Document/str 混合）
    # -----------------------------
    def rerank_docs(
        self,
        query: str,
        docs: Sequence[TextLike],
        top_k: int = 3,
        return_scores: bool = True,
        *,
        normalize: Optional[str] = None,  # None | "minmax" | "softmax"
        prefer_base_bias: float = 0.0,  # >0 对原始文档加分，<0 惩罚；基于 metadata['aug_type'] != 'doc2query'
        debug: bool = False,
    ):
        """
        支持 LangChain Document 或纯文本的高层接口。
        返回：
          - return_scores=True  -> (ranked_docs, scores)
          - return_scores=False -> ranked_docs
        """
        if not docs:
            return [] if not return_scores else ([], [])

        texts = [self._to_text(d) for d in docs]
        pairs = [(query, t) for t in texts]
        scores = self._batched_compute_score(pairs)

        # 对“原始文档”加权（aug_type != 'doc2query' 视作 base）
        if abs(prefer_base_bias) > 0:
            for i, d in enumerate(docs):
                aug_type = self._get_aug_type(d)
                if aug_type != "doc2query":
                    scores[i] += prefer_base_bias

        # 归一化（可选）
        scores = self._normalize(scores, normalize)

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        k = min(top_k, len(order))
        order = order[:k]
        ranked_docs = [docs[i] for i in order]
        ranked_scores = [float(scores[i]) for i in order]

        if debug:
            print("\n[Reranker Debug] ranked (docs):")
            for r, i in enumerate(order, 1):
                prev = texts[i].replace("\n", " ")[:100]
                aug = self._get_aug_type(docs[i])
                print(f"  #{r:02d} score={scores[i]:.4f} | aug_type={aug} | {prev}")

        if return_scores:
            return ranked_docs, ranked_scores
        return ranked_docs

    # -----------------------------
    # 归一化
    # -----------------------------
    @staticmethod
    def _normalize(scores: Sequence[float], mode: Optional[str]) -> List[float]:
        if not mode:
            return list(scores)
        if mode == "minmax":
            vmin, vmax = min(scores), max(scores)
            if abs(vmax - vmin) < 1e-12:
                return [1.0 for _ in scores]
            return [(v - vmin) / (vmax - vmin + 1e-12) for v in scores]
        if mode == "softmax":
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            z = sum(exps) + 1e-12
            return [e / z for e in exps]
        # 未知模式：原样返回
        return list(scores)
