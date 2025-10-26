# -*- coding: utf-8 -*-
from typing import List, Any, Optional, Callable, Tuple
from rank_bm25 import BM25Okapi
import os, pickle

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

class BM25Service:
    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        self.tokenizer = tokenizer or (lambda t: t.split())
        self._bm25 = None
        self._docs: List[Any] = []
        self._tokens: List[List[str]] = []

    def build_from_docs(self, docs: List[Any]) -> None:
        self._docs = docs
        self._tokens = [self.tokenizer(getattr(d, "page_content", "")) for d in docs]
        self._bm25 = BM25Okapi(self._tokens)

    def topk_with_scores(self, query: str, k: int) -> List[Tuple[Any, float]]:
        q_tokens = self.tokenizer(query)
        print(f"🔍 查询分词: {q_tokens}")
        scores = self._bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._docs[i], float(scores[i])) for i in order]


# ----------------------------------------------------------------------
# Main 示例：分词 + 召回分数清晰展示
# ----------------------------------------------------------------------
if __name__ == "__main__":
    class Document:
        def __init__(self, page_content: str):
            self.page_content = page_content

    docs = [
        Document("深度学习模型训练的优化技巧包括使用 AdamW 替代 SGD。"),
        Document("2023年诺贝尔物理学奖授予三位科学家，以表彰他们在量子纠缠领域的研究成果。"),
        Document("人工智能在医疗领域的应用包括医学影像分析与电子病历数据挖掘。"),
        Document("混合索引召回结合 BM25 与向量检索，提升 RAG 系统召回的准确性。"),
    ]

    def run_example(title: str, bm25: BM25Service, query: str):
        print("\n" + "=" * 80)
        print(f"🔹 {title}")
        print("=" * 80)

        # 文档分词展示
        for i, d in enumerate(docs):
            tokens = bm25.tokenizer(d.page_content)
            print(f"Doc {i+1} 分词: {tokens}")

        # 构建索引并召回
        bm25.build_from_docs(docs)
        results = bm25.topk_with_scores(query, 3)

        print("\n📖 Top-3 召回结果:")
        for rank, (doc, score) in enumerate(results, 1):
            preview = doc.page_content[:60]
            print(f"{rank:>2}. score={score:.4f} | {preview}")

    # 1️⃣ 默认空白分词
    bm25_default = BM25Service()
    run_example("Example 1: 默认空白分词", bm25_default, "量子 物理学奖")

    # 2️⃣ jieba 中文分词
    try:
        import jieba
        bm25_jieba = BM25Service(tokenizer=lambda t: list(jieba.cut(t)))
        run_example("Example 2: jieba 中文分词", bm25_jieba, "量子纠缠 研究成果")
    except ImportError:
        print("⚠️ 未安装 jieba，可 pip install jieba")

    # 3️⃣ spaCy 英文分词
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        def spacy_tokenizer(text: str):
            return [tok.text for tok in nlp(text) if not tok.is_punct and not tok.is_space]
        bm25_spacy = BM25Service(tokenizer=spacy_tokenizer)
        run_example("Example 3: spaCy 英文分词", bm25_spacy, "medical image analysis")
    except Exception as e:
        print("⚠️ spaCy 示例跳过（需安装并下载 en_core_web_sm）：", e)
