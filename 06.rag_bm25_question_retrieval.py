# -*- coding: utf-8 -*-
"""
knowledge_base_dual_bm25_optimizer.py
-------------------------------------

A module for building and evaluating dual BM25 indexes:
- One based on document content (content_bm25)
- One based on AI-generated questions (question_bm25)

This design allows comparing retrieval effectiveness between:
1. Direct content matching
2. Question-level semantic matching (generated via LLM)

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

# ============================================================
# 🔧 Import Dependencies
# ============================================================
import os
import json
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from utils.llm_utils import get_completion, preprocess_json_response
from utils.text_preprocess import preprocess_text, tokenize_text


# ============================================================
# 🧱 Class: KnowledgeBaseDualBM25Optimizer
# ============================================================
class KnowledgeBaseDualBM25Optimizer:
    """
    A class that constructs and evaluates dual BM25 indexes:
    - content_bm25: based on original text
    - question_bm25: based on AI-generated paraphrased questions
    """

    # --------------------------------------------------------
    # 🧩 Initialization
    # --------------------------------------------------------
    def __init__(self, tokenizer: str = "jieba", model: str = "qwen-turbo-latest"):
        """
        Args:
            tokenizer (str): Tokenizer to use ('jieba' / 'spacy' / 'split')
            model (str): LLM model for question generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.docs: List[Dict[str, Any]] = []
        self.content_bm25 = None
        self.question_bm25 = None
        self.questions_by_doc: List[List[str]] = []

    # --------------------------------------------------------
    # 🧠 Step 1: Generate Questions via LLM
    # --------------------------------------------------------
    def generate_questions_for_doc(self, content: str, n: int = 3) -> List[str]:
        """
        Use the LLM to generate representative questions for a document.

        Args:
            content (str): The original document content.
            n (int): Number of questions to generate.

        Returns:
            List[str]: Generated question list.
        """
        prompt = f"""
        你是一个知识问答专家。请基于以下内容生成 {n} 个可能的提问方式，
        这些问题可以帮助用户在搜索时更容易检索到该内容。

        内容：
        {content}

        请输出JSON数组，例如：
        ["问题1", "问题2", "问题3"]
        """

        response = get_completion(prompt, self.model)
        response = preprocess_json_response(response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []

    # --------------------------------------------------------
    # 📚 Step 2: Build Dual BM25 Indexes
    #    "question_bm25" and "content_tokens"
    # --------------------------------------------------------
    def build_indexes(self, docs: List[str]):
        """
        Build BM25 indexes for both document content and generated questions.

        Args:
            docs (List[str]): List of original documents.
        """
        print("\n--- 构建双BM25索引 ---")
        self.docs = docs

        # Clean and tokenize original content
        content_tokens = [
            tokenize_text(preprocess_text(doc), self.tokenizer) for doc in docs
        ]

        # Build content-based BM25
        print("📘 正在构建 content_bm25 ...")
        self.content_bm25 = BM25Okapi(content_tokens)

        # Generate questions per document
        print("🤖 正在生成问题并构建 question_bm25 ...")
        all_questions = []
        self.questions_by_doc = []
        for i, doc in enumerate(docs):
            questions = self.generate_questions_for_doc(doc)
            self.questions_by_doc.append(questions)
            for q in questions:
                all_questions.append((i, q))

        # Build question-based BM25
        question_texts = [q for _, q in all_questions]
        question_tokens = [
            tokenize_text(preprocess_text(q), self.tokenizer) for q in question_texts
        ]
        self.question_bm25 = BM25Okapi(question_tokens)

        print("✅ 双BM25索引构建完成。")

    # --------------------------------------------------------
    # 🔍 Step 3: Perform Search
    # --------------------------------------------------------
    def search_similar_chunks(
        self, query: str, search_type: str = "content", top_k: int = 3
    ) -> List[str]:
        """
        Search top-k matching documents using BM25.

        Args:
            query (str): The input query.
            search_type (str): "content" or "question"
            top_k (int): Number of results to return.

        Returns:
            List[str]: Top matching text snippets.
        """
        query_tokens = tokenize_text(preprocess_text(query), self.tokenizer)

        if search_type == "content":
            if not self.content_bm25:
                raise RuntimeError("content_bm25 not built.")
            scores = self.content_bm25.get_scores(query_tokens)
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]
            return [self.docs[i] for i in top_indices]

        elif search_type == "question":
            if not self.question_bm25:
                raise RuntimeError("question_bm25 not built.")
            scores = self.question_bm25.get_scores(query_tokens)
            # Match question back to document via mapping
            top_q_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_k]
            matched_docs = []
            all_questions_flat = [q for group in self.questions_by_doc for q in group]
            for q_idx in top_q_indices:
                q_text = all_questions_flat[q_idx]
                # Find which document this question belongs to
                for doc_idx, q_list in enumerate(self.questions_by_doc):
                    if q_text in q_list:
                        matched_docs.append(self.docs[doc_idx])
                        break
            return matched_docs

        else:
            raise ValueError("search_type must be 'content' or 'question'")

    # --------------------------------------------------------
    # 🧮 Step 4: Evaluate Retrieval Performance
    # --------------------------------------------------------
    def evaluate_retrieval_methods(self, test_queries: List[Dict[str, str]]):
        """
        Evaluate and compare retrieval performance between content-based
        and question-based BM25 methods.

        Args:
            test_queries (List[Dict[str, str]]):
                Each dict has:
                    - 'query': user query
                    - 'expected_doc': expected correct document text (or substring)
        """
        print("\n--- 评估检索效果 ---")
        total = len(test_queries)
        content_correct = 0
        question_correct = 0

        for i, item in enumerate(test_queries, 1):
            query = item["query"]
            expected = item["expected_doc"]
            print(f"\n[{i}/{total}] Query: {query}")

            # Run content-based search
            c_results = self.search_similar_chunks(query, "content", 3)
            q_results = self.search_similar_chunks(query, "question", 3)

            # Evaluate matches
            if any(expected in doc for doc in c_results):
                content_correct += 1
            if any(expected in doc for doc in q_results):
                question_correct += 1

            print(f"📘 content命中: {any(expected in doc for doc in c_results)}")
            print(f"❓ question命中: {any(expected in doc for doc in q_results)}")

        print("\n=== 评估结果 ===")
        print(
            f"BM25原文检索准确率: {content_correct}/{total} = {content_correct / total:.2%}"
        )
        print(
            f"BM25问题检索准确率: {question_correct}/{total} = {question_correct / total:.2%}"
        )
        print(f"问题检索改进查询数: {max(question_correct - content_correct, 0)}")


# ============================================================
# 🚀 Demonstration (Executed when run directly)
# ============================================================
def main():
    """Demonstration for dual BM25 optimization."""
    print("=== 双BM25知识库优化示例 ===")

    # ----------------------------------------
    # 🗂️ 示例文档
    # ----------------------------------------
    docs = [
        "上海迪士尼乐园平日成人票价为399元，周末为499元。",
        "迪士尼乐园每天开放时间为上午8点至晚上8点。",
        "从浦东机场乘地铁2号线换乘11号线即可到达迪士尼站。",
    ]

    # ----------------------------------------
    # 🧱 构建双索引
    # ----------------------------------------
    optimizer = KnowledgeBaseDualBM25Optimizer(tokenizer="jieba")
    optimizer.build_indexes(docs)

    # ----------------------------------------
    # 🔍 示例查询
    # ----------------------------------------
    test_queries = [
        {"query": "上海迪士尼门票多少钱", "expected_doc": "平日成人票价为399元"},
        {"query": "迪士尼几点开门", "expected_doc": "每天开放时间为上午8点"},
        {"query": "浦东机场怎么去迪士尼", "expected_doc": "乘地铁2号线换乘11号线"},
    ]

    optimizer.evaluate_retrieval_methods(test_queries)


# ============================================================
# 🏁 Entry Point
# ============================================================
if __name__ == "__main__":
    main()
