# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Optional, Sequence
import os
import pickle

from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings

from vector_stores.base import VectorStore


class KnowledgeBaseBuilder:
    """
    负责：
    1) 从 PDF 提取文本 + 页码；
    2) 文本切分并构建向量库（任意 VectorStore 实现）；
    3) 保存/加载向量库的本地元数据与页码信息。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: Embeddings,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[Sequence[str]] = None,
    ):
        self.vs = vector_store
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]

        self.page_info = {}  # {chunk_text: page_number}

    # ---------- Step 1: PDF -> (text, page_numbers) ----------
    @staticmethod
    def extract_text_with_page_numbers(pdf: PdfReader) -> Tuple[str, List[int]]:
        text = ""
        page_numbers: List[int] = []
        for page_number, page in enumerate(pdf.pages, start=1):
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
                page_numbers.extend([page_number] * len(extracted_text.split("\n")))
        return text, page_numbers

    # ---------- Step 2: split + build vector store ----------
    def process_text_with_splitter(
        self,
        text: str,
        page_numbers: List[int],
        save_path: Optional[str] = None,
    ) -> VectorStore:
        # 2.1 split
        splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_text(text)
        print(f"文本被分割成 {len(chunks)} 个块。") # 文本被分割成 5 个块

        # 2.2 构造 LangChain Documents（保留原内容到 metadata 以便回查页码）
        docs: List[Document] = [Document(page_content=c) for c in chunks]

        # 2.3 建库
        self.vs.build_from_documents(docs, self.embeddings)
        print("已从文本块创建知识库。")

        # 2.4 为每个 chunk 计算来源页码（与原实现一致的近似映射）
        lines = text.split("\n")
        page_info = {}
        for chunk in chunks:
            start_idx = text.find(chunk[:100])
            if start_idx == -1:
                for i, line in enumerate(lines):
                    if chunk.startswith(line[:min(50, len(line))]):
                        start_idx = i
                        break
                if start_idx == -1:
                    for i, line in enumerate(lines):
                        if line and line in chunk:
                            start_idx = text.find(line)
                            break

            if start_idx != -1:
                line_count = text[:start_idx].count("\n")
                if line_count < len(page_numbers):
                    page_info[chunk] = page_numbers[line_count]
                else:
                    page_info[chunk] = page_numbers[-1] if page_numbers else 1
            else:
                page_info[chunk] = -1

        self.page_info = page_info

        # 2.5 可选：保存向量库元数据 & 页码表
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.vs.save_local(save_path)
            with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
                pickle.dump(self.page_info, f)
            print(f"已保存：{save_path}（向量库元信息 + 页码信息）")

        return self.vs

    # ---------- Step 3: 从本地元数据恢复 ----------
    def load_knowledge_base(self, load_path: str, embeddings: Optional[Embeddings] = None) -> VectorStore:
        embs = embeddings or self.embeddings
        self.vs.load_local(load_path, embs)

        page_info_path = os.path.join(load_path, "page_info.pkl")
        if os.path.exists(page_info_path):
            with open(page_info_path, "rb") as f:
                self.page_info = pickle.load(f)
            print("页码信息已加载。")
        else:
            print("警告: 未找到页码信息文件。")
            self.page_info = {}
        return self.vs
