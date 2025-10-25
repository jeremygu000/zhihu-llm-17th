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
        *,
        reembed: bool = True,  # ← 新增：是否重新 embed & 写入
    ) -> VectorStore:
        splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_text(text)
        print(f"文本被分割成 {len(chunks)} 个块。")

        if reembed:
            # 正常建库写入
            docs: List[Document] = [Document(page_content=c) for c in chunks]
            self.vs.build_from_documents(docs, self.embeddings)
            print("已从文本块创建知识库。")

            # 计算并保存页码映射
            self.page_info = self._build_page_info(text, page_numbers, chunks)

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                self.vs.save_local(save_path)
                with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
                    pickle.dump(self.page_info, f)
                print(f"已保存：{save_path}（向量库元信息 + 页码信息）")
        else:
            # 只挂载现有集合（不写入）
            # 若你之前存过 meta（save_local）与 page_info，则这里尽量载入
            if save_path and os.path.exists(
                os.path.join(save_path, "milvus_meta.json")
            ):
                self.vs.load_local(save_path, self.embeddings)
                print("已从本地 meta 挂载 Milvus 集合。")
                pkl = os.path.join(save_path, "page_info.pkl")
                if os.path.exists(pkl):
                    with open(pkl, "rb") as f:
                        self.page_info = pickle.load(f)
                    print("页码信息已加载。")
                else:
                    print("提示：未找到本地页码信息 page_info.pkl。")
            else:
                # 既没有 meta，也要能工作：直接按集合名挂载
                self.vs.open_existing(self.embeddings)
                print("已直接挂载现有 Milvus 集合（未加载本地 meta/page_info）。")
                self.page_info = {}

        return self.vs

    def attach_existing_collection(self, meta_dir: Optional[str] = None) -> VectorStore:
        """
        仅挂载已有集合用于检索；如提供 meta_dir 则加载 page_info。
        """
        if meta_dir and os.path.exists(os.path.join(meta_dir, "milvus_meta.json")):
            self.vs.load_local(meta_dir, self.embeddings)
            pkl = os.path.join(meta_dir, "page_info.pkl")
            if os.path.exists(pkl):
                with open(pkl, "rb") as f:
                    self.page_info = pickle.load(f)
                print("页码信息已加载。")
            else:
                self.page_info = {}
        else:
            self.vs.open_existing(self.embeddings)
            self.page_info = {}
        return self.vs

    # 把页码映射逻辑抽成私有方法，便于复用/测试
    def _build_page_info(
        self, text: str, page_numbers: List[int], chunks: List[str]
    ) -> dict:
        lines = text.split("\n")
        page_info = {}
        for chunk in chunks:
            start_idx = text.find(chunk[:100])
            if start_idx == -1:
                for i, line in enumerate(lines):
                    if chunk.startswith(line[: min(50, len(line))]):
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
        return page_info

    # ---------- Step 3: 从本地元数据恢复 ----------
    def load_knowledge_base(
        self, load_path: str, embeddings: Optional[Embeddings] = None
    ) -> VectorStore:
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
