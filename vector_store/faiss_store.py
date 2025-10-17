# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional
import os
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from .base import VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self, path: str):
        self.path = path
        self._vs: Optional[FAISS] = None

    # --- lifecycle ---
    def load_local(self, path: Optional[str], embeddings: Embeddings) -> None:
        p = path or self.path
        self._vs = FAISS.load_local(p, embeddings, allow_dangerous_deserialization=True)

    def save_local(self, path: Optional[str] = None) -> None:
        if self._vs is None:
            raise RuntimeError("Vector store not initialized.")
        p = path or self.path
        os.makedirs(p, exist_ok=True)
        self._vs.save_local(p)

    # --- build / add ---
    def build_from_documents(self, docs: List[Document], embeddings: Embeddings) -> None:
        self._vs = FAISS.from_documents(docs, embeddings)

    def add_documents(self, docs: List[Document]) -> None:
        if self._vs is None:
            raise RuntimeError("Vector store not initialized.")
        self._vs.add_documents(docs)

    # --- query ---
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self._vs is None:
            raise RuntimeError("Vector store not initialized.")
        return self._vs.similarity_search(query, k=k)

    def count(self) -> int:
        if self._vs is None:
            return 0
        return self._vs.index.ntotal
