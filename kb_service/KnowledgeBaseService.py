# kb_service/service.py
from __future__ import annotations
from typing import List, Dict, Any, Sequence, Tuple, Optional
import uuid
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from vector_store.base import VectorStore
from meta_store.base import MetaStore


class KnowledgeBaseService:
    def __init__(self, vector_store: VectorStore, meta_store: MetaStore, embeddings: Embeddings):
        self.vs = vector_store
        self.ms = meta_store
        self.embeddings = embeddings

    # ---- 已有：从纯文本建库 ----
    def build_from_text(
        self,
        text: str,
        global_meta: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Sequence[str] = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", ""),
        persist: bool = True,
    ) -> int:
        splitter = RecursiveCharacterTextSplitter(
            separators=list(separators),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_text(text)
        docs: List[Document] = []

        for i, ch in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            meta = dict(global_meta or {})
            meta.update({"doc_id": doc_id, "chunk_index": i})
            self.ms.set_meta(doc_id, meta)
            docs.append(Document(page_content=ch, metadata={"doc_id": doc_id}))

        self.vs.build_from_documents(docs, self.embeddings)
        if persist:
            self.vs.save_local(None)
        return len(docs)

    # ---- 新增：直接从 PDF 建库（附页码）----
    def build_from_pdf(
        self,
        pdf_path: str,
        global_meta: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Sequence[str] = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", ""),
        persist: bool = True,
        text_mode: str = "text",   # 也可用 "html"、"rawdict"、"ocr"（扫描件）
    ) -> int:
        """
        读取 PDF 每一页文本，按页分块并写入向量库；页码写入 Redis 元数据。
        返回写入的 chunk 数量。
        """
        doc = fitz.open(pdf_path)
        splitter = RecursiveCharacterTextSplitter(
            separators=list(separators),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        docs: List[Document] = []
        chunk_idx = 0

        for i, page in enumerate(doc):
            page_num = i + 1
            # 提取文本；扫描件可改成 page.get_text("ocr")
            page_text = page.get_text(text_mode) or ""
            if not page_text.strip():
                continue

            page_chunks = splitter.split_text(page_text)
            for ch in page_chunks:
                doc_id = str(uuid.uuid4())
                # 元数据写 Redis（包含页码）
                meta = dict(global_meta or {})
                meta.update({
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "page": page_num,
                    "source": pdf_path,
                })
                self.ms.set_meta(doc_id, meta)
                # 向量库仅存 doc_id
                docs.append(Document(page_content=ch, metadata={"doc_id": doc_id}))
                chunk_idx += 1

        if not docs:
            return 0

        self.vs.build_from_documents(docs, self.embeddings)
        if persist:
            self.vs.save_local(None)
        return len(docs)

    # ---- 查询合并元数据（已存在）----
    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, Dict[str, Any]]]:
        hits = self.vs.similarity_search(query, k=k)
        ids = [h.metadata.get("doc_id", "") for h in hits]
        meta_map = self.ms.mget_meta(ids)
        return [(h, meta_map.get(h.metadata.get("doc_id", ""), {})) for h in hits]
