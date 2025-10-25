# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Dict, Any
import json
import os

from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Milvus as LCMilvus

try:
    from pymilvus import connections, utility, Collection
except Exception:
    connections = None
    utility = None
    Collection = None

from .base import VectorStore


class MilvusVectorStore(VectorStore):
    """
    Milvus 向量库实现，遵循 VectorStore 协议。
    - 数据持久化在 Milvus 服务器端；
    - save_local/load_local 仅保存/恢复连接与集合的元数据；
    - count() 依赖 pymilvus；
    """

    META_FILENAME = "milvus_meta.json"

    def __init__(
        self,
        collection_name: str,
        connection_args: Optional[Dict[str, Any]] = None,
        *,
        index_params: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        recreate_collection: bool = False,
    ):
        self.collection_name = collection_name
        self.connection_args = connection_args or {
            "host": "milvus-standalone",
            "port": "19530",
        }
        self.index_params = index_params or {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }
        self.search_params = search_params or {
            "metric_type": self.index_params.get("metric_type", "L2"),
            "params": {"nprobe": 16},
        }
        self.recreate_collection = recreate_collection

        self._vs: Optional[LCMilvus] = None
        self._embeddings: Optional[Embeddings] = None

    # ---------- internal ----------
    def _ensure_connection(self) -> None:
        if connections is None:
            return
        if "uri" in self.connection_args:
            if not connections.has_connection("default"):
                connections.connect("default", **self.connection_args)
        else:
            host = self.connection_args.get("host", "127.0.0.1")
            port = self.connection_args.get("port", "19530")
            if not connections.has_connection("default"):
                connections.connect("default", host=host, port=port)

    def _maybe_drop_collection(self) -> None:
        if not self.recreate_collection or utility is None:
            return
        self._ensure_connection()
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

    # ---------- lifecycle ----------
    def load_local(self, path: Optional[str], embeddings: Embeddings) -> None:
        p = path or "."
        meta_path = os.path.join(p, self.META_FILENAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Milvus meta file not found: {meta_path}")

        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        self.collection_name = meta["collection_name"]
        self.connection_args = meta["connection_args"]
        self.index_params = meta.get("index_params", self.index_params)
        self.search_params = meta.get("search_params", self.search_params)
        self._embeddings = embeddings

        self._vs = LCMilvus(
            embedding_function=self._embeddings,
            collection_name=self.collection_name,
            connection_args=self.connection_args,
            index_params=self.index_params,
            search_params=self.search_params,
        )

    def save_local(self, path: Optional[str] = None) -> None:
        p = path or "."
        os.makedirs(p, exist_ok=True)
        meta_path = os.path.join(p, self.META_FILENAME)
        meta = {
            "collection_name": self.collection_name,
            "connection_args": self.connection_args,
            "index_params": self.index_params,
            "search_params": self.search_params,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---------- build / add ----------
    def build_from_documents(
        self, docs: List[Document], embeddings: Embeddings
    ) -> None:
        self._embeddings = embeddings
        self._maybe_drop_collection()
        self._vs = LCMilvus.from_documents(
            docs,
            embedding=self._embeddings,
            collection_name=self.collection_name,
            connection_args=self.connection_args,
            index_params=self.index_params,
            search_params=self.search_params,
        )

    def add_documents(self, docs: List[Document]) -> None:
        if self._vs is None:
            if self._embeddings is None:
                raise RuntimeError(
                    "Vector store not initialized. Call build_from_documents() or load_local() first."
                )
            self._vs = LCMilvus(
                embedding_function=self._embeddings,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                index_params=self.index_params,
                search_params=self.search_params,
            )
        self._vs.add_documents(docs)

    # ---------- query ----------
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self._vs is None:
            if self._embeddings is None:
                raise RuntimeError(
                    "Vector store not initialized. Call build_from_documents() or load_local() first."
                )
            self._vs = LCMilvus(
                embedding_function=self._embeddings,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                index_params=self.index_params,
                search_params=self.search_params,
            )
        return self._vs.similarity_search(query, k=k)

    def count(self) -> int:
        if Collection is None:
            return -1
        self._ensure_connection()
        if not utility.has_collection(self.collection_name):
            return 0
        return int(Collection(self.collection_name).num_entities)

    def exists(self) -> bool:
        """集合是否已存在（需要 pymilvus）。未知返回 False。"""
        if utility is None:
            return False
        self._ensure_connection()
        return bool(utility.has_collection(self.collection_name))

    def open_existing(self, embeddings: Embeddings) -> None:
        """
        挂载已有集合：不创建、不索引、不写入，仅用于检索。
        """
        self._embeddings = embeddings
        # 这里不会触发 from_documents；只创建句柄
        self._vs = LCMilvus(
            embedding_function=self._embeddings,
            collection_name=self.collection_name,
            connection_args=self.connection_args,
            index_params=self.index_params,  # 仅供查询使用
            search_params=self.search_params,
        )
