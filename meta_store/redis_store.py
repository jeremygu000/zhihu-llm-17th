# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Sequence, List
import json
import redis
from .base import MetaStore


class RedisMetaStore(MetaStore):
    """
    Redis 元数据存储（Hash + Set）
    meta:{doc_id}   -> Hash 字段：meta 序列化为 str
    set:index:{ns}  -> doc_id 集合（便于清理/巡检）
    """
    def __init__(self, url: str, namespace: str = "kb"):
        self.r = redis.from_url(url, decode_responses=True)
        self.ns = namespace

    def _key(self, doc_id: str) -> str:
        return f"meta:{doc_id}"

    def set_meta(self, doc_id: str, meta: Dict[str, Any]) -> None:
        flat = {k: json.dumps(v, ensure_ascii=False) for k, v in meta.items()}
        self.r.hset(self._key(doc_id), mapping=flat)
        self.r.sadd(f"set:index:{self.ns}", doc_id)

    def get_meta(self, doc_id: str) -> Dict[str, Any]:
        raw = self.r.hgetall(self._key(doc_id))
        return {k: json.loads(v) for k, v in raw.items()}

    def mget_meta(self, doc_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        pipe = self.r.pipeline()
        for did in doc_ids:
            pipe.hgetall(self._key(did))
        res = pipe.execute()
        out: Dict[str, Dict[str, Any]] = {}
        for did, raw in zip(doc_ids, res):
            out[did] = {k: json.loads(v) for k, v in raw.items()}
        return out

    def delete(self, doc_id: str) -> None:
        self.r.delete(self._key(doc_id))
        self.r.srem(f"set:index:{self.ns}", doc_id)

    def all_ids(self) -> List[str]:
        return list(self.r.smembers(f"set:index:{self.ns}"))

    def purge_namespace(self) -> None:
        ids = self.all_ids()
        pipe = self.r.pipeline()
        for did in ids:
            pipe.delete(self._key(did))
        pipe.delete(f"set:index:{self.ns}")
        pipe.execute()
