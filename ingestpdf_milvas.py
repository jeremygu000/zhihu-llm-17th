# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Optional

from PyPDF2 import PdfReader
from langchain_community.embeddings import DashScopeEmbeddings

from vector_stores.milvus_store import MilvusVectorStore
from utils.knowledge_builder import KnowledgeBaseBuilder

# ---------- 基础配置 ----------
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# Milvus 连接（宿主机直连）
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# 默认集合名 & 目录（可按需修改）
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "spdb_xian_rm_assessment_policy")
DEFAULT_SAVE_DIR = os.getenv("SAVE_DIR", "./vector_db_milvus_meta")
DEFAULT_PDF_PATH = os.getenv("PDF_PATH", "./policy.pdf")


def ingest_pdf(
    pdf_path: str,
    save_dir: str,
    collection_name: str = DEFAULT_COLLECTION,
    *,
    recreate: bool = True,
    index_params: Optional[dict] = None,
    search_params: Optional[dict] = None,
) -> None:
    """
    首次或重新 Ingest：
    1) 提取 PDF 文本 + 页码
    2) 分块 + 嵌入
    3) 写入 Milvus
    4) 保存本地元数据 & 页码映射
    """
    pdf_path = str(Path(pdf_path).resolve())
    save_dir = str(Path(save_dir).resolve())
    print(f"[Ingest] PDF: {pdf_path}")
    print(f"[Ingest] Save dir: {save_dir}")
    print(f"[Ingest] Collection: {collection_name} (recreate={recreate})")

    # 1) Embeddings
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )

    # 2) 向量库
    milvus_vs = MilvusVectorStore(
        collection_name=collection_name,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        # index_params=index_params or {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}},
        # search_params=search_params or {"metric_type": "L2", "params": {"nprobe": 16}},
        index_params=index_params or {"index_type": "FLAT", "metric_type": "L2"},
        search_params=search_params
        or {"metric_type": "L2", "params": {}},  # 无需 nprobe
        recreate_collection=recreate,
    )

    kb = KnowledgeBaseBuilder(vector_store=milvus_vs, embeddings=embeddings)

    # 3) 读取 PDF 并提取文本/页码
    reader = PdfReader(pdf_path)
    text, page_numbers = KnowledgeBaseBuilder.extract_text_with_page_numbers(reader)
    print(f"[Ingest] 提取的文本长度: {len(text)} 字符。")

    # 4) 分块 & 写入 Milvus；保存 meta 与 page_info
    kb.process_text_with_splitter(text, page_numbers, save_path=save_dir, reembed=True)
    print("[Ingest] 导入完成。")


if __name__ == "__main__":
    # 简单 CLI：允许通过环境变量覆盖；或者直接改 DEFAULT_* 常量
    ingest_pdf(
        pdf_path=DEFAULT_PDF_PATH,
        save_dir=DEFAULT_SAVE_DIR,
        collection_name=DEFAULT_COLLECTION,
        recreate=True,  # 首次导入建议 True；后续可 False
    )
