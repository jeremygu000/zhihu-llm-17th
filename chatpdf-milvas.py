# -*- coding: utf-8 -*-
import os
from pathlib import Path

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from ingestpdf_milvas import ingest_pdf
from rerank_service import BgeRerankService
from vector_stores.milvus_store import MilvusVectorStore
from utils.knowledge_builder import KnowledgeBaseBuilder
from FlagEmbedding import FlagReranker

# ---------- 基础配置 ----------
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "spdb_xian_rm_assessment_policy")
SAVE_DIR = os.getenv("SAVE_DIR", "./vector_db_milvus_meta")
# 如果本地元数据不存在，需要用哪个 PDF 来“自动初始化导入”
PDF_PATH_IF_INIT = os.getenv("PDF_PATH", "./policy.pdf")


def ensure_kb_and_attach():
    """
    如果 save_dir 存在 meta，则挂载；否则自动进行初始导入后再挂载。
    返回 (kb, milvus_vs)
    """
    save_dir = Path(SAVE_DIR).resolve()
    meta_file = save_dir / "milvus_meta.json"

    # 1) embeddings & 向量库句柄
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    milvus_vs = MilvusVectorStore(
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )
    kb = KnowledgeBaseBuilder(milvus_vs, embeddings)

    # 2) 存在即挂载；否则自动初始化导入
    if meta_file.exists():
        print(f"[Query] 发现本地元数据：{meta_file}，直接挂载。")
        kb.attach_existing_collection(meta_dir=str(save_dir))
    else:
        print(f"[Query] 未发现本地元数据：{meta_file}。即将自动执行初始导入...")
        ingest_pdf(
            pdf_path=PDF_PATH_IF_INIT,
            save_dir=str(save_dir),
            collection_name=COLLECTION_NAME,
            recreate=True,  # 初始导入建议 True
        )
        # 导入后再挂载
        kb.attach_existing_collection(meta_dir=str(save_dir))

    return kb, milvus_vs


def run_query(query: str, top_k: int = 3):
    # 1) 确保已挂载（必要时自动初始化导入）
    kb, milvus_vs = ensure_kb_and_attach()

    # 2) 构建 QA 链
    llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)
    prompt = ChatPromptTemplate.from_template(
        """
    你是一位中文文档问答助手。仅依据提供的文档上下文回答问题；
    若上下文没有答案，请明确说“未在文档中找到明确答案”。

    <context>
    {context}
    </context>

    问题：{question}
    请用简洁、准确、可引用原文措辞的中文作答。
    """.strip()
    )
    qa_chain = create_stuff_documents_chain(llm, prompt)

    # 3) 检索 & 调用
    docs = milvus_vs.similarity_search(query, k=top_k)
    print(f"检索结果数量: {len(docs)}")
    for i, d in enumerate(docs, 1):
        print(f"Top{i} 内容片段:\n{d.page_content[:200]}\n")

    reranker = BgeRerankService(model_name="BAAI/bge-reranker-base", use_fp16=True)
    ranked_docs, scores = reranker.rerank_docs(query, docs=docs, top_k=1)

    for i, (doc, score) in enumerate(zip(ranked_docs, scores), 1):
        print(f"Top{i} | score={score:.4f}")
        print(doc.page_content[:100])
        print("-" * 50)

    answer = qa_chain.invoke({"context": docs, "question": query})

    print("=== 回答 ===")
    print(answer)

    # 4) 打印来源页码
    print("\n=== 参考来源页码 ===")
    seen = set()
    for d in docs:
        key = (getattr(d, "page_content", "") or "").strip()
        page = getattr(kb, "page_info", {}).get(key, "未知")
        if page not in seen:
            seen.add(page)
            print(f"- 页码：{page}")


if __name__ == "__main__":
    # 示例：直接在这里改 query 即可
    run_query(query="客户经理被投诉了，投诉一次扣多少分", top_k=2)
