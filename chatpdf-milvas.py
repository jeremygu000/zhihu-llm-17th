# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyPDF2 import PdfReader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

from vector_stores.milvus_store import MilvusVectorStore
from utils.knowledge_builder import KnowledgeBaseBuilder

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# 1) 准备 Embeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

# 2) 准备 Milvus 向量库（连到你的 docker compose）
milvus_vs = MilvusVectorStore(
    collection_name="spdb_xian_rm_assessment_policy",
    connection_args={"host": "localhost", "port": "19530"},
    index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}},
    search_params={"metric_type": "L2", "params": {"nprobe": 16}},
    recreate_collection=True,   # 首次构建建议 True；后续可以 False
)

# 3) 构建工具类
kb = KnowledgeBaseBuilder(vector_store=milvus_vs, embeddings=embeddings)

# 4) 提取 PDF 文本 + 页码
pdf_reader = PdfReader("./policy.pdf")
text, page_numbers = KnowledgeBaseBuilder.extract_text_with_page_numbers(pdf_reader)
print(f"提取的文本长度: {len(text)} 个字符。") # 提取的文本长度: 3881 个字符。

# 5) 切分并写入 Milvus；顺便把元数据与页码信息保存到本地目录（不包含向量数据本体）
save_dir = "./vector_db_milvus_meta"
knowledge_base = kb.process_text_with_splitter(text, page_numbers, save_path=save_dir)

# （可选）稍后或在其他进程中恢复
# kb.load_knowledge_base(save_dir)

# 6) 检索 + QA
llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)
query = "客户经理被投诉了，投诉一次扣多少分"

docs = knowledge_base.similarity_search(query, k=3)

prompt = ChatPromptTemplate.from_template("""
你是一位中文文档问答助手。仅依据提供的文档上下文回答问题；
若上下文没有答案，请明确说“未在文档中找到明确答案”。

<context>
{context}
</context>

问题：{question}
请用简洁、准确、可引用原文措辞的中文作答。
""".strip())

# 3) 构建 Stuff 链（等价于老版 chain_type="stuff"）
qa_chain = create_stuff_documents_chain(llm, prompt)


# 4) 检索 & 调用
query = "客户经理被投诉了，投诉一次扣多少分"
top_k = 2
docs = knowledge_base.similarity_search(query, k=top_k)

resp = qa_chain.invoke({
    "context": docs,     # 直接传入 List[Document]
    "question": query,
})

print("=== 回答 ===")
print(resp)  # create_stuff_documents_chain 返回字符串

# 5) 打印来源页码（基于你在 KnowledgeBaseBuilder 里维护的 kb.page_info）
print("\n=== 参考来源页码 ===")
seen = set()
for d in docs:
    text_key = (getattr(d, "page_content", "") or "").strip()
    page = getattr(kb, "page_info", {}).get(text_key, "未知")
    if page not in seen:
        seen.add(page)
        print(f"- 页码：{page}")
