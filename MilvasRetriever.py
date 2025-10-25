from typing import Any, List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


class MilvusRetriever(BaseRetriever):
    """兼容 LangChain 的 Milvus Retriever Wrapper"""

    vectorstore: Any
    k: int = 4

    def __init__(self, vectorstore, k=4, **kwargs):
        super().__init__(vectorstore=vectorstore, k=k, **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)
