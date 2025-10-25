from rerank_service import BgeRerankService


reranker = BgeRerankService("BAAI/bge-reranker-base", use_fp16=True)

# 文本输入
scores = reranker._model.compute_score(
    [
        ["what is panda?", "The giant panda is a bear species endemic to China."],
        ["what is panda?", "hi"],
    ]
)
print("scores", scores)

# 或封装后的接口
order, scores = reranker.rerank(
    query="what is panda?",
    docs=["hi", "The giant panda is a bear species endemic to China."],
    top_k=2,
)
print("order", order)
print("scores", scores)
