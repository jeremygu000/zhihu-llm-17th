# -*- coding: utf-8 -*-
"""
Bidirectional rewriting for RAG (Query2Doc + Doc2Query)

- Query2Doc: 将短查询扩写成“伪文档”（段落），提升与知识库文档的语义对齐度
- Doc2Query: 为文档生成若干“潜在查询”（问题），提升被命中的概率

依赖：
  - pip install dashscope python-dotenv
环境变量：
  - DASHSCOPE_API_KEY
"""

import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dashscope
from dotenv import load_dotenv

# ---------- 环境初始化 ----------
load_dotenv()
_api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
if not _api_key:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
dashscope.api_key = _api_key


# ---------- 工具函数 ----------
def _truncate(s: str, limit: int) -> str:
    if s is None:
        return ""
    return s if len(s) <= limit else (s[:limit] + " ...[truncated]")


def _json_loose_parse(s: str) -> Any:
    """优先严格 json.loads；失败则尝试截取 JSON 片段；仍失败返回原文"""
    try:
        return json.loads(s)
    except Exception:
        start = min([i for i in [s.find("["), s.find("{")] if i != -1] or [-1])
        end = max(s.rfind("]"), s.rfind("}"))
        if start != -1 and end != -1 and end > start:
            frag = s[start : end + 1]
            try:
                return json.loads(frag)
            except Exception:
                pass
        return s.strip()


# ---------- 统一 LLM 调用 ----------
@dataclass
class LLMConfig:
    model: str = "qwen-turbo-latest"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    seed: Optional[int] = 42
    timeout: int = 30
    retries: int = 3
    backoff_base: float = 0.6
    max_context_chars: int = 4000


def _call_llm(prompt: str, cfg: LLMConfig) -> str:
    """带重试/退避的通用调用"""
    msgs = [{"role": "user", "content": prompt}]
    last_err: Optional[Exception] = None
    for attempt in range(cfg.retries + 1):
        try:
            resp = dashscope.Generation.call(
                model=cfg.model,
                messages=msgs,
                result_format="message",
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_output_tokens=cfg.max_tokens,
                seed=cfg.seed,
                timeout=cfg.timeout,
            )
            out = getattr(resp, "output", None)
            if not out or not out.choices:
                raise RuntimeError(f"dashscope empty output: {resp}")
            return out.choices[0].message.content
        except Exception as e:
            last_err = e
            if attempt >= cfg.retries:
                break
            time.sleep(cfg.backoff_base * (2 ** attempt))
    raise RuntimeError(f"dashscope call failed after retries: {last_err}")


# ---------- 提示词 ----------
Q2D_PROMPT = """你是一个检索增强生成（RAG）的查询扩写专家。
请将用户的简短查询扩写为一段结构化的“伪文档”，用于向量检索匹配。
要求：
- 用中立、客观的百科式语气描述
- 覆盖与查询强相关的关键点、同义表达、常见上下文
- 保持简洁，避免无事实支撑的生成
- 列表用 1. 2. 3. 表达
- 仅输出“正文段落”，不加额外解释

【用户查询】
{query}

【扩写后的伪文档】
"""

D2Q_PROMPT = """你是一个检索增强生成（RAG）的查询生成器。
请基于下面的文档内容，生成 N 条“可能的用户查询”，用于反向索引。
要求：
- 每条查询为自然语言问题，尽量贴近真实用户问法
- 覆盖不同粒度与角度（定义/原因/对比/步骤/限制/注意事项等）
- 严禁引入文档之外的事实
- 输出 JSON 数组，形如 ["问题1", "问题2", ...]，不要额外文本

【文档片段】
{doc}

【N（问题条数）】
{n}
"""

class BidirectionalRewriter:
    """
    双向改写：
      - query2doc(query) -> str  （伪文档）
      - doc2queries(doc, n) -> List[str]  （一组查询问题）
      - rewrite_both(query, docs, n) -> Dict  （组合输出）
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.cfg = config or LLMConfig()

    # --- Query2Doc ---
    def query2doc(self, query: str) -> str:
        prompt = Q2D_PROMPT.format(query=query.strip())
        return _call_llm(prompt, self.cfg)

    # --- Doc2Query ---
    def doc2queries(self, doc: str, n: int = 5) -> List[str]:
        prompt = D2Q_PROMPT.format(
            doc=_truncate(doc, self.cfg.max_context_chars),
            n=max(1, int(n)),
        )
        out = _call_llm(prompt, self.cfg)
        parsed = _json_loose_parse(out)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
        # 兜底：行拆分
        return [s for s in str(parsed).splitlines() if s.strip()]

    # --- 组合：给多个文档分别生成查询，并给 query 扩写 ---
    def rewrite_both(self, query: str, docs: List[str], per_doc_n: int = 5) -> Dict[str, Any]:
        q2d = self.query2doc(query)
        d2q_all: List[Dict[str, Any]] = []
        for i, d in enumerate(docs, 1):
            qs = self.doc2queries(d, n=per_doc_n)
            d2q_all.append({"doc_id": i, "queries": qs})
        return {
            "original_query": query,
            "query2doc": q2d,
            "doc2query": d2q_all,
        }
