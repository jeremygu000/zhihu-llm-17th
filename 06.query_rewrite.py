# -*- coding: utf-8 -*-
import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

import dashscope

# === 异常兼容处理 ===
try:
    from dashscope.common.error import AuthenticationError
except ImportError:
    AuthenticationError = Exception  # fallback

# dashscope 在不同版本下异常类名不统一，统一走 Exception 捕获
DASHSCOPE_EXCEPTIONS = (AuthenticationError, Exception)

# -------------------------------
# 环境与全局配置
# -------------------------------
load_dotenv()
DASHSCOPE_API_KEY = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY  # 统一设置一次，后续无需在调用里传参


# -------------------------------
# 常量：Prompt 模板
# -------------------------------
CTX_DEP_PROMPT = """你是一个智能的查询优化助手。请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖于上下文。
如果依赖，请将当前问题改写成一个独立的、包含所有必要上下文信息的完整问题。
如果不依赖，直接返回原问题。

### 对话历史 ###
{conversation}

### 当前问题 ###
{query}

### 改写后的问题 ###"""

CMP_PROMPT = """你是一个查询分析专家。请分析用户的输入和相关的对话上下文，识别出问题中需要进行比较的多个对象。
然后，将原始问题改写成一个更明确、更适合在知识库中检索的对比性查询。

### 对话历史/上下文信息 ###
{context}

### 原始问题 ###
{query}

### 改写后的查询 ###"""

AMB_PROMPT = """你是一个消除语言歧义的专家。请分析用户的当前问题和对话历史，找出问题中“都”“它”“这个”等模糊指代词具体指向的对象。
然后，将这些指代词替换为明确的对象名称，生成一个清晰、无歧义的新问题。

### 对话历史 ###
{conversation}

### 当前问题 ###
{query}

### 改写后的问题 ###"""

MULTI_PROMPT = """你是一个任务分解机器人。请将用户的复杂问题分解成多个独立的、可以单独回答的简单问题。以 JSON 数组格式输出。

### 原始问题 ###
{query}

### 分解后的问题列表 ###
请仅输出 JSON 数组，例如：["问题1", "问题2", "问题3"]"""

RHE_PROMPT = """你是一个沟通理解大师。请分析用户的反问或带有情绪的陈述，识别其背后真实的意图和问题。
然后，将这个反问改写成一个中立、客观、可以直接用于知识库检索的问题。

### 对话历史 ###
{conversation}

### 当前问题 ###
{query}

### 改写后的问题 ###"""

AUTO_PROMPT = """你是一个智能的查询分析专家。请分析用户的查询，识别其属于以下哪种类型：
1. 上下文依赖型 - 包含“还有”“其他”等需要上下文理解的词汇
2. 对比型 - 包含“哪个”“比较”“更”“哪个更好”等比较词汇
3. 模糊指代型 - 包含“它”“他们”“都”“这个”等指代词
4. 多意图型 - 包含多个独立问题，用“、”或“？”分隔
5. 反问型 - 包含“不会”“难道”等反问语气
说明：如果同时存在多意图型、模糊指代型，优先级为 多意图型 > 模糊指代型。

请返回 JSON 格式的结果（只输出 JSON）：
{{
  "query_type": "查询类型",
  "rewritten_query": "改写后的查询",
  "confidence": 0.0
}}

### 对话历史 ###
{conversation}

### 上下文信息 ###
{context}

### 原始查询 ###
{query}

### 分析结果 ###"""


# -------------------------------
# 配置与工具
# -------------------------------
@dataclass
class QueryRewriterConfig:
    model: str = "qwen-turbo-latest"
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: Optional[int] = None  # None 表示按模型缺省
    seed: Optional[int] = 42          # 可复现
    timeout: int = 30                 # 请求超时秒
    retries: int = 3                  # 重试次数
    backoff_base: float = 0.6         # 退避基数
    max_context_chars: int = 4000     # 上下文截断，保护 token/费用


def _truncate(s: str, limit: int) -> str:
    if s is None:
        return ""
    return s if len(s) <= limit else (s[:limit] + " ...[truncated]")


def _call_llm(prompt, cfg):
    messages = [{"role": "user", "content": prompt}]
    last_err = None
    for attempt in range(cfg.retries + 1):
        try:
            resp = dashscope.Generation.call(
                model=cfg.model,
                messages=messages,
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
        except DASHSCOPE_EXCEPTIONS as e:
            last_err = e
            print(f"[DashScope Error] {type(e).__name__}: {e}")
            if attempt >= cfg.retries:
                break
            time.sleep(cfg.backoff_base * (2 ** attempt))
    raise RuntimeError(f"dashscope call failed after retries: {last_err}")


def _json_loose_parse(s: str) -> Any:
    """
    宽松解析：优先严格 json.loads；失败则尝试截取 JSON 片段；再失败返回原文。
    """
    try:
        return json.loads(s)
    except Exception:
        # 粗略截取第一个 '[' 或 '{' 开始到最后一个 ']' 或 '}' 结束
        start = min([i for i in [s.find("["), s.find("{")] if i != -1] or [-1])
        end = max(s.rfind("]"), s.rfind("}"))
        if start != -1 and end != -1 and end > start:
            snippet = s[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
        return s.strip()


# -------------------------------
# 主类
# -------------------------------
class QueryRewriter:
    def __init__(self, config: Optional[QueryRewriterConfig] = None):
        self.cfg = config or QueryRewriterConfig()

    # ---- 五类改写 ----
    def rewrite_context_dependent_query(self, current_query: str, conversation_history: str) -> str:
        prompt = CTX_DEP_PROMPT.format(
            conversation=_truncate(conversation_history, self.cfg.max_context_chars),
            query=current_query.strip(),
        )
        return _call_llm(prompt, self.cfg)

    def rewrite_comparative_query(self, query: str, context_info: str) -> str:
        prompt = CMP_PROMPT.format(
            context=_truncate(context_info, self.cfg.max_context_chars),
            query=query.strip(),
        )
        return _call_llm(prompt, self.cfg)

    def rewrite_ambiguous_reference_query(self, current_query: str, conversation_history: str) -> str:
        prompt = AMB_PROMPT.format(
            conversation=_truncate(conversation_history, self.cfg.max_context_chars),
            query=current_query.strip(),
        )
        return _call_llm(prompt, self.cfg)

    def rewrite_multi_intent_query(self, query: str) -> List[str]:
        prompt = MULTI_PROMPT.format(query=query.strip())
        out = _call_llm(prompt, self.cfg)
        parsed = _json_loose_parse(out)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
        # 兜底：拆分行，过滤空
        return [q for q in str(parsed).splitlines() if q.strip()]

    def rewrite_rhetorical_query(self, current_query: str, conversation_history: str) -> str:
        prompt = RHE_PROMPT.format(
            conversation=_truncate(conversation_history, self.cfg.max_context_chars),
            query=current_query.strip(),
        )
        return _call_llm(prompt, self.cfg)

    # ---- 自动识别 + 改写 ----
    def auto_rewrite_query(self, query: str, conversation_history: str = "", context_info: str = "") -> Dict[str, Any]:
        prompt = AUTO_PROMPT.format(
            conversation=_truncate(conversation_history, self.cfg.max_context_chars),
            context=_truncate(context_info, self.cfg.max_context_chars),
            query=query.strip(),
        )
        out = _call_llm(prompt, self.cfg)
        parsed = _json_loose_parse(out)
        # 兜底结构
        if not isinstance(parsed, dict):
            parsed = {"query_type": "未知类型", "rewritten_query": query, "confidence": 0.5}
        parsed.setdefault("query_type", "未知类型")
        parsed.setdefault("rewritten_query", query)
        try:
            parsed["confidence"] = float(parsed.get("confidence", 0.5))
        except Exception:
            parsed["confidence"] = 0.5
        return parsed

    def auto_rewrite_and_execute(self, query: str, conversation_history: str = "", context_info: str = "") -> Dict[str, Any]:
        result = self.auto_rewrite_query(query, conversation_history, context_info)
        qtype = result.get("query_type", "")
        if "上下文依赖" in qtype:
            final = self.rewrite_context_dependent_query(query, conversation_history)
        elif "对比" in qtype:
            final = self.rewrite_comparative_query(query, context_info or conversation_history)
        elif "模糊指代" in qtype:
            final = self.rewrite_ambiguous_reference_query(query, conversation_history)
        elif "多意图" in qtype:
            final = self.rewrite_multi_intent_query(query)
        elif "反问" in qtype:
            final = self.rewrite_rhetorical_query(query, conversation_history)
        else:
            final = result.get("rewritten_query", query)
        return {
            "original_query": query,
            "detected_type": qtype,
            "confidence": result.get("confidence", 0.5),
            "rewritten_query": final,
            "auto_rewrite_result": result,
        }


# -------------------------------
# Demo
# -------------------------------
def main():
    rewriter = QueryRewriter()

    print("=== 示例：上下文依赖型 ===")
    conv = (
        '用户: 我想了解一下上海迪士尼乐园的最新项目。\n'
        'AI: 上海迪士尼乐园最新推出了“疯狂动物城”主题园区……\n'
        '用户: 这个园区有什么游乐设施？\n'
        'AI: ……\n'
    )
    q = "还有其他设施吗？"
    print(rewriter.rewrite_context_dependent_query(q, conv), "\n")

    print("=== 示例：对比型 ===")
    conv2 = 'AI: 上海迪士尼乐园推出了疯狂动物城与蜘蛛侠主题园区'
    q2 = "哪个游玩的时间比较长，比较有趣"
    print(rewriter.rewrite_comparative_query(q2, conv2), "\n")

    print("=== 示例：模糊指代型 ===")
    conv3 = 'AI: 上海与香港迪士尼都有烟花表演'
    q3 = "都什么时候开始？"
    print(rewriter.rewrite_ambiguous_reference_query(q3, conv3), "\n")

    print("=== 示例：多意图型 ===")
    q4 = "门票多少钱？需要提前预约吗？停车费怎么收？"
    print(rewriter.rewrite_multi_intent_query(q4), "\n")

    print("=== 示例：反问型 ===")
    conv5 = 'AI: 下周六门票售罄'
    q5 = "这不会也要提前一个月预订吧？"
    print(rewriter.rewrite_rhetorical_query(q5, conv5), "\n")

    print("=== 示例：自动识别 ===")
    tests = ["还有其他游乐项目吗？", "哪个园区更好玩？", "都适合小朋友吗？", "有什么餐厅？价格怎么样？", "这不会也要排队两小时吧？"]
    for t in tests:
        print(t, "->", rewriter.auto_rewrite_query(t))


if __name__ == "__main__":
    main()
