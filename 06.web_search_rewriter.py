# -*- coding: utf-8 -*-
# Query 联网搜索改写（稳健版）
from __future__ import annotations
import os, json, time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

import dashscope

# ---- 全局配置：API Key 一次性设置 ----
load_dotenv()
DASHSCOPE_API_KEY = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

# ---- 常量：Prompt 模板 ----
IDENTIFY_PROMPT = """你是一个智能的查询分析专家。请分析用户的查询，判断是否需要联网搜索来获取最新、最准确的信息。

需要联网搜索的情况包括：
1. 时效性信息（“最新”“今天”“现在”“实时”“当前”等）
2. 价格信息（“多少钱”“价格”“费用”“票价”等）
3. 营业信息（“营业时间”“开放时间”“是否开放”等）
4. 活动信息（“活动”“演出”“庆典”等）
5. 天气信息（“天气”“下雨”“温度”等）
6. 交通信息（“怎么去”“地铁”“公交”等）
7. 预订信息（“预订”“预约”“订票”等）
8. 实时状态（“排队”“拥挤”“人流量”等）

只输出 JSON：
{{
  "need_web_search": true/false,
  "search_reason": "字符串",
  "confidence": 0.0
}}

### 对话历史 ###
{conversation}

### 用户查询 ###
{query}

### 分析结果（仅 JSON） ###
"""

REWRITE_PROMPT = """你是专业的搜索查询优化专家。请将用户的查询改写为适合搜索引擎检索的形式。
技巧：加地点/时间范围、关键词化、去口语化、补充同义词与相关词、明确意图。

只输出 JSON：
{{
  "rewritten_query": "字符串",
  "search_keywords": ["关键词1","关键词2"],
  "search_intent": "字符串",
  "suggested_sources": ["官方网站","旅游网站"]
}}

### 原始查询 ###
{query}

### 搜索类型 ###
{search_type}

### 改写结果（仅 JSON） ###
"""

STRATEGY_PROMPT = """你是搜索策略专家。请制定检索策略：
- 主要搜索词（核心）
- 扩展搜索词（相关/同义）
- 搜索平台（中文/英文）
- 时间范围（结合今天日期）

今天日期：{today}

只输出 JSON：
{{
  "primary_keywords": ["主要关键词"],
  "extended_keywords": ["扩展关键词"],
  "search_platforms": ["平台1","平台2"],
  "time_range": "时间范围"
}}

### 用户查询 ###
{query}

### 搜索类型 ###
{search_type}

### 搜索策略（仅 JSON） ###
"""

# ---- 配置 ----
@dataclass
class LLMConfig:
    model: str = "qwen-turbo-latest"
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: Optional[int] = None   # 让 SDK 使用默认也可以
    seed: Optional[int] = 42           # 可复现
    timeout: int = 30
    retries: int = 2
    backoff_base: float = 0.7
    max_context_chars: int = 3500      # 上下文截断，控成本

# ---- 工具函数 ----
def _truncate(s: str, limit: int) -> str:
    s = s or ""
    return s if len(s) <= limit else (s[:limit] + " …[truncated]")

def _json_loose_parse(s: str) -> Any:
    """宽松 JSON 解析：先严格，再截片段，最后兜底原文"""
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        start = min([i for i in [s.find("["), s.find("{")] if i != -1] or [-1])
        end = max(s.rfind("]"), s.rfind("}"))
        if start != -1 and end > start:
            snippet = s[start:end+1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
        return s

def _call_llm(prompt: str, cfg: LLMConfig) -> str:
    messages = [{"role": "user", "content": prompt}]
    last_err: Optional[Exception] = None
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
        except Exception as e:
            last_err = e
            if attempt >= cfg.retries:
                break
            time.sleep(cfg.backoff_base * (2 ** attempt))
    raise RuntimeError(f"LLM 调用失败（已重试）：{last_err}")

def _norm_keywords(items: List[str]) -> List[str]:
    """关键词标准化：去空/小写/去重/长度上限"""
    seen, out = set(), []
    for it in items or []:
        t = (it or "").strip().lower()
        if not t or len(t) > 128:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ---- 主类 ----
class WebSearchQueryRewriter:
    def __init__(self, model: str = "qwen-turbo-latest", cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig(model=model)

    # 识别是否需要联网
    def identify_web_search_needs(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        prompt = IDENTIFY_PROMPT.format(
            conversation=_truncate(conversation_history, self.cfg.max_context_chars),
            query=query.strip(),
        )
        raw = _call_llm(prompt, self.cfg)
        data = _json_loose_parse(raw)
        # 结构兜底
        out = {
            "need_web_search": False,
            "search_reason": "无法解析",
            "confidence": 0.5,
        }
        if isinstance(data, dict):
            out["need_web_search"] = bool(data.get("need_web_search", False))
            out["search_reason"] = str(data.get("search_reason", out["search_reason"]))[:500]
            try:
                out["confidence"] = float(data.get("confidence", 0.5))
            except Exception:
                pass
        return out

    # 为搜索改写
    def rewrite_for_web_search(self, query: str, search_type: str = "general") -> Dict[str, Any]:
        prompt = REWRITE_PROMPT.format(query=query.strip(), search_type=search_type)
        raw = _call_llm(prompt, self.cfg)
        data = _json_loose_parse(raw)
        out = {
            "rewritten_query": query.strip(),
            "search_keywords": _norm_keywords([query]),
            "search_intent": "信息查询",
            "suggested_sources": ["官方网站", "资讯网站"],
        }
        if isinstance(data, dict):
            rq = str(data.get("rewritten_query") or "").strip()
            if rq:
                out["rewritten_query"] = rq
            out["search_keywords"] = _norm_keywords(data.get("search_keywords") or out["search_keywords"])
            out["search_intent"] = (data.get("search_intent") or out["search_intent"]).strip()[:100]
            ss = data.get("suggested_sources") or out["suggested_sources"]
            out["suggested_sources"] = [s.strip() for s in ss if isinstance(s, str) and s.strip()][:6]
        return out

    # 搜索策略
    def generate_search_strategy(self, query: str, search_type: str = "general") -> Dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = STRATEGY_PROMPT.format(today=today, query=query.strip(), search_type=search_type)
        raw = _call_llm(prompt, self.cfg)
        data = _json_loose_parse(raw)
        out = {
            "primary_keywords": _norm_keywords([query]),
            "extended_keywords": [],
            "search_platforms": ["Google", "Bing", "百度"],
            "time_range": "最近30天",
        }
        if isinstance(data, dict):
            out["primary_keywords"] = _norm_keywords(data.get("primary_keywords") or out["primary_keywords"])
            out["extended_keywords"] = _norm_keywords(data.get("extended_keywords") or out["extended_keywords"])
            sps = data.get("search_platforms") or out["search_platforms"]
            out["search_platforms"] = [s.strip() for s in sps if isinstance(s, str) and s.strip()][:8]
            tr = str(data.get("time_range") or "").strip()
            if tr:
                out["time_range"] = tr[:100]
        return out

    # 自动流程：识别 → 改写 → 策略
    def auto_web_search_rewrite(self, query: str, conversation_history: str = "", search_type: str = "general") -> Dict[str, Any]:
        analysis = self.identify_web_search_needs(query, conversation_history)
        print('identify_web_search_needs', analysis)
        
        need_search = analysis.get("need_web_search", False)
        if not need_search:
            return {
                "need_web_search": False,
                "reason": "查询不需要联网搜索",
                "original_query": query,
                "confidence": analysis.get("confidence", 0.5),
            }
        
        rewritten = self.rewrite_for_web_search(query, search_type=search_type)
        print('rewrite_for_web_search', rewritten)

        strategy = self.generate_search_strategy(query, search_type=search_type)
        print('generate_search_strategy', strategy)

        return {
            "need_web_search": True,
            "search_reason": analysis.get("search_reason", ""),
            "confidence": analysis.get("confidence", 0.5),
            "original_query": query,
            "rewritten_query": rewritten["rewritten_query"],
            "search_keywords": rewritten["search_keywords"],
            "search_intent": rewritten["search_intent"],
            "suggested_sources": rewritten["suggested_sources"],
            "search_strategy": strategy,
        }

# ---- Demo ----
if __name__ == "__main__":
    ws = WebSearchQueryRewriter()
    q1 = "上海迪士尼乐园今天开放吗？现在人多不多？"
    conv1 = "用户：我想去上海迪士尼乐园玩\nAI：好呀"
    res1 = ws.auto_web_search_rewrite(q1, conv1)
    print(json.dumps(res1, ensure_ascii=False, indent=2))
