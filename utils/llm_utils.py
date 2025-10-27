import json
import dashscope
import os
from dotenv import load_dotenv

load_dotenv()

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def get_completion(prompt, model="qwen-turbo-latest", temperature=0.3):
    """调用 DashScope LLM 并返回文本结果"""
    response = dashscope.Generation.call(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        result_format="message",
        temperature=temperature,
    )
    return response.output.choices[0].message.content


def preprocess_json_response(response: str) -> str:
    """去除 LLM 返回中包含的 Markdown ``` 包裹"""
    if not response:
        return ""
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    return response.strip()


def safe_json_loads(text: str, default=None):
    """稳健地解析 JSON，失败时返回默认值"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default or {}
