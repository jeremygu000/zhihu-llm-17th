这是因为 **OpenAI Function Calling 的对话协议规范**。`"role": "function"` 是专门用于标识函数执行结果的消息类型。

## 1. **对话角色协议**

在 OpenAI Function Calling 中，有几种固定的角色：

| 角色 | 用途 | 示例 |
|------|------|------|
| `"user"` | 用户输入 | `{"role": "user", "content": "大连天气怎样"}` |
| `"assistant"` | 助手回复 | `{"role": "assistant", "content": "今天天气..."}` |
| `"system"` | 系统提示 | `{"role": "system", "content": "你是一个助手"}` |
| `"function"` | **函数执行结果** | `{"role": "function", "name": "get_weather", "content": "..."}` |

## 2. **完整的对话流程**

```python
# 完整的 messages 历史示例
messages = [
    # 第1轮：用户提问
    {"role": "user", "content": "大连的天气怎样"},
    
    # 第2轮：助手决定调用函数
    {
        "role": "assistant", 
        "content": "",
        "function_call": {
            "name": "get_current_weather",
            "arguments": "{\"location\": \"大连\"}"
        }
    },
    
    # 第3轮：函数执行结果
    {
        "role": "function",           # ← 关键：标识这是函数结果
        "name": "get_current_weather", # 哪个函数的执行结果
        "content": "{\"temperature\": 10, \"unit\": \"celsius\"}"  # 执行结果
    },
    
    # 第4轮：助手基于函数结果给出最终回答
    {
        "role": "assistant", 
        "content": "大连当前温度10摄氏度，天气晴朗"
    }
]
```

## 3. **为什么需要区分角色？**

### **模型需要知道消息来源**
```python
# 如果混用角色会怎样？
messages = [
    {"role": "user", "content": "大连天气"},
    {"role": "assistant", "content": "", "function_call": {...}},
    # 错误写法：把函数结果也标为 assistant
    {"role": "assistant", "content": "{\"temp\": 10}"},  # ❌ 混淆
    # 正确写法：明确标识来源
    {"role": "function", "name": "get_weather", "content": "{\"temp\": 10}"}  # ✅ 清晰
]
```

### **模型处理逻辑不同**
- `"assistant"` 角色：模型生成的文本
- `"function"` 角色：外部工具返回的数据
- 模型需要知道哪些内容是它自己生成的，哪些是外部数据

## 4. **实际执行过程**

```python
def run_conversation():
    # 初始用户消息
    messages = [{"role": "user", "content": "大连的天气怎样"}]
    
    # 第1次调用：模型返回 function_call
    response = get_response(messages)
    message = response.output.choices[0].message  # 这是 assistant 角色
    
    # 添加 assistant 的 function_call 决定
    messages.append({
        "role": "assistant",           # ← 助手决定调用函数
        "content": message.content,
        "function_call": message.function_call
    })
    
    # 执行函数调用
    tool_response = get_current_weather("大连")
    
    # 添加函数执行结果
    messages.append({
        "role": "function",            # ← 明确标识这是函数结果
        "name": "get_current_weather",  # 来自哪个函数
        "content": tool_response        # 函数返回的数据
    })
    
    # 第2次调用：模型基于函数结果生成回答
    response = get_response(messages)
    final_message = response.output.choices[0].message  # 这是最终的 assistant 回答
```

## 5. **如果你错误使用角色**

```python
# 错误示例：把函数结果标为 assistant
tool_info = {"role": "assistant", "content": tool_response}  # ❌

# 结果可能是：
# - 模型混淆数据来源
# - 可能把函数结果当作之前的对话内容
# - 影响后续的推理和回答质量
```

## 总结

`"role": "function"` 是 **OpenAI Function Calling 协议的标准要求**，用于：

1. **明确数据来源**：区分模型生成内容和外部工具数据
2. **维护对话上下文**：清晰的对话历史结构
3. **支持复杂推理**：模型知道何时依赖外部工具，何时自己回答
4. **遵循规范**：符合 OpenAI API 的设计约定

这样可以确保模型能够正确理解整个对话流程：用户提问 → 助手决定调用工具 → 工具执行 → 助手基于工具结果回答。