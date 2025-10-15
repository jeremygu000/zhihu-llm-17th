import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# 天气函数
def get_current_weather(location, unit="celsius"):
    """获取当前天气"""
    temperature = -1
    if '大连' in location or 'Dalian' in location:
        temperature = 10
    elif '上海' in location or 'Shanghai' in location:
        temperature = 36
    elif '深圳' in location or 'Shenzhen' in location:
        temperature = 37
    
    weather_info = {
        "location": location,
        "temperature": temperature,
        "unit": unit,
        "forecast": ["晴天", "微风"],
    }
    return json.dumps(weather_info, ensure_ascii=False)

# 函数定义
functions = [
    {
        "name": "get_current_weather",
        "description": "获取指定城市的当前天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：北京、上海、大连"
                },
                "unit": {
                    "type": "string", 
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位，摄氏度或华氏度"
                }
            },
            "required": ["location"]
        }
    }
]

def run_conversation_with_functions():
    """使用 Functions 进行对话"""
    llm = ChatOpenAI(model="gpt-4.1-mini")
    
    query = "大连的天气怎样？"
    messages = [HumanMessage(content=query)]
    # 第一次调用
    response = llm.invoke(messages, functions=functions)
    messages.append(response)
    
    print(f"AI回复: {response.content}")
    
    # 检查是否需要调用函数
    if response.additional_kwargs and "function_call" in response.additional_kwargs:
        function_call = response.additional_kwargs["function_call"]
        function_name = function_call["name"]
        arguments = json.loads(function_call["arguments"])
        
        print(f"检测到函数调用: {function_name}")
        print(f"参数: {arguments}")
        
        # 执行函数调用
        if function_name == "get_current_weather":
            location = arguments.get("location")
            unit = arguments.get("unit", "celsius")
            function_response = get_current_weather(location, unit)
            
            # 将函数执行结果加入对话
            messages.append(AIMessage(
                content="",
                additional_kwargs={
                    "function_call": function_call,
                    "name": function_name
                }
            ))
            
            # 添加函数执行结果
            messages.append(HumanMessage(
                content=function_response,
                name=function_name
            ))
            
            # 第二次调用，使用 invoke
            final_response = llm.invoke(messages)
            messages.append(final_response)
            
            print(f"最终回复: {final_response.content}")
            return final_response.content
    
    return response.content

if __name__ == "__main__":
    result = run_conversation_with_functions()
    print("\n最终结果:", result)