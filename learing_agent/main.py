import os
import re
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tavily import TavilyClient

# --- 修复点 1：强制读取当前脚本所在目录下的 .env 文件 ---
# 这样无论你在哪里运行 python 命令，都能准确定位到 .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- 准备工作 ---
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_ID = os.getenv("OPENAI_MODEL_ID")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# 检查配置是否加载成功
if not API_KEY:
    print("错误：未检测到 OPENAI_API_KEY，请检查 .env 文件路径或内容！")
    exit(1)
if not TAVILY_KEY:
    print("警告：未检测到 TAVILY_API_KEY，搜索功能将无法使用。")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL 
)

# --- 核心 Prompt ---
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具：
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求：
你的每次回复必须严格遵循以下格式，包含一对 Thought 和 Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action 的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示：
- 每次只输出一对 Thought-Action
- Action 必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束
"""

def get_weather(city: str) -> str:
    """通过调用 wttr.in API 查询真实的天气信息。"""
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        return f"{city}当前天气:{weather_desc}, 气温{temp_c}摄氏度"
    except Exception as e:
        return f"错误:查询天气失败 - {e}"
    
def get_attraction(city: str, weather: str) -> str:
    """根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"
    
    tavily = TavilyClient(api_key=api_key)
    query = f"'{city}' 在 '{weather}'天气下最值得去的旅游景点推荐及理由"
    
    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        if response.get("answer"):
            return response["answer"]
        
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
            
        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"
    
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        print(f"正在调用模型: {self.model} ...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用 API 发生错误: {e}")
            return f"Error: {e}"

# --- 初始化 ---
llm = OpenAICompatibleClient(model=MODEL_ID, api_key=API_KEY, base_url=BASE_URL)

user_prompt = "你好，请帮我查询一下今天深圳的天气，然后根据天气推荐一个适合的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "="*40)

# --- 主循环 ---
for i in range(5):
    print(f"--- 循环 {i+1} ---")
    full_prompt = "\n".join(prompt_history)
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    
    # 简单的错误处理：如果返回包含 Error 则跳过
    if "Error:" in llm_output:
        print("模型调用失败，停止循环。")
        break

    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|$)|\Z)', llm_output, re.DOTALL)
    if match:
        llm_output = match.group(1).strip()
    
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)
    
    if "Action: Finish" in llm_output:
        try:
            final_answer = llm_output.split("Finish[")[1].split("]")[0]
            print(f"\n最终答案: {final_answer}")
        except:
            print(f"\n最终答案 (解析格式略有偏差): {llm_output}")
        break
        
    action_match = re.search(r'Action:\s*(\w+)\((.*)\)', llm_output)
    if action_match:
        tool_name = action_match.group(1)
        args_str = action_match.group(2)
        # 简单的参数解析，注意：如果参数值里有引号可能会解析失败，但简单场景够用了
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
        
        print(f"系统正在执行工具: {tool_name}，参数: {kwargs}")
        
        if tool_name in available_tools:
            try:
                tool_result = available_tools[tool_name](**kwargs)
            except Exception as e:
                tool_result = f"工具执行出错: {e}"
        else:
            tool_result = f"错误: 找不到名为 {tool_name} 的工具"
            
        observation = f"Observation: {tool_result}"
        print(f"{observation}\n" + "="*40)
        prompt_history.append(observation)
    else:
        print("警告: 未检测到有效的 Action 格式，正在重试...")