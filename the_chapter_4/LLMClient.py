import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
from serpapi import SerpApiClient
from typing import Dict, Any
import re

# 加载 .env 文件中的环境变量
load_dotenv()

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"正在调用 {self.model} 模型...")
        try:    # 这行代码就是“秘书拿起电话，开始拨号并交流”的那一瞬间。
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                
                delta = chunk.choices[0].delta
                
                # 检查是否有内容 (content)
                if delta.content:
                    content = delta.content
                    print(content, end="", flush=True)
                    collected_content.append(content)
                
                # 调试: 如果没有内容，也没有refusal，打印一下delta看看是什么 (可能是tool_calls)
                elif not (hasattr(delta, 'refusal') and delta.refusal):
                    # 为了避免刷屏，只打印非空且非content的delta
                    if delta.tool_calls:
                         print(f"\n[Debug:检测到工具调用请求] {delta.tool_calls}")
                    elif delta.role:
                         pass # 忽略 role 消息
                    else:
                         # 即使是空delta也可能是重要的结束标志，或者包含其他信息
                         # print(f"[Debug: Raw Delta] {delta}")
                         pass

                # 双重保险：检查是否有 refusal
                if hasattr(delta, 'refusal') and delta.refusal:
                    print(f"\n[模型拒绝响应]: {delta.refusal}")
                    return delta.refusal

            print()  # 在流式输出结束后换行
            
            full_response = "".join(collected_content)
            
            # 如果是工具调用导致的空内容，我们给一个友好的提示（虽然ReAct框架里我们希望它直接输出Action文本）
            if not full_response:
                print("警报: 接收到的内容为空。可能模型正在尝试直接调用API定义的工具(Tool Calls)，而不是输出ReAct文本格式。")
                print("建议: 尝试在 prompt 中强调‘不要使用 native tool calling，仅输出文本’，或者检查模型参数。")
                
            return full_response

        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return None

def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])

# ReAct 提示词模板 - 系统指令部分
REACT_SYSTEM_PROMPT = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。
"""

# ReAct 提示词模板 - 用户输入部分
REACT_USER_PROMPT = """
现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            
            system_content = REACT_SYSTEM_PROMPT.format(tools=tools_desc)
            user_content = REACT_USER_PROMPT.format(question=question, history=history_str)

            # 2. 调用LLM进行思考 (使用 System + User 消息结构)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

# (这段逻辑在 run 方法的 while 循环内)
            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                finish_match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                final_answer = finish_match.group(1) if finish_match else action
                print(f" 最终答案: {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue

            print(f" 行动: {tool_name}[{tool_input}]")
            
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具
            print(f" 观察: {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None


    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。"""
        thought_match = re.search(r"Thought:\s*(.*?)\s*Action:", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*)", text, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else None
        
        if not thought:
            # Fallback if "Action:" is not found or formatting is loose
            thought_single = re.search(r"Thought: (.*)", text)
            if thought_single:
                 thought = thought_single.group(1).strip()

        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。"""
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None



# # --- 客户端使用示例 ---
# if __name__ == '__main__':
#     try:
#         llmClient = HelloAgentsLLM()
        
#         exampleMessages = [
#             {"role": "system", "content": "You are a helpful assistant that writes Python code."},
#             {"role": "user", "content": "写一个快速排序算法"}
#         ]
        
#         print("--- 调用LLM ---")
#         responseText = llmClient.think(exampleMessages)
#         if responseText:
#             print("\n\n--- 完整模型响应 ---")
#             print(responseText)

#     except ValueError as e:
#         print(e)

# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    
    # 3. 打印可用的工具
    # print("\n--- 可用的工具 ---")
    # print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    #print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    agent = ReActAgent(llm_client=HelloAgentsLLM(), tool_executor=toolExecutor)
    final_answer = agent.run("今天币圈发生了什么？")
    
    # tool_name = "Search"
    # tool_input = "英伟达最新的GPU型号是什么"

    # tool_function = toolExecutor.getTool(tool_name)
    # if tool_function:
    #     observation = tool_function(tool_input)
    #     print("--- 观察 (Observation) ---")
    #     print(observation)
    # else:
    #     print(f"错误:未找到名为 '{tool_name}' 的工具。")
        
