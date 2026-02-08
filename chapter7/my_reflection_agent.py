from typing import List, Dict, Any, Optional
from collections.abc import Iterator as IteratorABC
from hello_agents import HelloAgentsLLM

class Memory:
    """
    一个简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """

    def __init__(self):
        """
        初始化一个空列表来存储所有记录。
        """
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        向记忆中添加一条新记录。

        参数:
        - record_type (str): 记录的类型 ('execution' 或 'reflection')。
        - content (str): 记录的具体内容 (例如，生成的代码或反思的反馈)。
        """
        record = {"type": record_type, "content": content}
        self.records.append(record)
        print(f"记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有记忆记录格式化为一个连贯的字符串文本，用于构建提示词。
        """
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory_parts.append(f"--- 上一轮尝试 (代码) ---\n{record['content']}")
            elif record['type'] == 'reflection':
                trajectory_parts.append(f"--- 评审员反馈 ---\n{record['content']}")
        
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        获取最近一次的执行结果 (例如，最新生成的代码)。
        如果不存在，则返回 None。
        """
        for record in reversed(self.records):# reversed是从后往前翻就是倒序
            if record['type'] == 'execution':
                return record['content']
        return None

DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务:

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间:

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答:

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}

# 假设 llm_client.py 和 memory.py 已定义


class MyReflectionAgent:
    def __init__(self, name: str, llm, max_iterations=3, custom_prompts: Optional[Dict[str, str]] = None):
        self.name = name
        self.llm = llm
        self.memory = Memory()
        self.max_iterations = max_iterations
        self.custom_prompts = custom_prompts or {}

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = self.custom_prompts.get("initial", DEFAULT_PROMPTS["initial"]).format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环:反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = self.custom_prompts.get("reflect", DEFAULT_PROMPTS["reflect"]).format(task=task, content=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            print(f"\n--- 反思反馈 ---\n{feedback}")
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback:
                print("\n反思认为代码已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = self.custom_prompts.get("refine", DEFAULT_PROMPTS["refine"]).format(
                task=task,
                last_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)
        
        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        """一个辅助方法,用于调用LLM并获取完整的流式响应。"""
        messages = [{"role": "user", "content": prompt}]
        llm_client = getattr(self, "llm_client", None) or getattr(self, "llm", None)
        if llm_client is None:
            raise AttributeError("MyReflectionAgent未配置llm_client")
        if hasattr(llm_client, "invoke"):
            try:
                return llm_client.invoke(messages=messages)
            except Exception:
                pass
        response = llm_client.think(messages=messages)
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, IteratorABC):
            return "".join(chunk for chunk in response)
        return str(response)
