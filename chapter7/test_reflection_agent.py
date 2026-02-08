# test_reflection_agent.py
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from my_reflection_agent import MyReflectionAgent

load_dotenv()
llm = HelloAgentsLLM()

# 使用默认通用提示词
general_agent = MyReflectionAgent(name="我的反思助手", llm=llm)

# 使用自定义代码生成提示词（类似第四章）
code_prompts = {
    "initial": "你是写作专家，请编写文章：{task}",
    "reflect": "请审查文章的合理性：\n任务：{task}\n代码：{code}",
    "refine": "请根据反馈优化文章：\n任务：{task}\n反馈：{feedback}"
}
code_agent = MyReflectionAgent(
    name="我的文章生成助手",
    llm=llm,
    custom_prompts=code_prompts
)

# 测试使用
result = code_agent.run("写一篇关于人工智能发展历程的简短文章，字数要求在200字以内。")
print(f"最终结果: {result}")