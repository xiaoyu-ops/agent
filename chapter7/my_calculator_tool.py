"""简单的计算器工具模块。

提供安全的表达式计算函数 `my_calculate`，以及用于注册该工具的 `create_calculator_registry`。

实现要点：
- 使用 `ast.parse(..., mode='eval')` 将表达式解析为 AST，避免直接使用 eval 导致的安全问题。
- 仅允许受控的运算符和函数（见下方 `operators` 和 `functions`），其他节点会被拒绝。
"""

import ast
import operator
import math
from hello_agents import ToolRegistry

def my_calculate(expression: str) -> str:
    """简单的数学计算函数。

    接受一个数学表达式字符串，返回计算结果的字符串形式。
    - 若表达式为空，返回友好提示。
    - 若解析或计算失败，返回错误提示而不是抛出异常。

    安全说明：函数通过 AST 解析和受限节点求值来避免任意代码执行。
    """
    if not expression.strip():
        return "计算表达式不能为空"

    # 仅允许的二元运算符映射（类型->实现函数）
    operators = {
        ast.Add: operator.add,      # +
        ast.Sub: operator.sub,      # -
        ast.Mult: operator.mul,     # *
        ast.Div: operator.truediv,  # /
    }

    # 允许的函数/常量（名称->可调用或值）
    functions = {
        'sqrt': math.sqrt,
        'pi': math.pi,
    }

    try:
        # 将输入的表达式解析为 AST（Expression 节点），mode='eval' 表示只解析单个表达式
        node = ast.parse(expression, mode='eval')
        # 人眼中的字符串："3 + 4 * 2"。我们根据常识知道先算乘法。

        # 计算机眼中的字符串：一串字符编码 ['3', ' ', '+', ' ', '4', ' ', '*', ' ', '2']。

        # ast.parse 的任务：按照 Python 的语法规则，把这串扁平的字符按照优先级和逻辑叠起来，变成一个立体结构。
        # 从 Expression 节点获取主体（root expression），传入自定义求值函数进行递归计算
        result = _eval_node(node.body, operators, functions) # node.body 是 Expression 节点下的实际表达式节点
        return str(result)
    except Exception:
        # 捕获并返回友好错误信息，避免抛出内部异常
        return "计算失败，请检查表达式格式"

def _eval_node(node, operators, functions):
    """简化的表达式求值。

    递归遍历 AST 节点，按节点类型计算结果。仅支持受限节点：
    - ast.Constant: 直接返回常量值
    - ast.BinOp: 处理二元运算（使用预定义的 operators 字典）
    - ast.Call: 支持简单函数调用（函数名在 functions 中）
    - ast.Name: 支持预定义常量（例如 pi）

    对于不支持的节点或形式，会抛出 ValueError，由上层捕获并返回友好信息。
    """
    # 常量（数字）
    if isinstance(node, ast.Constant):
        return node.value

    # 二元运算（例如 a + b）
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left, operators, functions)
        right = _eval_node(node.right, operators, functions)
        op = operators.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符: {type(node.op)}")
        return op(left, right)

    # 函数调用（例如 sqrt(x)）
    elif isinstance(node, ast.Call):
        # 仅支持简单的名称调用，不支持 lambda 或复杂表达式作为函数
        if isinstance(node.func, ast.Name): # node.func是括号的左边部分
            func_name = node.func.id
            if func_name in functions:
                args = [_eval_node(arg, operators, functions) for arg in node.args]
                return functions[func_name](*args)
        raise ValueError("不支持的函数调用形式")

    # 预定义名称（如 pi）
    elif isinstance(node, ast.Name):
        if node.id in functions:
            return functions[node.id]

    # 其他不受支持的节点
    raise ValueError(f"不支持的表达式节点: {type(node)}")

def create_calculator_registry():
    """创建包含计算器的工具注册表"""
    registry = ToolRegistry()

    # 注册计算器函数
    registry.register_function(
        name="my_calculator",
        description="简单的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数",
        func=my_calculate
    )

    return registry