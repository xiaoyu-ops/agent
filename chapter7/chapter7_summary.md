## 本地模型调试
- 问题处理反思

- 表象错误是 Python 端的 502，但根因在服务端兼容层，先用最小化的 HTTP 请求验证服务本体可用性很关键。
- 通过对比 `Invoke-RestMethod` 与 `openai` SDK 行为，确认 SDK 的请求细节触发了 Ollama 的兼容性问题，避免在 SDK 层死磕。
- 将 `localhost` 改为 `127.0.0.1` 可规避本地解析/路由差异带来的 404/502，属于低成本但高收益的排查项。
- 在不确定的情况下，先用非流式调用再逐步恢复流式能力，有助于缩小问题面。
- 记录关键假设与验证步骤（模型能否 `ollama run`，`/v1/chat/completions` 是否 200），能让后续复现与排错更快。

## 抽象基类（ABC）与普通继承的区别
简单来说，ABC 就是用来“订规矩”的模板：它强制要求子类必须实现指定的方法（如 think），否则程序就不准运行。
普通继承：如果你没写指定方法，代码能跑，直到运行到那一行发现没方法时才会崩溃（这叫“死在半路上”）。

ABC 继承：如果你没写指定方法，Python 在你尝试创建这个类的实例（即 llm = MyLLM()）的那一秒就会直接报错并拦截。它会告诉你：“规矩里说了必须有 think，你没写，所以我拒绝创建这个对象。”

## 类型注解详解：Optional[str] = None（optional的用处）
1. Optional[str]（类型说明）
它告诉 Python 插件和阅读代码的人，这个变量的取值范围有两种可能：

字符串 (str)：比如 "gpt-4", "hello"。

空值 (None)：表示什么都没有。

2. = None（默认值）
这代表默认设置。

如果你在调用函数时没有给这个参数传值，它就会自动变成 None。

##  @abstractmethod 装饰器的作用
`@abstractmethod` 是 Python 的一个装饰器，用在抽象基类（ABC）的方法上。它的主要作用是强制子类必须实现这个方法，否则子类就不能被实例化。

## 流式模式的实现
打字机模式 (yield 和 chunk)
Python
for chunk in self.llm.stream_invoke(messages, **kwargs):
    full_response += chunk
    print(chunk, end="", flush=True)
    yield chunk
这是最关键的部分：

stream_invoke：告诉大模型（比如你刚装好的 Ollama），请用“流”的方式回话。

chunk (数据块)：AI 返回的不是一句话，而是一小块碎片（可能是一个词或一个字）。

flush=True：强制 Python 立即把字符打印到屏幕上，不要放在缓冲区里攒着。

yield：这个关键字把函数变成了一个 “生成器”。它不会一次性 return 结果，而是每拿到一个 chunk 就向外“吐”一个，这样调用这个方法的人也能实时拿到数据。