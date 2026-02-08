import json
import os
import urllib.request
from typing import Iterator, Optional
from openai import OpenAI
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    """
    一个自定义的LLM客户端，通过继承增加了对ModelScope的支持。
    """
    def __init__(
            self,
            model:Optional[str] = None,
            api_key:Optional[str] = None,
            base_url:Optional[str] = None,
            provider:Optional[str] = "auto",
            **kwargs
    ):
        if provider in {"modelscope", "ollama"}:
            print(f"使用自定义的 {provider} provider")
            self.provider = provider
            # 解析凭证
            if provider == "ollama":
                self.api_key = api_key or os.getenv("LLM_API_KEY") or "ollama"
                self.base_url = base_url or os.getenv("LLM_BASE_URL") or "http://127.0.0.1:11434/v1"
                # Ensure local requests bypass any proxy settings.
                no_proxy = os.getenv("NO_PROXY", "")
                localhost_no_proxy = "127.0.0.1,localhost"
                if localhost_no_proxy not in no_proxy:
                    os.environ["NO_PROXY"] = f"{no_proxy},{localhost_no_proxy}".strip(",")
            else:
                self.api_key = api_key or os.getenv("LLM_API_KEY")
                self.base_url = base_url or os.getenv("LLM_BASE_URL")

            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("请提供LLM_API_KEY环境变量或api_key参数")

            # 设置默认模型和其他参数
            default_model = "llama3" if provider == "ollama" else "gemini-3-flash-preview"
            self.model = model or os.getenv("LLM_MODEL_ID", default_model)
            self.timeout = int(os.getenv("LLM_TIMEOUT", "60"))
            self.max_tokens = kwargs.get("max_tokens")
            self.temperature = kwargs.get("temperature", 0.7)

            # 使用获取的参数创建OpenAI客户端实例
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            # Match base-class attribute naming for downstream usage.
            self._client = self.client
        else:
            super().__init__(model, api_key, base_url, provider, **kwargs)

    def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
            """
            [修复版] 调用大语言模型进行思考，并增加对空响应包的安全检查。
            """
            print(f"正在通过自定义客户端调用 {self.model} 模型...")
            if getattr(self, "provider", None) == "ollama":
                payload = {
                    "model": self.model,
                    "messages": messages,
                }
                resolved_temperature = temperature if temperature is not None else getattr(self, "temperature", None)
                if resolved_temperature is not None:
                    payload["temperature"] = resolved_temperature
                if self.max_tokens is not None:
                    payload["max_tokens"] = self.max_tokens

                try:
                    base_url = self.base_url.rstrip("/")
                    if not base_url.endswith("/v1"):
                        base_url = f"{base_url}/v1"

                    def request_json(url: str) -> dict:
                        req = urllib.request.Request(
                            url,
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                            return json.loads(resp.read().decode("utf-8"))

                    url = f"{base_url}/chat/completions"
                    try:
                        data = request_json(url)
                    except urllib.error.HTTPError as http_error:
                        if http_error.code == 404 and "localhost" in base_url:
                            fallback_base = base_url.replace("localhost", "127.0.0.1")
                            data = request_json(f"{fallback_base}/chat/completions")
                        else:
                            raise

                    content = ""
                    choices = data.get("choices") or []
                    if choices and "message" in choices[0]:
                        content = choices[0]["message"].get("content") or ""

                    if content:
                        print(content, end="", flush=True)
                        yield content
                        print()
                    return
                except Exception as e:
                    print(f"调用LLM API时发生错误: {e}")
                    exception_cls = getattr(HelloAgentsLLM, "HelloAgentsException", RuntimeError)
                    raise exception_cls(f"LLM调用失败: {str(e)}") from e

            stream_mode = getattr(self, "provider", None) != "ollama"
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "stream": stream_mode,
            }
            resolved_temperature = temperature if temperature is not None else getattr(self, "temperature", None)
            if resolved_temperature is not None:
                request_kwargs["temperature"] = resolved_temperature
            if self.max_tokens is not None:
                request_kwargs["max_tokens"] = self.max_tokens

            try:
                # 确保 client 已经正确初始化
                response = self._client.chat.completions.create(**request_kwargs)

                print("大语言模型响应成功:")
                if stream_mode:
                    for chunk in response:
                        # --- 核心修复点：增加安全防御 ---
                        # 检查 chunk 是否有效，且包含 choices 列表且不为空
                        if not hasattr(chunk, 'choices') or not chunk.choices:
                            continue
                        
                        # 安全获取 content
                        # 使用 getattr 进一步防止某些特殊的 delta 对象缺失 content 属性
                        delta = chunk.choices[0].delta
                        content = getattr(delta, 'content', "") or ""
                        
                        if content:
                            print(content, end="", flush=True)
                            yield content
                else:
                    if response.choices:
                        content = response.choices[0].message.content or ""
                        if content:
                            print(content, end="", flush=True)
                            yield content

                print()  # 在流式输出结束后换行

            except Exception as e:
                if getattr(self, "provider", None) == "ollama" and stream_mode:
                    try:
                        fallback_kwargs = dict(request_kwargs)
                        fallback_kwargs["stream"] = False
                        fallback_response = self._client.chat.completions.create(**fallback_kwargs)
                        if fallback_response.choices:
                            content = fallback_response.choices[0].message.content or ""
                            if content:
                                print(content, end="", flush=True)
                                yield content
                                print()
                                return
                    except Exception:
                        pass

                # 这里的 e 会捕获到 404, 401 等详细的 API 错误
                print(f"调用LLM API时发生错误: {e}")
                exception_cls = getattr(HelloAgentsLLM, "HelloAgentsException", RuntimeError)
                raise exception_cls(f"LLM调用失败: {str(e)}") from e