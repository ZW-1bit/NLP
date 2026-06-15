"""DeepSeek API 调用模块"""

from openai import OpenAI
from config import API_KEY, BASE_URL, MODEL

# 复用客户端，避免每次请求重建连接
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def chat(
    msgs: list[dict],
    model: str = MODEL,
    temp: float = 0.7,
    max_tok: int = 1024,
    stream: bool = False,
):
    """
    调用DeepSeek对话接口

    Args:
        msgs: 消息列表 [{"role": "user", "content": "..."}]
        model: 模型名称
        temp: 温度参数
        max_tok: 最大token数
        stream: 是否流式输出

    Returns:
        流式: 生成器; 非流式: 完整回复文本
    """
    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=temp,
        max_tokens=max_tok,
        stream=stream,
    )
    if stream:
        return _stream_resp(resp)
    return resp.choices[0].message.content


def _stream_resp(resp):
    """流式响应生成器"""
    for chunk in resp:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
