"""问答核心逻辑 - 多会话管理、在线/离线模式"""

import dsapi
import local_qa
import db

MAX_TURNS = 10


def build_msgs(sid: str, question: str) -> list[dict]:
    """根据会话ID构建完整消息列表"""
    info = db.get_session(sid)
    prompt = info["prompt"] if info and info["prompt"] else "你是一个智能问答助手，请用中文准确、有条理地回答问题。"
    msgs = [{"role": "system", "content": prompt}]
    history = db.get_msgs(sid)
    recent = history[-MAX_TURNS * 2:]
    for m in recent:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": question})
    return msgs


def ask(sid: str, question: str, mode: str = "online", **kwargs) -> str:
    """提问并获取完整回复，mode: online/offline"""
    if mode == "offline":
        answer = local_qa.local_answer(question)
        db.add_msg(sid, "user", question)
        db.add_msg(sid, "assistant", answer)
        return answer
    info = db.get_session(sid)
    if info and info["model"]:
        kwargs.setdefault("model", info["model"])
    msgs = build_msgs(sid, question)
    answer = dsapi.chat(msgs, **kwargs)
    db.add_msg(sid, "user", question)
    db.add_msg(sid, "assistant", answer)
    return answer


def ask_stream(sid: str, question: str, mode: str = "online", **kwargs):
    """提问并流式获取回复，mode: online/offline"""
    if mode == "offline":
        answer = local_qa.local_answer(question)
        db.add_msg(sid, "user", question)
        db.add_msg(sid, "assistant", answer)
        # 离线模式模拟逐字输出
        for i in range(0, len(answer), 3):
            yield answer[i:i+3]
        return
    info = db.get_session(sid)
    if info and info["model"]:
        kwargs.setdefault("model", info["model"])
    msgs = build_msgs(sid, question)
    kwargs["stream"] = True
    full = []
    for chunk in dsapi.chat(msgs, **kwargs):
        full.append(chunk)
        yield chunk
    db.add_msg(sid, "user", question)
    db.add_msg(sid, "assistant", "".join(full))
