"""Flask 应用 - API路由与页面服务"""
import json
from flask import Flask, request, Response, render_template, jsonify
import db
import qa_core
import dsapi

app = Flask(__name__)
db.init_db()


# ---------- 页面 ----------

@app.route("/")
def index():
    return render_template("index.html")


# ---------- 会话管理 API ----------

@app.route("/api/sessions", methods=["GET"])
def api_list():
    return jsonify(db.list_sessions())


@app.route("/api/sessions", methods=["POST"])
def api_create():
    d = request.json or {}
    sid = db.new_session(
        title=d.get("title", "新对话"),
        prompt=d.get("prompt", ""),
        model=d.get("model", "deepseek-chat"),
    )
    return jsonify({"id": sid})


@app.route("/api/sessions/<sid>", methods=["GET"])
def api_get(sid):
    info = db.get_session(sid)
    if not info:
        return jsonify({"error": "not found"}), 404
    info["messages"] = db.get_msgs(sid)
    return jsonify(info)


@app.route("/api/sessions/<sid>", methods=["PUT"])
def api_update(sid):
    d = request.json or {}
    db.update_session(sid, title=d.get("title"), prompt=d.get("prompt"), model=d.get("model"))
    return jsonify({"ok": True})


@app.route("/api/sessions/<sid>", methods=["DELETE"])
def api_delete(sid):
    db.del_session(sid)
    return jsonify({"ok": True})


@app.route("/api/sessions/<sid>/title", methods=["POST"])
def api_gen_title(sid):
    """根据会话内容自动生成标题"""
    msgs = db.get_msgs(sid)
    if not msgs:
        return jsonify({"title": "新对话"})
    # 在线模式：用DeepSeek总结
    try:
        summary_msgs = [
            {"role": "system", "content": "请用不超过10个字概括以下对话的主题，只输出标题，不要标点符号。"},
        ]
        for m in msgs[:4]:
            summary_msgs.append({"role": m["role"], "content": m["content"][:200]})
        title = dsapi.chat(summary_msgs, max_tokens=30, temp=0.3)
        title = title.strip().strip('"\'""''')
        if len(title) > 20:
            title = title[:20]
    except Exception:
        # 离线/失败时用首条消息生成
        q = msgs[0]["content"] if msgs else "新对话"
        title = _simple_title(q)
    db.update_session(sid, title=title)
    return jsonify({"title": title})


def _simple_title(text: str) -> str:
    """离线模式：简单提取标题"""
    prefixes = ['计算', '算一下', '算', '翻译', '翻译一下', '介绍一下', '解释一下', '什么是', '为什么', '怎么样']
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()
            break
    import re
    text = re.sub(r'[，,。.？?！!、\\/]', '', text)
    return text[:20] or '新对话'


# ---------- 问答 API ----------

@app.route("/api/chat/<sid>", methods=["POST"])
def api_chat(sid):
    d = request.json or {}
    question = d.get("message", "").strip()
    if not question:
        return jsonify({"error": "empty message"}), 400

    mode = d.get("mode", "online")
    temp = d.get("temp", 0.7)
    max_tok = d.get("max_tok", 1024)

    def generate():
        for chunk in qa_core.ask_stream(sid, question, mode=mode, temp=temp, max_tok=max_tok):
            yield f"data:{json.dumps({'c': chunk}, ensure_ascii=False)}\n\n"
        yield "data:[DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ---------- 文档上传 API ----------

@app.route("/api/upload/<sid>", methods=["POST"])
def api_upload(sid):
    """上传文档内容作为上下文注入"""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    content = f.read().decode("utf-8", errors="ignore")[:8000]
    # 将文档内容作为系统提示的一部分
    info = db.get_session(sid)
    old_prompt = info["prompt"] if info and info["prompt"] else ""
    new_prompt = old_prompt + f"\n\n参考文档:\n{content}" if old_prompt else f"参考文档:\n{content}"
    db.update_session(sid, prompt=new_prompt)
    return jsonify({"ok": True, "chars": len(content)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)
