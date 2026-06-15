"""SQLite 数据库模块 - 会话与消息持久化"""

import sqlite3
import uuid
from datetime import datetime

DB_PATH = "qa_data.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    conn = _conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS session (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            prompt TEXT DEFAULT '',
            model TEXT DEFAULT 'deepseek-chat',
            created TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS message (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sid TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            ts TEXT NOT NULL,
            FOREIGN KEY (sid) REFERENCES session(id)
        );
    """)
    conn.close()


def new_session(title: str = "新对话", prompt: str = "", model: str = "deepseek-chat") -> str:
    """创建新会话，返回会话ID"""
    sid = uuid.uuid4().hex[:12]
    conn = _conn()
    conn.execute(
        "INSERT INTO session (id, title, prompt, model, created) VALUES (?,?,?,?,?)",
        (sid, title, prompt, model, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return sid


def list_sessions() -> list[dict]:
    """获取所有会话列表"""
    conn = _conn()
    rows = conn.execute(
        "SELECT id, title, prompt, model, created FROM session ORDER BY created DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session(sid: str) -> dict | None:
    """获取单个会话信息"""
    conn = _conn()
    row = conn.execute(
        "SELECT id, title, prompt, model, created FROM session WHERE id=?", (sid,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def del_session(sid: str):
    """删除会话及其消息"""
    conn = _conn()
    conn.execute("DELETE FROM message WHERE sid=?", (sid,))
    conn.execute("DELETE FROM session WHERE id=?", (sid,))
    conn.commit()
    conn.close()


def update_session(sid: str, title: str = None, prompt: str = None, model: str = None):
    """更新会话信息"""
    conn = _conn()
    s = get_session(sid)
    if not s:
        conn.close()
        return
    conn.execute(
        "UPDATE session SET title=?, prompt=?, model=? WHERE id=?",
        (
            title if title is not None else s["title"],
            prompt if prompt is not None else s["prompt"],
            model if model is not None else s["model"],
            sid,
        ),
    )
    conn.commit()
    conn.close()


def add_msg(sid: str, role: str, content: str):
    """添加一条消息"""
    conn = _conn()
    conn.execute(
        "INSERT INTO message (sid, role, content, ts) VALUES (?,?,?,?)",
        (sid, role, content, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_msgs(sid: str) -> list[dict]:
    """获取会话的所有消息"""
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content, ts FROM message WHERE sid=? ORDER BY id", (sid,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_msgs(sid: str) -> int:
    """获取会话消息数"""
    conn = _conn()
    n = conn.execute("SELECT COUNT(*) FROM message WHERE sid=?", (sid,)).fetchone()[0]
    conn.close()
    return n
