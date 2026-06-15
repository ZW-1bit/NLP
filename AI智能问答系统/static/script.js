/* 全局状态 */
let curSid = null;
let sending = false;
let curMode = 'online'; // online / offline

/* 辅助函数 */
async function autoTitle(sid) {
    try {
        const res = await fetch(`/api/sessions/${sid}/title`, { method: 'POST' });
        const data = await res.json();
        if (data.title) loadSessions();
    } catch {}
}

/* DOM */
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

const elList = $('#session-list');
const elWelcome = $('#welcome');
const elChat = $('#chat-area');
const elMsgs = $('#messages');
const elInput = $('#msg-input');
const elSend = $('#btn-send');
const elModal = $('#modal');
const elFile = $('#file-input');

/* ---- 会话管理 ---- */

async function loadSessions() {
    const res = await fetch('/api/sessions');
    const data = await res.json();
    elList.innerHTML = '';
    data.forEach(s => {
        const li = document.createElement('li');
        li.dataset.id = s.id;
        if (s.id === curSid) li.className = 'active';
        li.innerHTML = `<span>${esc(s.title)}</span><button class="del" title="删除">&times;</button>`;
        li.querySelector('span').onclick = () => switchSession(s.id);
        li.querySelector('.del').onclick = e => { e.stopPropagation(); deleteSession(s.id); };
        elList.appendChild(li);
    });
}

async function createSession(title) {
    const res = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: title || '新对话' }),
    });
    const data = await res.json();
    await loadSessions();
    switchSession(data.id);
    return data.id;
}

async function switchSession(sid) {
    curSid = sid;
    // 高亮侧边栏
    $$('#session-list li').forEach(li => li.classList.toggle('active', li.dataset.id === sid));
    // 加载消息
    const res = await fetch(`/api/sessions/${sid}`);
    const data = await res.json();
    elWelcome.classList.add('hidden');
    elChat.classList.remove('hidden');
    elMsgs.innerHTML = '';
    (data.messages || []).forEach(m => appendBubble(m.role, m.content));
    scrollBottom();
    // 同步设置
    $('#set-prompt').value = data.prompt || '';
    $('#set-model').value = data.model || 'deepseek-chat';
}

async function deleteSession(sid) {
    await fetch(`/api/sessions/${sid}`, { method: 'DELETE' });
    if (curSid === sid) {
        curSid = null;
        elWelcome.classList.remove('hidden');
        elChat.classList.add('hidden');
    }
    loadSessions();
}

/* ---- 聊天 ---- */

function appendBubble(role, content) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.innerHTML = `<div class="bubble">${renderMd(content)}</div>`;
    elMsgs.appendChild(div);
    return div;
}

function renderMd(text) {
    try { return marked.parse(text); }
    catch { return esc(text).replace(/\n/g, '<br>'); }
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function scrollBottom() {
    elChat.scrollTop = elChat.scrollHeight;
}

async function sendMessage(text) {
    if (sending || !text.trim()) return;
    // 确保有会话
    if (!curSid) {
        try {
            await createSession();
        } catch (e) {
            alert('创建会话失败，请刷新页面重试');
            return;
        }
    }
    if (!curSid) {
        alert('会话创建异常，请刷新页面');
        return;
    }
    sending = true;
    elSend.disabled = true;
    const q = text.trim();
    elInput.value = '';

    // 用户气泡
    appendBubble('user', q);
    // AI气泡（流式）
    const aiDiv = document.createElement('div');
    aiDiv.className = 'msg assistant';
    const bubble = document.createElement('div');
    bubble.className = 'bubble typing';
    aiDiv.appendChild(bubble);
    elMsgs.appendChild(aiDiv);
    scrollBottom();

    const temp = parseFloat($('#set-temp').value);
    const maxTok = parseInt($('#set-tok').value);

    try {
        const res = await fetch(`/api/chat/${curSid}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: q, mode: curMode, temp, max_tok: maxTok }),
        });
        if (!res.ok) {
            throw new Error(`服务器错误: ${res.status}`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let full = '';
        let buf = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data:')) continue;
                const payload = line.slice(5).trim();
                if (payload === '[DONE]') break;
                try {
                    const obj = JSON.parse(payload);
                    full += obj.c;
                    bubble.innerHTML = renderMd(full);
                    scrollBottom();
                } catch {}
            }
        }
        bubble.classList.remove('typing');
        if (!full) {
            bubble.innerHTML = '<span style="color:#ff9800">未收到回复，请重试或切换模式。</span>';
        }
        // 首次对话后自动总结标题
        const cnt = elMsgs.querySelectorAll('.msg.user').length;
        if (cnt === 1) {
            autoTitle(curSid);
        }
    } catch (e) {
        bubble.classList.remove('typing');
        bubble.innerHTML = `<span style="color:#ff6b6b">请求失败: ${e.message}</span>`;
    }
    sending = false;
    elSend.disabled = false;
    elInput.focus();
}

/* ---- 文档上传 ---- */

async function uploadFile(file) {
    if (!curSid) { alert('请先创建对话'); return; }
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(`/api/upload/${curSid}`, { method: 'POST', body: fd });
    const data = await res.json();
    if (data.ok) {
        appendBubble('assistant', `已加载文档 (${data.chars} 字符)，现在可以基于文档内容提问了。`);
        scrollBottom();
    }
}

/* ---- 设置 ---- */

function openSettings() {
    if (!curSid) { alert('请先创建对话'); return; }
    elModal.classList.remove('hidden');
}

async function saveSettings() {
    await fetch(`/api/sessions/${curSid}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            prompt: $('#set-prompt').value,
            model: $('#set-model').value,
        }),
    });
    elModal.classList.add('hidden');
}

/* ---- 事件绑定 ---- */

// 模式切换
$('#btn-mode').onclick = () => {
    curMode = curMode === 'online' ? 'offline' : 'online';
    const btn = $('#btn-mode');
    btn.classList.toggle('online', curMode === 'online');
    btn.classList.toggle('offline', curMode === 'offline');
    btn.querySelector('.mode-text').textContent = curMode === 'online' ? '在线' : '离线';
    elInput.placeholder = curMode === 'online' ? '输入你的问题...' : '离线模式 - 简单问答、计算、常识...';
};

$('#btn-new').onclick = () => createSession();
$('#btn-settings').onclick = openSettings;
$('#btn-save').onclick = saveSettings;
$('#btn-cancel').onclick = () => elModal.classList.add('hidden');

elSend.onclick = () => sendMessage(elInput.value);
elInput.onkeydown = e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(elInput.value);
    }
};

// 输入时自动创建会话
elInput.oninput = async () => {
    if (!curSid && elInput.value.trim()) {
        await createSession();
    }
};

elFile.onchange = e => {
    if (e.target.files[0]) uploadFile(e.target.files[0]);
    e.target.value = '';
};

// 快捷问题
$$('.q-btn').forEach(btn => {
    btn.onclick = () => sendMessage(btn.dataset.q);
});

// 滑块实时值
$('#set-temp').oninput = e => $('#temp-val').textContent = e.target.value;
$('#set-tok').oninput = e => $('#tok-val').textContent = e.target.value;

/* ---- 语音识别 ---- */
const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
const elMic = $('#btn-mic');
let micRec = null;
let isRecording = false;

if (SpeechRec) {
    elMic.style.display = 'flex';
    micRec = new SpeechRec();
    micRec.lang = 'zh-CN';
    micRec.interimResults = true;
    micRec.continuous = false;

    micRec.onresult = e => {
        let text = '';
        for (let i = 0; i < e.results.length; i++) {
            text += e.results[i][0].transcript;
        }
        elInput.value = text;
    };

    micRec.onend = () => {
        isRecording = false;
        elMic.classList.remove('recording');
        // 如果有内容则自动发送
        if (elInput.value.trim()) sendMessage(elInput.value);
    };

    micRec.onerror = () => {
        isRecording = false;
        elMic.classList.remove('recording');
    };

    elMic.onclick = () => {
        if (isRecording) {
            micRec.stop();
        } else {
            isRecording = true;
            elMic.classList.add('recording');
            elInput.value = '';
            elInput.placeholder = '正在听...';
            micRec.start();
        }
    };
}

// 初始化
loadSessions();
