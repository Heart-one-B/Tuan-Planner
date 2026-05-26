from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.web.session_service import WorkflowSessionService


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Existing session id, if any")
    user_input: str | None = Field(default=None, description="New user request or updated query")
    runtime_origin_area: str | None = Field(default=None, description="User location / origin area")
    location_permission_granted: bool | None = Field(default=None, description="Location permission response")
    clarification_response: str | None = Field(default=None, description="Clarification answer")
    user_confirmed: bool | None = Field(default=None, description="Final confirmation answer")


class ChatResponse(BaseModel):
    session_id: str
    status: str
    ui: dict[str, Any]
    state: dict[str, Any]
    display_text: str = ""
    final_message: str = ""
    llm_answer: str = ""
    execution_result: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


service = WorkflowSessionService()
app = FastAPI(title="Local Life Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def render_frontend_page() -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Local Life Agent Console</title>
    <style>
        :root {{ color-scheme: dark; --bg: #0b1020; --panel: #121a33; --panel-2: #182343; --text: #e9eefc; --muted: #9aa7c7; --accent: #7c9cff; --border: rgba(255,255,255,.10); --danger: #ff7c7c; --ok: #65d6a5; }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: radial-gradient(circle at top, #1a2550, var(--bg) 48%); color: var(--text); }}
        .shell {{ max-width: 1280px; margin: 0 auto; padding: 28px 18px 40px; }}
        .hero {{ display: grid; grid-template-columns: 1fr auto; gap: 16px; align-items: start; margin-bottom: 18px; }}
        .eyebrow {{ color: var(--accent); text-transform: uppercase; letter-spacing: .14em; font-size: 12px; margin-bottom: 8px; }}
        h1 {{ margin: 0 0 8px; font-size: 30px; }}
        p {{ margin: 0; color: var(--muted); line-height: 1.7; }}
        .status-card, .panel {{ background: rgba(18,26,51,.92); border: 1px solid var(--border); border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,.22); }}
        .status-card {{ padding: 16px 18px; min-width: 240px; }}
        .status-pill {{ display: inline-flex; padding: 6px 12px; border-radius: 999px; font-size: 12px; margin-bottom: 12px; background: rgba(124,156,255,.16); color: #cfd9ff; }}
        .grid {{ display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr) minmax(0, 1fr); gap: 16px; }}
        .panel {{ padding: 18px; }}
        .panel h2 {{ margin: 0 0 14px; font-size: 18px; }}
        .stack {{ display: grid; gap: 14px; }}
        .field {{ display: grid; gap: 8px; }}
        .field span, .mini-card span, .output-title, .prompt-label {{ color: var(--muted); font-size: 13px; }}
        textarea, input {{ width: 100%; border-radius: 14px; border: 1px solid var(--border); background: rgba(255,255,255,.04); color: var(--text); padding: 12px 14px; font: inherit; outline: none; }}
        textarea {{ resize: vertical; min-height: 110px; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
        .action-row, .option-row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        button {{ border: 0; border-radius: 999px; padding: 11px 16px; color: var(--text); font: inherit; cursor: pointer; transition: transform .15s ease, opacity .15s ease, background .15s ease; }}
        button:hover {{ transform: translateY(-1px); }}
        button:disabled {{ opacity: .55; cursor: not-allowed; transform: none; }}
        .primary-btn {{ background: linear-gradient(135deg, #7c9cff, #6a7dff); }}
        .ghost-btn {{ background: rgba(255,255,255,.06); border: 1px solid var(--border); }}
        .mini-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }}
        .mini-card {{ border: 1px solid var(--border); border-radius: 16px; padding: 14px; background: rgba(255,255,255,.03); }}
        .mini-card strong {{ display: block; margin-top: 6px; font-size: 16px; }}
        .prompt-box, .output-card, .error-box, .empty-state {{ border: 1px solid var(--border); border-radius: 18px; background: rgba(255,255,255,.03); padding: 14px; }}
        .output-stack {{ display: grid; gap: 12px; }}
        pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; color: #e8ecff; font: inherit; line-height: 1.65; }}
        .error-banner {{ margin-top: 12px; color: #ffd0d0; background: rgba(255,124,124,.12); border: 1px solid rgba(255,124,124,.25); padding: 12px 14px; border-radius: 14px; }}
        .session-line {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
        .ok {{ background: rgba(101,214,165,.16); color: #baf3d7; }}
        .running {{ background: rgba(124,156,255,.16); color: #cfd9ff; }}
        .warning {{ background: rgba(255,199,95,.16); color: #ffe5ad; }}
        @media (max-width: 1100px) {{ .hero, .grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="shell">
        <header class="hero">
            <div>
                <div class="eyebrow">Local Life Agent Console</div>
                <h1>本地生活 Agent 标准版前端</h1>
                <p>这个页面直接由后端提供，访问 <code>/</code> 就能完成输入、澄清、确认和结果查看。</p>
            </div>
            <div class="status-card">
                <span id="status-pill" class="status-pill running">idle</span>
                <div class="session-line">Session: <span id="session-id">未开始</span></div>
                <div class="session-line">Pending: <span id="pending-step">无</span></div>
            </div>
        </header>

        <main class="grid">
            <section class="panel">
                <h2>1. 发起需求</h2>
                <form id="request-form" class="stack">
                    <label class="field"><span>需求输入</span><textarea id="user-input">今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。</textarea></label>
                    <div class="two-col">
                        <label class="field"><span>起点区域（可选）</span><input id="origin-area" placeholder="例如：望京、朝阳、国贸" /></label>
                        <label class="field"><span>会话 ID</span><input id="session-input" readonly placeholder="提交后自动生成" /></label>
                    </div>
                    <div class="action-row">
                        <button class="primary-btn" id="start-btn" type="submit">开始规划</button>
                        <button class="ghost-btn" id="reset-btn" type="button">新会话</button>
                    </div>
                </form>
                <div class="mini-grid">
                    <div class="mini-card"><span>提示状态</span><strong id="mini-step">无</strong></div>
                    <div class="mini-card"><span>提示类型</span><strong id="mini-kind">无</strong></div>
                </div>
            </section>

            <section class="panel">
                <h2>2. 当前交互</h2>
                <div id="interaction-empty" class="empty-state">在这里会显示定位授权、澄清问题或确认执行按钮。</div>
                <div id="interaction-box" class="stack" style="display:none">
                    <div class="prompt-box">
                        <div class="prompt-label">系统提示</div>
                        <p id="pending-prompt"></p>
                        <div id="option-row" class="option-row"></div>
                    </div>
                    <form id="reply-form" class="stack">
                        <label class="field"><span id="reply-label">补充说明</span><textarea id="reply-input" rows="4" placeholder="请输入补充内容"></textarea></label>
                        <button class="primary-btn" id="reply-btn" type="submit">提交补充信息</button>
                    </form>
                </div>
            </section>

            <section class="panel">
                <h2>3. 输出面板</h2>
                <div class="output-stack">
                    <article class="output-card"><div class="output-title">最终展示文本</div><pre id="summary-text">等待开始</pre></article>
                    <article class="output-card"><div class="output-title">执行结果 / 状态</div><pre id="execution-result">{{}}</pre></article>
                    <article class="output-card"><div class="output-title">原始状态</div><pre id="state-text">{{}}</pre></article>
                </div>
                <div id="error-box" class="error-banner" style="display:none"></div>
            </section>
        </main>
    </div>

    <script>
        const API_BASE = '';
        const DEFAULT_REQUEST = '今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。';
        const state = {{ sessionId: '', response: null, loading: false, error: '' }};

        const el = (id) => document.getElementById(id);

        function prettify(value) {{
            if (typeof value === 'string') return value;
            try {{ return JSON.stringify(value, null, 2); }} catch {{ return String(value); }}
        }}

        function getPending() {{
            const response = state.response;
            return {{
                step: response?.ui?.step || response?.state?.pending_action || '',
                prompt: response?.ui?.prompt || '',
                kind: response?.ui?.kind || '',
                options: response?.ui?.options || [],
            }};
        }}

        function setLoading(loading) {{
            state.loading = loading;
            el('start-btn').disabled = loading;
            el('reply-btn').disabled = loading;
            el('start-btn').textContent = loading ? '处理中...' : '开始规划';
        }}

        function render() {{
            const pending = getPending();
            const response = state.response;

            el('session-id').textContent = state.sessionId || '未开始';
            el('session-input').value = state.sessionId || '';
            el('status-pill').textContent = response?.status || 'idle';
            el('status-pill').className = `status-pill ${{response?.status || 'running'}}`;
            el('pending-step').textContent = pending.step || '无';
            el('mini-step').textContent = pending.step || '无';
            el('mini-kind').textContent = pending.kind || '无';

            el('summary-text').textContent = response?.display_text || response?.final_message || response?.llm_answer || '当前没有可展示的最终文本。';
            el('execution-result').textContent = prettify(response?.execution_result || {{}});
            el('state-text').textContent = prettify(response?.state || {{}});

            const interactionEmpty = el('interaction-empty');
            const interactionBox = el('interaction-box');
            const pendingPrompt = el('pending-prompt');
            const optionRow = el('option-row');
            const replyForm = el('reply-form');
            const replyLabel = el('reply-label');
            const replyInput = el('reply-input');

            optionRow.innerHTML = '';
            if (pending.step) {{
                interactionEmpty.style.display = 'none';
                interactionBox.style.display = 'grid';
                pendingPrompt.textContent = pending.prompt || '请继续';
                if (pending.kind === 'confirm') {{
                    replyForm.style.display = 'none';
                }} else {{
                    replyForm.style.display = 'grid';
                    replyLabel.textContent = pending.step === 'location_fallback' ? '补充城市 / 区域' : '补充说明';
                    replyInput.placeholder = pending.step === 'location_fallback' ? '例如：北京朝阳' : '请输入补充内容';
                }}

                for (const option of pending.options) {{
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = 'ghost-btn';
                    btn.textContent = option.label;
                    btn.addEventListener('click', () => {{
                        if (pending.step === 'location_permission') {{
                            sendPayload({{ location_permission_granted: Boolean(option.value) }});
                        }} else if (pending.step === 'confirmation') {{
                            sendPayload({{ user_confirmed: Boolean(option.value) }});
                        }}
                    }});
                    optionRow.appendChild(btn);
                }}
            }} else {{
                interactionEmpty.style.display = 'block';
                interactionBox.style.display = 'none';
            }}

            const errorBox = el('error-box');
            if (state.error) {{
                errorBox.style.display = 'block';
                errorBox.textContent = state.error;
            }} else {{
                errorBox.style.display = 'none';
                errorBox.textContent = '';
            }}
        }}

        async function sendPayload(payload) {{
            setLoading(true);
            state.error = '';
            try {{
                const res = await fetch(`${{API_BASE}}/api/chat`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ session_id: state.sessionId || undefined, ...payload }}),
                }});
                if (!res.ok) {{
                    throw new Error(await res.text() || `请求失败: ${{res.status}}`);
                }}
                const data = await res.json();
                state.sessionId = data.session_id;
                state.response = data;
            }} catch (error) {{
                state.error = error instanceof Error ? error.message : '请求失败';
            }} finally {{
                setLoading(false);
                render();
            }}
        }}

        el('request-form').addEventListener('submit', (event) => {{
            event.preventDefault();
            sendPayload({{ user_input: el('user-input').value, runtime_origin_area: el('origin-area').value || undefined }});
        }});

        el('reply-form').addEventListener('submit', (event) => {{
            event.preventDefault();
            const pending = getPending();
            const reply = el('reply-input').value.trim();
            if (!pending.step || !reply) return;
            if (pending.step === 'location_fallback') {{
                sendPayload({{ runtime_origin_area: reply }});
            }} else if (pending.step === 'clarification') {{
                sendPayload({{ clarification_response: reply }});
            }}
        }});

        el('reset-btn').addEventListener('click', () => {{
            state.sessionId = '';
            state.response = null;
            state.error = '';
            localStorage.removeItem('local-life-agent-session-id');
            el('reply-input').value = '';
            render();
        }});

        (function init() {{
            const cached = localStorage.getItem('local-life-agent-session-id');
            if (cached) state.sessionId = cached;
            if (state.sessionId) el('session-input').value = state.sessionId;
            el('user-input').value = DEFAULT_REQUEST;
            render();
        }})();
    </script>
</body>
</html>"""


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(render_frontend_page())


@app.get("/api/sessions/{session_id}", response_model=ChatResponse)
def get_session(session_id: str) -> ChatResponse:
    snapshot = service.snapshot(session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return ChatResponse(**snapshot)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.session_id and not request.user_input:
        raise HTTPException(status_code=400, detail="user_input is required for a new session")

    session_id, state = service.run_turn(
        session_id=request.session_id,
        user_input=request.user_input,
        runtime_origin_area=request.runtime_origin_area,
        location_permission_granted=request.location_permission_granted,
        clarification_response=request.clarification_response,
        user_confirmed=request.user_confirmed,
    )
    return ChatResponse(**service.format_response(session_id, state))
