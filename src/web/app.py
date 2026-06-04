from __future__ import annotations

import html
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.graph.nodes.execution_node import execution_node
from src.graph.workflow import build_workflow
from src.graph.nodes.confirmation_node import _update_preference_document

app = FastAPI(title="Local Life Planner", version="0.1.0")
workflow_app = build_workflow(MemorySaver())


class CreateSessionRequest(BaseModel):
    user_input: str = Field(min_length=1, description="用户输入的规划需求")


class ClarifySessionRequest(BaseModel):
    user_reply: str = Field(min_length=1, description="针对系统追问的回复")


class ConfirmSessionRequest(BaseModel):
    confirmed: bool = Field(description="是否确认执行当前方案")


SESSIONS: dict[str, dict[str, Any]] = {}


def _thread_config(session_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": session_id}}


def _create_initial_state(user_input: str) -> dict[str, Any]:
    return {
        "user_input": user_input,
        "runtime_origin_area": "",
        "conversation_turns": [user_input],
        "clarification_round": 0,
        "errors": [],
        "web_preview_mode": True,
    }


def _capture_latest_state(session_id: str) -> dict[str, Any]:
    snapshot = workflow_app.get_state(_thread_config(session_id))
    values = dict(snapshot.values or {}) if snapshot else {}
    values.setdefault("web_preview_mode", True)
    return values


def _status_from_state(state: dict[str, Any]) -> str:
  if state.get("execution_result"):
    return "completed"
  if state.get("final_message") or state.get("llm_answer"):
    return "completed"
  if state.get("pending_clarification"):
    return "awaiting_clarification"
  if state.get("pending_confirmation"):
    return "awaiting_confirmation"
  return "running"


def _store_session(session_id: str, state: dict[str, Any]) -> dict[str, Any]:
    session = {
        "session_id": session_id,
        "status": _status_from_state(state),
        "state": state,
        "display_text": state.get("display_text") or "",
        "plan": state.get("plan") or {},
        "pending_clarification": state.get("pending_clarification") or "",
        "pending_confirmation": state.get("pending_confirmation") or {},
        "final_message": state.get("final_message") or "",
        "execution_result": state.get("execution_result") or {},
    }
    SESSIONS[session_id] = session
    return session


def _run_planning(session_id: str, user_input: str) -> dict[str, Any]:
    workflow_app.invoke(_create_initial_state(user_input), config=_thread_config(session_id))
    state = _capture_latest_state(session_id)
    return _store_session(session_id, state)


def _resume_with_reply(session_id: str, user_reply: str) -> dict[str, Any]:
    workflow_app.invoke(
        Command(resume=None, update={"user_reply": user_reply}),
        config=_thread_config(session_id),
    )
    state = _capture_latest_state(session_id)
    return _store_session(session_id, state)


def _run_confirmation(session_id: str, confirmed: bool) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    state = dict(session.get("state") or {})
    state["user_confirmed"] = confirmed
    state["web_preview_mode"] = False

    # ── 新增：确认时同步触发偏好文档更新（补全原 confirmation_node 中的逻辑）──
    if confirmed:
        try:
            _update_preference_document(state)
        except Exception as exc:
            print(f"[Web Confirm] 用户偏好文档更新失败：{exc}")

    execution_state = execution_node(state)
    merged_state = dict(state)
    merged_state.update(execution_state)
    merged_state["web_preview_mode"] = False
    merged_state["pending_confirmation"] = {}
    return _store_session(session_id, merged_state)


def _render_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>本地生活规划前端</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: rgba(15, 23, 42, 0.82);
      --panel-strong: rgba(15, 23, 42, 0.96);
      --line: rgba(148, 163, 184, 0.22);
      --text: #e5eefc;
      --muted: #94a3b8;
      --accent: #f5b942;
      --accent-2: #6ee7b7;
      --danger: #fb7185;
      --shadow: 0 32px 80px rgba(2, 6, 23, 0.5);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(245, 185, 66, 0.22), transparent 28%),
        radial-gradient(circle at top right, rgba(110, 231, 183, 0.16), transparent 24%),
        linear-gradient(160deg, #060816 0%, #0b1020 42%, #111827 100%);
      font-family: "Georgia", "Songti SC", "Microsoft YaHei", serif;
    }
    .shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.3fr 0.9fr;
      gap: 20px;
      align-items: stretch;
      margin-bottom: 20px;
    }
    .hero-card, .panel {
      background: var(--panel);
      backdrop-filter: blur(16px);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }
    .hero-card {
      padding: 28px;
      overflow: hidden;
      position: relative;
    }
    .hero-card::after {
      content: "";
      position: absolute;
      inset: auto -60px -60px auto;
      width: 180px;
      height: 180px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(245, 185, 66, 0.25), transparent 68%);
      pointer-events: none;
    }
    .eyebrow {
      color: var(--accent-2);
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-size: 12px;
      margin-bottom: 12px;
    }
    h1, h2, h3 { margin: 0; font-family: "Palatino Linotype", "Songti SC", serif; }
    h1 { font-size: clamp(34px, 5vw, 60px); line-height: 1.04; margin-bottom: 14px; }
    .lead { color: var(--muted); line-height: 1.7; max-width: 56rem; }
    .chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
    .chip {
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.12);
      border: 1px solid rgba(148, 163, 184, 0.2);
      color: var(--text);
      font-size: 13px;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
    }
    .panel {
      padding: 22px;
      background: var(--panel-strong);
    }
    .panel h2 { font-size: 24px; margin-bottom: 10px; }
    .label { display: block; margin: 12px 0 8px; color: var(--muted); font-size: 14px; }
    textarea, input[type="text"] {
      width: 100%;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.26);
      background: rgba(2, 6, 23, 0.5);
      color: var(--text);
      padding: 14px 15px;
      outline: none;
      font-size: 15px;
      font-family: inherit;
    }
    textarea { min-height: 160px; resize: vertical; }
    textarea:focus, input:focus { border-color: rgba(245, 185, 66, 0.8); box-shadow: 0 0 0 3px rgba(245, 185, 66, 0.12); }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
    button {
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      font-size: 14px;
      cursor: pointer;
      color: #0b1020;
      background: linear-gradient(135deg, var(--accent), #ffd98b);
      font-weight: 700;
    }
    button.secondary { background: rgba(148, 163, 184, 0.14); color: var(--text); border: 1px solid rgba(148, 163, 184, 0.22); }
    button.danger { background: linear-gradient(135deg, var(--danger), #fda4af); }
    .meta { color: var(--muted); font-size: 13px; margin-top: 10px; }
    .result {
      white-space: pre-wrap;
      line-height: 1.8;
      padding: 18px;
      border-radius: 18px;
      border: 1px solid rgba(148, 163, 184, 0.22);
      background: rgba(2, 6, 23, 0.46);
      overflow: auto;
      font-size: 14px;
    }
    .result.plan {
      min-height: 220px;
      font-family: inherit;
    }
    .result.message {
      min-height: 140px;
      font-family: inherit;
    }
    .split { display: grid; gap: 14px; grid-template-columns: 1fr 1fr; }
    .card {
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 16px;
      background: rgba(148, 163, 184, 0.06);
      padding: 14px;
    }
    .section-title {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
      font-size: 15px;
    }
    .section-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      border-radius: 999px;
      background: rgba(245, 185, 66, 0.18);
      color: #ffd98b;
      font-weight: 700;
      flex: 0 0 auto;
    }
    .section-copy {
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
      line-height: 1.6;
    }
    .tag {
      display: inline-block;
      margin: 4px 8px 0 0;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(110, 231, 183, 0.14);
      border: 1px solid rgba(110, 231, 183, 0.18);
      font-size: 12px;
    }
    .status { color: var(--accent-2); font-weight: 700; }
    @media (max-width: 980px) {
      .hero, .grid, .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">FastAPI Frontend</div>
        <h1>本地生活规划与确认面板</h1>
        <p class="lead">输入你的出行需求，系统会生成最终规划结果；如果需要补充信息，会继续追问；当方案生成后，你可以在页面里直接确认执行。</p>
        <div class="chips">
          <span class="chip">用户输入接口</span>
          <span class="chip">最终规划结果展示接口</span>
          <span class="chip">确认执行接口</span>
        </div>
      </div>
      <div class="hero-card">
        <div class="eyebrow">How it works</div>
        <h2>三步闭环</h2>
        <p class="lead">先提交需求，再查看规划详情，最后确认是否执行。页面支持追问回复，适配当前工作流的澄清场景。</p>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>输入需求</h2>
        <label class="label" for="userInput">规划需求</label>
        <textarea id="userInput" placeholder="例如：今天下午有空，想和老婆孩子出去玩几个小时，不要离家太远。"></textarea>
        <div class="actions">
          <button id="startBtn">生成方案</button>
          <button id="resetBtn" class="secondary">清空会话</button>
        </div>
        <div class="meta">当前会话：<span id="sessionId">未开始</span></div>

        <div id="clarifyBox" style="display:none; margin-top:18px;">
          <h3 style="margin-bottom:8px;">系统追问</h3>
          <div class="card" id="clarifyText" style="margin-bottom:12px;"></div>
          <label class="label" for="clarifyReply">补充回答</label>
          <input id="clarifyReply" type="text" placeholder="请输入你的补充信息" />
          <div class="actions">
            <button id="clarifyBtn" class="secondary">提交回复</button>
          </div>
        </div>

        <div id="confirmBox" style="display:none; margin-top:18px;">
          <h3 style="margin-bottom:8px;">执行确认</h3>
          <div class="card" id="confirmText" style="margin-bottom:12px;"></div>
          <div class="actions">
            <button id="confirmYesBtn">确认执行</button>
            <button id="confirmNoBtn" class="danger">暂不执行</button>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>规划结果</h2>
        <div class="meta">状态：<span id="statusText" class="status">等待输入</span></div>
        <div class="split" style="margin:14px 0;">
          <div class="card">
            <div class="section-title"><span class="section-badge">1</span><strong>行程安排</strong></div>
            <div id="planSectionOne" class="result plan">生成结果后会显示在这里。</div>
            <div class="section-copy">展示当前规划中的时间安排、活动与餐厅，不展示结构化原始结果。</div>
          </div>
          <div class="card">
            <div class="section-title"><span class="section-badge">2</span><strong>方案说明</strong></div>
            <div id="planSectionTwo" class="result plan">生成结果后会显示在这里。</div>
            <div class="section-copy">展示方案为什么这样安排，以及当前方案的核心理由。</div>
          </div>
        </div>
        <div class="card">
          <strong>执行反馈</strong>
          <div id="finalMessage" class="result message" style="margin-top:10px;">尚未执行。</div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const sessionIdEl = document.getElementById('sessionId');
    const statusTextEl = document.getElementById('statusText');
    const planSectionOneEl = document.getElementById('planSectionOne');
    const planSectionTwoEl = document.getElementById('planSectionTwo');
    const finalMessageEl = document.getElementById('finalMessage');
    const clarifyBoxEl = document.getElementById('clarifyBox');
    const clarifyTextEl = document.getElementById('clarifyText');
    const clarifyReplyEl = document.getElementById('clarifyReply');
    const confirmBoxEl = document.getElementById('confirmBox');
    const confirmTextEl = document.getElementById('confirmText');
    const userInputEl = document.getElementById('userInput');

    let currentSessionId = localStorage.getItem('planner_session_id') || '';

    function escapeHtml(value) {
      return String(value || '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }

    function extractSection(text, sectionNumber) {
      const content = String(text || '').trim();
      if (!content) return '';
      const regex = sectionNumber === 1
        ? /##\\s*第一部分[：:]\\s*行程安排\\s*([\\s\\S]*?)(?=\\n##\\s*第二部分[：:]\\s*方案说明|\\n##\\s*第三部分[：:]\\s*确认提示|$)/
        : /##\\s*第二部分[：:]\\s*方案说明\\s*([\\s\\S]*?)(?=\\n##\\s*第三部分[：:]\\s*确认提示|$)/;
      const match = content.match(regex);
      const body = (match && match[1] ? match[1] : '').trim();
      return body || (sectionNumber === 1 ? '暂无行程安排。' : '暂无方案说明。');
    }

    function renderSession(session) {
      if (!session) return;
      sessionIdEl.textContent = session.session_id || '未开始';
      statusTextEl.textContent = session.status || 'unknown';
      planSectionOneEl.innerHTML = escapeHtml(extractSection(session.display_text, 1));
      planSectionTwoEl.innerHTML = escapeHtml(extractSection(session.display_text, 2));
      finalMessageEl.textContent = session.final_message || '尚未执行。';

      const pendingClarification = session.pending_clarification || '';
      clarifyBoxEl.style.display = pendingClarification ? 'block' : 'none';
      if (pendingClarification) {
        clarifyTextEl.textContent = pendingClarification;
      }

      const pendingConfirmation = session.pending_confirmation || {};
      confirmBoxEl.style.display = pendingConfirmation.message ? 'block' : 'none';
      if (pendingConfirmation.message) {
        confirmTextEl.textContent = pendingConfirmation.message;
      }
    }

    async function requestJson(url, body) {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || response.statusText);
      }
      return response.json();
    }

    async function startPlanning() {
      const userInput = userInputEl.value.trim();
      if (!userInput) {
        alert('请先输入规划需求。');
        return;
      }
      const data = await requestJson('/api/sessions', { user_input: userInput });
      currentSessionId = data.session_id;
      localStorage.setItem('planner_session_id', currentSessionId);
      renderSession(data);
    }

    async function submitClarification() {
      if (!currentSessionId) return;
      const reply = clarifyReplyEl.value.trim();
      if (!reply) {
        alert('请先输入补充回答。');
        return;
      }
      const data = await requestJson(`/api/sessions/${currentSessionId}/clarify`, { user_reply: reply });
      clarifyReplyEl.value = '';
      renderSession(data);
    }

    async function confirmExecution(confirmed) {
      if (!currentSessionId) return;
      const data = await requestJson(`/api/sessions/${currentSessionId}/confirm`, { confirmed });
      renderSession(data);
    }

    async function loadCurrentSession() {
      if (!currentSessionId) return;
      const response = await fetch(`/api/sessions/${currentSessionId}`);
      if (!response.ok) return;
      renderSession(await response.json());
    }

    document.getElementById('startBtn').addEventListener('click', () => startPlanning().catch(err => alert(err.message)));
    document.getElementById('clarifyBtn').addEventListener('click', () => submitClarification().catch(err => alert(err.message)));
    document.getElementById('confirmYesBtn').addEventListener('click', () => confirmExecution(true).catch(err => alert(err.message)));
    document.getElementById('confirmNoBtn').addEventListener('click', () => confirmExecution(false).catch(err => alert(err.message)));
    document.getElementById('resetBtn').addEventListener('click', () => {
      currentSessionId = '';
      localStorage.removeItem('planner_session_id');
      sessionIdEl.textContent = '未开始';
      statusTextEl.textContent = '等待输入';
      planSectionOneEl.textContent = '生成结果后会显示在这里。';
      planSectionTwoEl.textContent = '生成结果后会显示在这里。';
      finalMessageEl.textContent = '尚未执行。';
      clarifyBoxEl.style.display = 'none';
      confirmBoxEl.style.display = 'none';
      clarifyReplyEl.value = '';
    });

    loadCurrentSession().catch(() => {});
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_render_html())


@app.post("/api/sessions")
def create_session(payload: CreateSessionRequest) -> JSONResponse:
    session_id = uuid.uuid4().hex
    session = _run_planning(session_id, payload.user_input.strip())
    return JSONResponse(session)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> JSONResponse:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(session)


@app.get("/api/sessions/{session_id}/plan")
def get_session_plan(session_id: str) -> JSONResponse:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(
        {
            "session_id": session_id,
            "status": session.get("status"),
            "display_text": session.get("display_text") or "",
            "plan": session.get("plan") or {},
            "pending_confirmation": session.get("pending_confirmation") or {},
        }
    )


@app.post("/api/sessions/{session_id}/clarify")
def clarify_session(session_id: str, payload: ClarifySessionRequest) -> JSONResponse:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _resume_with_reply(session_id, payload.user_reply.strip())
    return JSONResponse(session)


@app.post("/api/sessions/{session_id}/confirm")
def confirm_session(session_id: str, payload: ConfirmSessionRequest) -> JSONResponse:
    session = _run_confirmation(session_id, payload.confirmed)
    return JSONResponse(session)