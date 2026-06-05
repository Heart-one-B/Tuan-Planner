from __future__ import annotations

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


# ---------------------------------------------------------------------------
# 后端辅助函数
# ---------------------------------------------------------------------------

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
        "llm_answer": state.get("llm_answer") or "",
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


# ---------------------------------------------------------------------------
# 前端 HTML
# ---------------------------------------------------------------------------

def _render_html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>本地生活规划助手</title>
  <style>
    :root {
      --bg-top:#f4efe3; --bg-mid:#efe6d3; --bg-btm:#e5d4b7;
      --text:#2a231a; --muted:#6d604f;
      --line:rgba(112,92,60,.20);
      --ai-bg:#fff9ed; --user-bg:#3b2e1f; --user-text:#fff8ed;
      --accent:#b9812f; --accent2:#7ea06d; --danger:#b14e3d;
      --shadow:0 20px 56px rgba(58,40,12,.18);
    }
    *,*::before,*::after{box-sizing:border-box;}
    body{
      margin:0; min-height:100vh;
      font-family:"Avenir Next","PingFang SC","Microsoft YaHei",sans-serif;
      color:var(--text);
      background:
        radial-gradient(circle at 10% 0%,rgba(177,78,61,.12),transparent 26%),
        radial-gradient(circle at 90% 0%,rgba(126,160,109,.17),transparent 28%),
        linear-gradient(180deg,var(--bg-top) 0%,var(--bg-mid) 50%,var(--bg-btm) 100%);
      display:flex; justify-content:center; padding:20px 14px;
    }
    .shell{
      width:min(1020px,100%); min-height:calc(100vh - 40px);
      border:1px solid var(--line); border-radius:28px;
      background:linear-gradient(160deg,rgba(255,250,240,.93),rgba(247,238,220,.89));
      box-shadow:var(--shadow);
      display:grid; grid-template-rows:auto 1fr auto; overflow:hidden;
    }

    /* ─── 顶栏 ─── */
    .topbar{
      display:flex; justify-content:space-between; align-items:center; gap:12px;
      border-bottom:1px solid var(--line); padding:14px 20px;
      background:rgba(255,249,236,.92); backdrop-filter:blur(8px);
    }
    .title-group h1{margin:0;font-size:clamp(17px,2.6vw,24px);letter-spacing:.02em;}
    .title-group p{margin:4px 0 0;color:var(--muted);font-size:12px;}
    .topbar-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
    .meta-pill{
      border:1px solid var(--line); border-radius:999px; padding:4px 10px;
      background:rgba(255,253,248,.9); color:#5f4f3a; font-size:12px;
      max-width:180px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    }
    .sdot{
      display:inline-block; width:7px; height:7px; border-radius:50%;
      background:#c0b49a; margin-right:4px; vertical-align:middle; transition:background .3s;
    }
    .sdot.active{background:var(--accent2);animation:pulse 1.6s ease infinite;}
    .sdot.done{background:var(--accent2);}
    .sdot.err{background:var(--danger);}

    /* ─── 对话区 ─── */
    .chat-area{overflow-y:auto;padding:20px 18px;scroll-behavior:smooth;}
    .welcome{
      border:1px dashed rgba(112,92,60,.30); border-radius:20px;
      padding:20px 22px; background:rgba(255,252,244,.72);
      color:#6f5f48; line-height:1.75; animation:fadeUp 500ms ease;
    }
    .welcome strong{color:var(--accent);}
    .welcome ul{margin:10px 0 0;padding-left:20px;}
    .welcome li{margin:5px 0;}
    .msg-row{display:flex;gap:10px;margin:12px 0;animation:fadeUp 300ms ease;}
    .msg-row.user{justify-content:flex-end;}
    .msg-row.ai{justify-content:flex-start;}
    .msg-row.sys{justify-content:center;}
    .avatar{
      flex:0 0 auto; width:32px; height:32px; border-radius:10px;
      background:rgba(255,251,242,.96); border:1px solid var(--line);
      display:flex; align-items:center; justify-content:center;
      font-size:12px; font-weight:700; color:#5d4d36;
    }
    .bubble{
      max-width:min(780px,92vw); padding:12px 15px; border-radius:18px;
      border:1px solid var(--line); line-height:1.75; font-size:14px;
      box-shadow:0 6px 18px rgba(82,64,33,.07);
    }
    .msg-row.ai .bubble{background:var(--ai-bg);color:#3e3120;border-top-left-radius:5px;}
    .msg-row.user .bubble{
      background:var(--user-bg);color:var(--user-text);
      border-color:rgba(255,243,222,.20);border-top-right-radius:5px;
    }
    .msg-row.sys .bubble{
      background:rgba(200,190,175,.22);color:var(--muted);
      font-size:12px;border-radius:999px;padding:4px 14px;border-color:rgba(112,92,60,.12);
    }

    /* ─── 加载动效 ─── */
    .loading-wrap{display:flex;flex-direction:column;gap:9px;}
    .ld-row{display:flex;align-items:center;gap:9px;font-size:13px;color:#7a6a52;}
    .spinner{
      flex:0 0 auto;width:15px;height:15px;border-radius:50%;
      border:2px solid rgba(185,129,47,.22);border-top-color:var(--accent);
      animation:spin 700ms linear infinite;
    }
    .ld-row.done{color:var(--accent2);}
    .ld-row.done::before{content:"✓ ";font-weight:700;}
    .ld-row.pend{opacity:.4;color:#c0a87a;}
    .prog-track{height:3px;background:rgba(185,129,47,.14);border-radius:999px;overflow:hidden;margin-top:3px;}
    .prog-fill{height:100%;background:linear-gradient(90deg,var(--accent),#d5a95f);border-radius:999px;transition:width .8s ease;}

    /* ─── 追问 ─── */
    .clar-wrap{display:flex;flex-direction:column;gap:6px;}
    .clar-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;}
    .clar-q{font-size:14px;color:#3e3120;line-height:1.75;white-space:pre-wrap;}

    /* ─── 方案整体容器 ─── */
    .plan-wrap{display:flex;flex-direction:column;gap:14px;}
    .plan-hd{
      display:flex;align-items:flex-start;gap:10px;
      padding-bottom:10px;border-bottom:1px solid var(--line);
    }
    .plan-hd-icon{font-size:20px;flex:0 0 auto;margin-top:2px;}
    .plan-hd-title{font-size:15px;font-weight:700;color:#2e2416;}
    .plan-hd-sub{font-size:12px;color:var(--muted);margin-top:3px;}

    /* 分区 */
    .plan-sec{display:flex;flex-direction:column;gap:10px;}
    .plan-sec-hd{display:flex;align-items:center;gap:8px;}
    .plan-sec-badge{
      width:22px;height:22px;border-radius:8px;
      background:rgba(185,129,47,.15);color:#a0711f;
      font-size:11px;font-weight:700;
      display:flex;align-items:center;justify-content:center;flex:0 0 auto;
    }
    .plan-sec-title{font-size:13.5px;font-weight:700;color:#5b482f;}

    /* ─── 时间线 ─── */
    .tl{display:flex;flex-direction:column;}
    .tl-row{
      display:grid;
      grid-template-columns:82px 36px 1fr;
      gap:0 10px;
    }
    .tl-time{
      text-align:right;padding-top:7px;
      font-size:11px;font-weight:700;color:var(--accent);
      line-height:1.3;word-break:break-all;
    }
    .tl-spine{display:flex;flex-direction:column;align-items:center;}
    .tl-node{
      width:32px;height:32px;border-radius:50%;
      background:rgba(185,129,47,.10);border:2px solid rgba(185,129,47,.50);
      display:flex;align-items:center;justify-content:center;
      font-size:15px;flex:0 0 auto;z-index:1;
    }
    .tl-vbar{
      flex:1;width:2px;min-height:16px;
      background:linear-gradient(to bottom,rgba(185,129,47,.35),rgba(185,129,47,.06));
    }
    .tl-body{padding:5px 0 14px;}
    .tl-name{font-size:14px;font-weight:700;color:#2e2416;line-height:1.4;}
    .tl-loc{font-size:12px;color:var(--muted);margin-top:3px;}
    .tl-empty{color:var(--muted);font-size:13px;padding:6px 0;}

    /* ─── 地点详情卡片 ─── */
    .venue-list{display:flex;flex-direction:column;gap:10px;}
    .venue-card{
      border:1px solid rgba(112,92,60,.16);border-radius:14px;
      background:#fffcf5;padding:13px 14px;
      transition:box-shadow .2s;
    }
    .venue-card:hover{box-shadow:0 4px 16px rgba(58,40,12,.10);}
    .venue-hd{display:flex;gap:11px;align-items:flex-start;margin-bottom:9px;}
    .venue-icon{
      width:40px;height:40px;border-radius:11px;
      background:rgba(185,129,47,.10);
      display:flex;align-items:center;justify-content:center;
      font-size:20px;flex:0 0 auto;
    }
    .venue-info{flex:1;}
    .venue-name{font-size:14.5px;font-weight:700;color:#2e2416;line-height:1.35;}
    .venue-tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:5px;}
    .vtag{
      font-size:11.5px;padding:2px 8px;border-radius:999px;
      display:inline-flex;align-items:center;gap:2px;
    }
    .vtag-star{background:rgba(255,196,0,.15);color:#7a5e00;}
    .vtag-dist{background:rgba(126,160,109,.15);color:#3a5e32;}
    .vtag-cat {background:rgba(148,163,184,.15);color:#415060;}
    .vtag-price{background:rgba(185,129,47,.14);color:#7a5218;}
    .venue-addr{font-size:12px;color:var(--muted);margin-top:5px;}
    .venue-details{display:flex;flex-direction:column;gap:4px;margin-top:2px;}
    .venue-li{
      font-size:13px;color:#4a3d2e;padding-left:14px;
      position:relative;line-height:1.6;
    }
    .venue-li::before{content:"•";position:absolute;left:2px;color:var(--accent);font-weight:700;}

    /* ─── 方案说明（纯文本区） ─── */
    .plan-prose{
      font-size:13.5px;color:#4a3d2e;line-height:1.8;
    }
    .plan-prose .md-h3{font-weight:700;color:#3e3020;margin:8px 0 3px;display:block;}
    .plan-prose .md-li{padding-left:15px;display:block;position:relative;}
    .plan-prose .md-li::before{content:"·";position:absolute;left:2px;color:var(--accent);}
    .plan-prose .md-bold{font-weight:700;color:#3e3020;}
    .plan-prose .md-hr{border:none;border-top:1px solid var(--line);margin:8px 0;display:block;}

    /* ─── 确认按钮栏 ─── */
    .confirm-wrap{display:flex;flex-direction:column;gap:10px;}
    .confirm-msg{font-size:14px;color:#3e3120;line-height:1.75;white-space:pre-wrap;}
    .confirm-actions{display:flex;gap:8px;flex-wrap:wrap;}

    /* ─── 执行反馈卡 ─── */
    .fb-card{
      display:flex;align-items:flex-start;gap:12px;
      background:rgba(242,252,235,.75);border:1px solid rgba(126,160,109,.28);
      border-radius:14px;padding:13px 14px;
    }
    .fb-card.cancel{background:rgba(255,245,242,.75);border-color:rgba(177,78,61,.22);}
    .fb-icon{font-size:22px;flex:0 0 auto;}
    .fb-title{font-size:14px;font-weight:700;color:#3a2e1e;margin-bottom:5px;}
    .fb-msg{font-size:13.5px;color:#4a3d2e;line-height:1.72;white-space:pre-wrap;}

    /* ─── 按钮 ─── */
    button{
      border:none;border-radius:12px;padding:9px 15px;
      font-size:13px;font-weight:700;cursor:pointer;font-family:inherit;
      transition:transform 120ms ease,box-shadow 120ms ease,opacity 120ms ease;
    }
    button:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.12);}
    button:disabled{opacity:.5;cursor:not-allowed;transform:none!important;}
    .btn-primary{background:linear-gradient(135deg,#bb8534,#d5a95f);color:#fff9ef;}
    .btn-sec{background:rgba(98,78,48,.10);color:#4e3f2d;border:1px solid rgba(112,92,60,.22);}
    .btn-green{background:linear-gradient(135deg,#5a9e6b,#7ec487);color:#f0fff4;}
    .btn-danger{background:linear-gradient(135deg,#ba5e4f,#d67f6f);color:#fff4ef;}

    /* ─── 输入区 ─── */
    .composer{
      border-top:1px solid var(--line);
      background:rgba(255,250,238,.96);padding:12px 14px;
    }
    .composer-inner{
      display:grid;grid-template-columns:1fr auto;gap:10px;align-items:end;
      max-width:960px;margin:0 auto;
    }
    .composer textarea{
      width:100%;resize:none;min-height:50px;max-height:160px;
      border-radius:14px;border:1px solid rgba(112,92,60,.25);
      background:#fffdf8;padding:11px 13px;font-size:14px;
      outline:none;color:#362a1c;font-family:inherit;line-height:1.6;
    }
    .composer textarea:focus{
      border-color:rgba(185,129,47,.75);
      box-shadow:0 0 0 3px rgba(185,129,47,.12);
    }
    .composer-footer{
      display:flex;align-items:center;justify-content:space-between;gap:8px;
      margin-top:7px;max-width:960px;margin-left:auto;margin-right:auto;
    }
    .mode-badge{font-size:12px;padding:3px 10px;border-radius:999px;font-weight:600;}
    .mode-new{background:rgba(185,129,47,.13);color:#8a5e1a;}
    .mode-clar{background:rgba(126,160,109,.18);color:#3e6636;}
    .mode-wait{background:rgba(177,78,61,.12);color:#8a2e21;}
    .hint-text{font-size:12px;color:var(--muted);}

    @media(max-width:820px){
      .shell{border-radius:18px;min-height:calc(100vh - 18px);}
      .topbar{flex-direction:column;align-items:flex-start;}
      .tl-row{grid-template-columns:68px 32px 1fr;gap:0 8px;}
      .composer-inner{grid-template-columns:1fr;}
      .composer button{width:100%;}
    }
    @keyframes spin{to{transform:rotate(360deg);}}
    @keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}
    @keyframes fadeUp{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
  </style>
</head>
<body>
<div class="shell">

  <header class="topbar">
    <div class="title-group">
      <h1>本地生活规划助手</h1>
      <p>输入出行需求 · 智能追问补全 · 生成规划方案 · 一键确认执行</p>
    </div>
    <div class="topbar-right">
      <span class="meta-pill" id="sessionPill">未开始</span>
      <span class="meta-pill">
        <span class="sdot" id="sdot"></span>
        <span id="statusLabel">等待输入</span>
      </span>
      <button id="resetBtn" class="btn-sec" type="button">新建对话</button>
    </div>
  </header>

  <main class="chat-area" id="chatArea">
    <div class="welcome" id="welcomeCard">
      <strong>👋 欢迎使用本地生活规划助手！</strong>
      <ul>
        <li>直接输入出行需求，例如：<em>今天下午 4 小时，想带孩子出去玩，不要离家太远</em></li>
        <li>系统会追问缺少的信息（出发地、时间、偏好等）</li>
        <li>收集完整后自动生成可视化行程方案与地点详情</li>
        <li>对话中直接点击「确认执行」或「暂不执行」</li>
      </ul>
    </div>
    <div id="msgList"></div>
  </main>

  <footer class="composer">
    <div class="composer-inner">
      <textarea id="inputBox" placeholder="输入你的规划需求，按 Enter 发送，Shift+Enter 换行"></textarea>
      <button id="sendBtn" class="btn-primary" type="button">发送</button>
    </div>
    <div class="composer-footer">
      <span class="mode-badge mode-new" id="modeBadge">新需求</span>
      <span class="hint-text" id="hintText">Enter 发送 · Shift+Enter 换行</span>
    </div>
  </footer>

</div>
<script>
(function(){
  /* ── DOM ── */
  const chatArea    = document.getElementById('chatArea');
  const msgList     = document.getElementById('msgList');
  const welcomeCard = document.getElementById('welcomeCard');
  const inputBox    = document.getElementById('inputBox');
  const sendBtn     = document.getElementById('sendBtn');
  const resetBtn    = document.getElementById('resetBtn');
  const sessionPill = document.getElementById('sessionPill');
  const sdot        = document.getElementById('sdot');
  const statusLabel = document.getElementById('statusLabel');
  const modeBadge   = document.getElementById('modeBadge');
  const hintText    = document.getElementById('hintText');

  /* ── State ── */
  let sessionId  = localStorage.getItem('lp_sid') || '';
  let mode       = 'new';
  let busy       = false;
  let loadingRow = null;
  let stageTimer = null;

  /* ── Stage configs ── */
  const STAGES_PLAN = [
    {icon:'🧭', text:'解析规划意图…'},
    {icon:'📍', text:'获取地理位置…'},
    {icon:'🌤️', text:'收集天气实况…'},
    {icon:'🔍', text:'搜索候选活动与餐厅…'},
    {icon:'📋', text:'制定行程方案…'},
    {icon:'✅', text:'规则校验与可行性评估…'},
    {icon:'⭐', text:'方案评分排序…'},
    {icon:'✨', text:'生成最终展示内容…'},
  ];
  const STAGES_CLAR = [
    {icon:'💬', text:'理解补充信息…'},
    {icon:'🔄', text:'更新规划约束…'},
    {icon:'📋', text:'重新规划方案…'},
  ];
  const STAGES_CONF = [
    {icon:'📝', text:'更新用户偏好…'},
    {icon:'🚀', text:'执行方案…'},
  ];

  /* ════════════ 工具函数 ════════════ */
  function esc(s){
    return String(s||'')
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
  function nl2br(s){ return esc(s).replace(/\n/g,'<br>'); }

  /* ─ Markdown 轻量渲染（用于方案说明纯文本区） ─ */
  function renderMd(text){
    if(!text) return '';
    let html = '';
    for(const raw of text.split('\n')){
      const l = raw.trimEnd();
      if(/^###\s/.test(l)){
        html += `<span class="md-h3">${esc(l.replace(/^###\s+/,''))}</span>`;
      } else if(/^[-*]\s/.test(l)){
        html += `<span class="md-li">${boldEsc(l.replace(/^[-*]\s+/,''))}</span>`;
      } else if(/^\d+\.\s/.test(l)){
        html += `<span class="md-li">${boldEsc(l.replace(/^\d+\.\s+/,''))}</span>`;
      } else if(/^---+$/.test(l)){
        html += `<span class="md-hr"></span>`;
      } else {
        html += boldEsc(l) + '\n';
      }
    }
    return html;
  }
  function boldEsc(s){
    return esc(s).replace(/\*\*(.+?)\*\*/g,'<span class="md-bold">$1</span>');
  }

  /* ─ 解析 display_text 各 ## 节点 ─ */
  function parseSections(rawText){
    if(!rawText) return {};
    const result = {};
    let key = null, buf = [];
    const norm = h => h.replace(/^第[一二三四五六七八九十\d]+部分[：:]\s*/,'').trim();
    for(const line of rawText.split('\n')){
      if(line.startsWith('## ')){
        if(key !== null) result[key] = buf.join('\n').trim();
        key = norm(line.slice(3).trim());
        buf = [];
      } else if(line.startsWith('# ')){
        if(key !== null) result[key] = buf.join('\n').trim();
        key = '__title__';
        buf = [line.slice(2).trim()];
      } else if(key !== null){
        buf.push(line);
      }
    }
    if(key !== null) result[key] = buf.join('\n').trim();
    return result;
  }

  /* ─ 根据名称猜图标 ─ */
  function guessIcon(name){
    if(!name) return '📌';
    if(/乐园|游乐|儿童|亲子|公园/.test(name)) return '🎡';
    if(/餐|菜|火锅|饭|面|饺|川|粤|烤|烧|麻辣|小吃/.test(name)) return '🍽️';
    if(/KTV|唱歌|卡拉/.test(name)) return '🎤';
    if(/咖啡|茶|奶茶|甜品|蛋糕|面包|冰淇淋|冰激/.test(name)) return '☕';
    if(/电影|影院|IMAX/.test(name)) return '🎬';
    if(/商场|购物|广场|超市/.test(name)) return '🛍️';
    if(/博物馆|展览|美术/.test(name)) return '🏛️';
    if(/温泉|spa|按摩/.test(name)) return '♨️';
    if(/酒吧|清吧|夜店/.test(name)) return '🍸';
    return '📍';
  }

  /* ─ 时间线渲染 ─ */
  function renderTimeline(text){
    if(!text) return '<div class="tl-empty">暂无行程安排。</div>';
    const lines = text.split('\n').map(l=>l.trim()).filter(Boolean);
    const tableLines = lines.filter(l=>l.startsWith('|'));

    let rows = [];
    if(tableLines.length >= 3){
      // Markdown table: skip header + separator rows
      rows = tableLines.slice(2)
        .map(r => r.split('|').map(c=>c.trim()).filter(Boolean))
        .filter(r=>r.length>0);
    } else {
      // Bullet/numbered list fallback
      const listLines = lines.filter(l=>/^([-*]|\d+\.)/.test(l));
      for(const l of listLines){
        const raw = l.replace(/^([-*]|\d+\.)\s*/,'');
        const m   = raw.match(/^(\d{1,2}:\d{2}(?:[–\-]\d{1,2}:\d{2})?)\s+(.+)/);
        rows.push(m ? [m[1], m[2]] : ['', raw]);
      }
    }

    if(rows.length === 0) return `<div class="plan-prose">${renderMd(text)}</div>`;

    let html = '<div class="tl">';
    for(let i=0; i<rows.length; i++){
      const cells  = rows[i];
      const time   = cells[0] || '';
      const label  = cells[1] || cells[0] || '';
      const loc    = cells[2] || '';
      const isLast = i === rows.length-1;
      html += `<div class="tl-row">
        <div class="tl-time">${esc(time)}</div>
        <div class="tl-spine">
          <div class="tl-node">${guessIcon(label)}</div>
          ${!isLast ? '<div class="tl-vbar"></div>' : ''}
        </div>
        <div class="tl-body">
          <div class="tl-name">${esc(label)}</div>
          ${loc && loc!==label ? `<div class="tl-loc">📍 ${esc(loc)}</div>` : ''}
        </div>
      </div>`;
    }
    html += '</div>';
    return html;
  }

  /* ─ 地点详情卡片渲染 ─ */
  function renderVenueCards(text){
    if(!text) return '';
    const lines  = text.split('\n');
    const venues = [];
    let cur = null;
    for(const line of lines){
      const t = line.trim();
      if(/^###\s/.test(t)){
        if(cur) venues.push(cur);
        cur = {name: t.replace(/^###\s+\d+\.\s*/,'').replace(/^###\s+/,''), bullets:[]};
      } else if(cur && t.startsWith('-')){
        cur.bullets.push(t.slice(1).trim());
      }
    }
    if(cur) venues.push(cur);
    if(venues.length === 0) return `<div class="plan-prose">${renderMd(text)}</div>`;

    let html = '<div class="venue-list">';
    for(const v of venues){
      // Parse meta from first bullet: 类别：xxx｜评分：xxx｜距离：xxx｜人均：xxx
      const metaBullet = v.bullets[0] || '';
      let rating='', distance='', category='', price='', address='';
      for(const part of metaBullet.split('｜')){
        const [k,...rest] = part.split(/[：:]/);
        const val = rest.join('：').trim();
        if(k==='评分') rating=val;
        else if(k==='距离') distance=val;
        else if(k==='类别') category=val;
        else if(k==='人均') price=val;
      }
      const addrBullet = v.bullets.find(b=>b.startsWith('地址：'));
      if(addrBullet) address = addrBullet.slice(3);
      // Detail bullets: skip meta (idx 0) and address bullet
      const details = v.bullets.filter((b,i)=>
        i>0 && !b.startsWith('地址：') && !b.startsWith('类别：') && b.trim()
      );

      html += `<div class="venue-card">
        <div class="venue-hd">
          <div class="venue-icon">${guessIcon(v.name)}</div>
          <div class="venue-info">
            <div class="venue-name">${esc(v.name)}</div>
            <div class="venue-tags">
              ${rating   ? `<span class="vtag vtag-star">⭐ ${esc(rating)}</span>` : ''}
              ${distance ? `<span class="vtag vtag-dist">🚗 ${esc(distance)}</span>` : ''}
              ${category ? `<span class="vtag vtag-cat">${esc(category)}</span>` : ''}
              ${price    ? `<span class="vtag vtag-price">💰 ${esc(price)}</span>` : ''}
            </div>
            ${address ? `<div class="venue-addr">📍 ${esc(address)}</div>` : ''}
          </div>
        </div>
        ${details.length>0 ? `<div class="venue-details">${
          details.map(b=>`<div class="venue-li">${esc(b)}</div>`).join('')
        }</div>` : ''}
      </div>`;
    }
    html += '</div>';
    return html;
  }

  /* ════════════ 状态栏 ════════════ */
  function setStatus(label, type){
    statusLabel.textContent = label;
    sdot.className = 'sdot' + (type ? ' '+type : '');
  }
  function setMode(m){
    mode = m;
    if(m==='clarify'){
      modeBadge.className='mode-badge mode-clar'; modeBadge.textContent='补充信息';
      inputBox.placeholder='请输入补充信息，帮助我继续完善方案…';
      hintText.textContent='Enter 发送 · Shift+Enter 换行';
    } else if(m==='await_confirm'){
      modeBadge.className='mode-badge mode-wait'; modeBadge.textContent='等待确认';
      inputBox.placeholder='请点击上方「确认执行」或「暂不执行」按钮';
      hintText.textContent='请通过按钮操作';
    } else {
      modeBadge.className='mode-badge mode-new'; modeBadge.textContent='新需求';
      inputBox.placeholder='输入你的规划需求，按 Enter 发送，Shift+Enter 换行';
      hintText.textContent='Enter 发送 · Shift+Enter 换行';
    }
  }
  function setBusy(b){
    busy=b; sendBtn.disabled=b; inputBox.disabled=b; resetBtn.disabled=b;
  }

  /* ════════════ 消息渲染 ════════════ */
  function appendRow(role, innerHtml){
    const row = document.createElement('div');
    row.className = 'msg-row '+role;
    if(role==='ai'){
      const av=document.createElement('div'); av.className='avatar'; av.textContent='AI';
      row.appendChild(av);
    }
    const bub=document.createElement('div'); bub.className='bubble';
    bub.innerHTML=innerHtml; row.appendChild(bub);
    msgList.appendChild(row);
    welcomeCard.style.display='none';
    requestAnimationFrame(()=>{ chatArea.scrollTop=chatArea.scrollHeight; });
    return row;
  }
  function appendUser(t){ return appendRow('user', nl2br(t)); }
  function appendAI(t)  { return appendRow('ai',   nl2br(t)); }
  function appendSys(t) { return appendRow('sys',  esc(t));   }

  /* ─ 加载动效 ─ */
  function stagesHtml(stages, active){
    let h='<div class="loading-wrap">';
    for(let i=0;i<stages.length;i++){
      const s=stages[i];
      const cls = i<active ? 'ld-row done' : i===active ? 'ld-row' : 'ld-row pend';
      h += `<div class="${cls}">
        ${i===active ? '<span class="spinner"></span>' : ''}
        ${esc(s.icon)} ${esc(s.text)}
      </div>`;
    }
    const pct = Math.round((Math.min(active+1,stages.length)/stages.length)*100);
    h += `<div class="prog-track"><div class="prog-fill" style="width:${pct}%"></div></div>`;
    h += '</div>';
    return h;
  }
  function startLoading(stages){
    stopLoading();
    let idx=0;
    loadingRow = appendRow('ai', stagesHtml(stages,idx));
    stageTimer = setInterval(()=>{
      idx = Math.min(idx+1, stages.length-1);
      const bub = loadingRow && loadingRow.querySelector('.bubble');
      if(bub) bub.innerHTML = stagesHtml(stages,idx);
      chatArea.scrollTop = chatArea.scrollHeight;
    }, 1800);
  }
  function stopLoading(){
    if(stageTimer){ clearInterval(stageTimer); stageTimer=null; }
    if(loadingRow){ loadingRow.remove(); loadingRow=null; }
  }

  /* ─ 追问气泡 ─ */
  function appendClarify(q){
    return appendRow('ai', `<div class="clar-wrap">
      <div class="clar-label">💬 需要补充一些信息</div>
      <div class="clar-q">${nl2br(q)}</div>
    </div>`);
  }

  /* ─ 方案卡片（行程+详情+说明，全部纵向排列） ─ */
  function appendPlan(session){
    const raw = session.display_text || '';
    const llm = (session.llm_answer || '').trim();
    if(!raw.trim() && !llm) return;
    if(!raw.trim() && llm){ appendAI(llm); return; }

    const S = parseSections(raw);

    // 取标题
    const titleLines = (S['__title__'] || '').split('\n');
    const titleLine  = titleLines[0].trim();
    const subLine    = titleLines.slice(1).join('\n').trim();

    // 行程流程（支持两种 key）
    const flowText   = S['推荐游玩流程'] || S['行程安排'] || S['行程'] || '';
    // 地点详情
    const detailText = S['地点详情'] || '';
    // 方案说明
    const explainText= S['方案说明'] || '';

    if(!flowText && !detailText && !explainText){ appendAI(raw); return; }

    let sn = 0;
    let html = '<div class="plan-wrap">';

    if(titleLine){
      html += `<div class="plan-hd">
        <div class="plan-hd-icon">📍</div>
        <div>
          <div class="plan-hd-title">${esc(titleLine)}</div>
          ${subLine ? `<div class="plan-hd-sub">${esc(subLine)}</div>` : ''}
        </div>
      </div>`;
    }

    if(flowText){
      sn++;
      html += `<div class="plan-sec">
        <div class="plan-sec-hd">
          <span class="plan-sec-badge">${sn}</span>
          <span class="plan-sec-title">推荐游玩流程</span>
        </div>
        ${renderTimeline(flowText)}
      </div>`;
    }

    if(detailText){
      sn++;
      html += `<div class="plan-sec">
        <div class="plan-sec-hd">
          <span class="plan-sec-badge">${sn}</span>
          <span class="plan-sec-title">地点详情</span>
        </div>
        ${renderVenueCards(detailText)}
      </div>`;
    }

    if(explainText){
      sn++;
      html += `<div class="plan-sec">
        <div class="plan-sec-hd">
          <span class="plan-sec-badge">${sn}</span>
          <span class="plan-sec-title">方案说明</span>
        </div>
        <div class="plan-prose">${renderMd(explainText)}</div>
      </div>`;
    }

    html += '</div>';
    appendRow('ai', html);
  }

  /* ─ 确认操作栏 ─ */
  function appendConfirm(message){
    const row = appendRow('ai', `<div class="confirm-wrap">
      <div class="confirm-msg">${nl2br(message||'是否确认执行当前方案？')}</div>
      <div class="confirm-actions">
        <button id="cYes" class="btn-green" type="button">✅ 确认执行</button>
        <button id="cNo"  class="btn-danger" type="button">✖ 暂不执行</button>
      </div>
    </div>`);
    row.querySelector('#cYes').addEventListener('click', ()=> doConfirm(true));
    row.querySelector('#cNo' ).addEventListener('click', ()=> doConfirm(false));
    return row;
  }

  /* ─ 执行反馈（确认/取消后仅显示此卡，不重复显示方案） ─ */
  function renderConfirmFeedback(session, confirmed){
    const finalMsg = String(session.final_message||'').trim();
    const isOk = confirmed;
    const defaultMsg = isOk
      ? '方案已确认，正在安排执行。如有后续进展将通知你。'
      : '已取消执行当前方案。如需重新规划，请输入新的需求。';
    const html = `<div class="fb-card${isOk?'':' cancel'}">
      <div class="fb-icon">${isOk?'✅':'✖️'}</div>
      <div>
        <div class="fb-title">${isOk?'方案已确认执行':'方案已取消'}</div>
        <div class="fb-msg">${nl2br(finalMsg||defaultMsg)}</div>
      </div>
    </div>`;
    appendRow('ai', html);
    setMode('new');
    setStatus('已完成','done');
  }

  /* ════════════ 渲染完整会话结果 ════════════ */
  function renderResult(session){
    sessionPill.textContent = (session.session_id||'').slice(0,12)||'—';

    const hasDisplay  = String(session.display_text||'').trim().length > 0;
    const hasLlm      = String(session.llm_answer  ||'').trim().length > 0;
    const clarify     = String(session.pending_clarification||'').trim();
    const confirmMsg  = String((session.pending_confirmation||{}).message||'').trim();

    if(hasDisplay || hasLlm) appendPlan(session);

    if(clarify){
      appendClarify(clarify);
      setMode('clarify');
      setStatus('等待补充','active');
    } else if(confirmMsg){
      appendConfirm(confirmMsg);
      setMode('await_confirm');
      setStatus('等待确认','active');
    } else {
      setMode('new');
      setStatus(session.status==='completed'?'已完成':'就绪',
                session.status==='completed'?'done':'');
    }
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  /* ════════════ API ════════════ */
  async function req(url, body){
    const r = await fetch(url,{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body),
    });
    if(!r.ok) throw new Error((await r.text())||r.statusText);
    return r.json();
  }

  async function doStart(text){
    startLoading(STAGES_PLAN);
    setStatus('规划中','active');
    const data = await req('/api/sessions',{user_input:text});
    sessionId = data.session_id;
    localStorage.setItem('lp_sid', sessionId);
    stopLoading();
    renderResult(data);
  }

  async function doClarify(text){
    if(!sessionId){ appendAI('当前没有活跃会话，请先输入新需求。'); setMode('new'); return; }
    startLoading(STAGES_CLAR);
    setStatus('处理补充中','active');
    const data = await req(`/api/sessions/${sessionId}/clarify`,{user_reply:text});
    stopLoading();
    renderResult(data);
  }

  async function doConfirm(confirmed){
    if(!sessionId){ appendAI('没有可确认的会话。'); setMode('new'); return; }
    // Disable both confirm buttons immediately
    document.querySelectorAll('#cYes,#cNo').forEach(b=>{ b.disabled=true; });
    appendUser(confirmed ? '确认执行当前方案。' : '暂不执行，取消方案。');
    startLoading(STAGES_CONF);
    setStatus('执行中','active');
    setBusy(true);
    try{
      const data = await req(`/api/sessions/${sessionId}/confirm`,{confirmed});
      stopLoading();
      renderConfirmFeedback(data, confirmed);   // ← 只展示反馈，不再重渲方案
    } catch(err){
      stopLoading();
      appendAI(`操作失败：${err.message}`);
      setMode('new'); setStatus('出错','err');
    } finally { setBusy(false); }
  }

  /* ─ 发送入口 ─ */
  async function handleSend(){
    if(busy) return;
    const text = inputBox.value.trim();
    if(!text) return;
    if(mode==='await_confirm'){
      appendSys('请通过上方按钮确认或拒绝方案。'); return;
    }
    appendUser(text);
    inputBox.value=''; inputBox.style.height='50px';
    setBusy(true);
    try{
      if(mode==='clarify') await doClarify(text);
      else                  await doStart(text);
    } catch(err){
      stopLoading();
      appendAI(`请求失败：${err.message||'网络错误'}`);
      setStatus('出错','err'); setMode('new');
    } finally { setBusy(false); }
  }

  /* ─ 恢复会话 ─ */
  async function loadSession(){
    if(!sessionId) return;
    try{
      const r = await fetch(`/api/sessions/${sessionId}`);
      if(!r.ok){ localStorage.removeItem('lp_sid'); sessionId=''; return; }
      appendSys('已恢复上次会话状态。');
      renderResult(await r.json());
    } catch(_){}
  }

  /* ════════════ 事件 ════════════ */
  inputBox.addEventListener('input', ()=>{
    inputBox.style.height='50px';
    inputBox.style.height = Math.min(inputBox.scrollHeight,160)+'px';
  });
  inputBox.addEventListener('keydown', e=>{
    if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); handleSend(); }
  });
  sendBtn.addEventListener('click', handleSend);
  resetBtn.addEventListener('click', ()=>{
    if(busy) return;
    sessionId=''; localStorage.removeItem('lp_sid');
    msgList.innerHTML=''; welcomeCard.style.display='';
    sessionPill.textContent='未开始';
    inputBox.value=''; inputBox.style.height='50px';
    setMode('new'); setStatus('等待输入','');
    appendAI('新对话已创建，请告诉我你的出行需求。');
  });

  /* ── 初始化 ── */
  setMode('new'); setStatus('等待输入','');
  loadSession().catch(()=>{ appendAI('欢迎使用本地生活规划助手，请输入你的出行需求。'); });
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------

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
    return JSONResponse({
        "session_id":           session_id,
        "status":               session.get("status"),
        "display_text":         session.get("display_text") or "",
        "plan":                 session.get("plan") or {},
        "pending_confirmation": session.get("pending_confirmation") or {},
    })


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