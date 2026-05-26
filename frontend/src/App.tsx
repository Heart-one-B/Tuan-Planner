import { FormEvent, useEffect, useMemo, useState } from 'react';

type UiState = {
  step: string;
  prompt: string;
  kind: string;
  options: Array<{ label: string; value: boolean | string }>;
};

type ChatResponse = {
  session_id: string;
  status: string;
  ui: UiState;
  state: Record<string, any>;
  display_text: string;
  final_message: string;
  llm_answer: string;
  execution_result: Record<string, any>;
  errors: string[];
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
const DEFAULT_REQUEST = '今天下午是空的，想和老婆孩子出去玩几个小时。老婆最近在减肥，孩子5岁。';

function prettify(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  return JSON.stringify(value, null, 2);
}

export default function App() {
  const [sessionId, setSessionId] = useState('');
  const [userInput, setUserInput] = useState(DEFAULT_REQUEST);
  const [originArea, setOriginArea] = useState('');
  const [reply, setReply] = useState('');
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const cached = window.localStorage.getItem('local-life-agent-session-id');
    if (cached) {
      setSessionId(cached);
    }
  }, []);

  useEffect(() => {
    if (sessionId) {
      window.localStorage.setItem('local-life-agent-session-id', sessionId);
    }
  }, [sessionId]);

  const pendingStep = response?.ui?.step || response?.state?.pending_action || '';
  const pendingPrompt = response?.ui?.prompt || '';
  const pendingKind = response?.ui?.kind || '';
  const pendingOptions = response?.ui?.options || [];

  const summaryText = useMemo(() => {
    if (!response) {
      return '等待开始';
    }
    if (response.display_text) {
      return response.display_text;
    }
    if (response.final_message) {
      return response.final_message;
    }
    if (response.llm_answer) {
      return response.llm_answer;
    }
    return '当前没有可展示的最终文本。';
  }, [response]);

  async function sendPayload(payload: Record<string, unknown>) {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId || undefined,
          ...payload,
        }),
      });

      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `请求失败: ${res.status}`);
      }

      const data = (await res.json()) as ChatResponse;
      setSessionId(data.session_id);
      setResponse(data);
      setReply('');
    } catch (err) {
      setError(err instanceof Error ? err.message : '请求失败');
    } finally {
      setLoading(false);
    }
  }

  function handleSubmitNewRequest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    sendPayload({
      user_input: userInput,
      runtime_origin_area: originArea || undefined,
    });
  }

  function handleClarifySubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!pendingStep) {
      return;
    }
    if (pendingStep === 'location_fallback') {
      sendPayload({ runtime_origin_area: reply });
      return;
    }
    if (pendingStep === 'clarification') {
      sendPayload({ clarification_response: reply });
    }
  }

  return (
    <div className="shell">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <header className="hero">
        <div>
          <div className="eyebrow">Local Life Agent Console</div>
          <h1>本地生活 Agent 标准版前端</h1>
          <p>
            这个页面把输入、澄清、确认和结果展示统一到一个界面，直接对接后端 workflow。
          </p>
        </div>
        <div className="status-card">
          <span className={`status-pill ${response?.status || 'idle'}`}>{response?.status || 'idle'}</span>
          <div className="session-line">Session: {sessionId || '未开始'}</div>
          <div className="session-line">Pending: {pendingStep || '无'}</div>
        </div>
      </header>

      <main className="grid">
        <section className="panel panel-primary">
          <h2>1. 发起需求</h2>
          <form className="stack" onSubmit={handleSubmitNewRequest}>
            <label className="field">
              <span>需求输入</span>
              <textarea
                value={userInput}
                onChange={(event) => setUserInput(event.target.value)}
                rows={6}
                placeholder="输入你想安排的本地生活需求"
              />
            </label>
            <div className="two-col">
              <label className="field">
                <span>起点区域（可选）</span>
                <input
                  value={originArea}
                  onChange={(event) => setOriginArea(event.target.value)}
                  placeholder="例如：望京、朝阳、国贸"
                />
              </label>
              <label className="field">
                <span>会话 ID</span>
                <input value={sessionId} readOnly placeholder="提交后自动生成" />
              </label>
            </div>
            <div className="action-row">
              <button className="primary-btn" type="submit" disabled={loading}>
                {loading ? '处理中...' : '开始规划'}
              </button>
              <button
                className="ghost-btn"
                type="button"
                onClick={() => {
                  setSessionId('');
                  setResponse(null);
                  setReply('');
                  setError('');
                  window.localStorage.removeItem('local-life-agent-session-id');
                }}
              >
                新会话
              </button>
            </div>
          </form>

          <div className="mini-grid">
            <div className="mini-card">
              <span>提示状态</span>
              <strong>{pendingStep || '无'}</strong>
            </div>
            <div className="mini-card">
              <span>提示类型</span>
              <strong>{pendingKind || '无'}</strong>
            </div>
          </div>
        </section>

        <section className="panel">
          <h2>2. 当前交互</h2>
          {pendingStep ? (
            <div className="stack">
              <div className="prompt-box">
                <div className="prompt-label">系统提示</div>
                <p>{pendingPrompt}</p>
                {pendingOptions.length > 0 ? (
                  <div className="option-row">
                    {pendingOptions.map((option) => (
                      <button
                        key={`${option.label}-${String(option.value)}`}
                        className="ghost-btn"
                        type="button"
                        onClick={() => {
                          if (pendingStep === 'location_permission' || pendingStep === 'confirmation') {
                            sendPayload(
                              pendingStep === 'location_permission'
                                ? { location_permission_granted: Boolean(option.value) }
                                : { user_confirmed: Boolean(option.value) },
                            );
                          }
                        }}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
              {pendingKind !== 'confirm' ? (
                <form className="stack" onSubmit={handleClarifySubmit}>
                  <label className="field">
                    <span>
                      {pendingStep === 'location_fallback' ? '补充城市 / 区域' : '补充说明'}
                    </span>
                    <textarea
                      value={reply}
                      onChange={(event) => setReply(event.target.value)}
                      rows={4}
                      placeholder={pendingStep === 'location_fallback' ? '例如：北京朝阳' : '请输入补充内容'}
                    />
                  </label>
                  <button className="primary-btn" type="submit" disabled={loading}>
                    提交补充信息
                  </button>
                </form>
              ) : null}
            </div>
          ) : (
            <div className="empty-state">
              在这里会显示定位授权、澄清问题或确认执行按钮。
            </div>
          )}
        </section>

        <section className="panel panel-output">
          <h2>3. 输出面板</h2>
          <div className="output-stack">
            <article className="output-card">
              <div className="output-title">最终展示文本</div>
              <pre>{summaryText}</pre>
            </article>
            <article className="output-card">
              <div className="output-title">执行结果 / 状态</div>
              <pre>{prettify(response?.execution_result || {})}</pre>
            </article>
            <article className="output-card">
              <div className="output-title">原始状态</div>
              <pre>{prettify(response?.state || {})}</pre>
            </article>
          </div>
          {response?.errors?.length ? (
            <div className="error-box">
              <div className="output-title">错误信息</div>
              <pre>{response.errors.join('\n')}</pre>
            </div>
          ) : null}
          {error ? <div className="error-banner">{error}</div> : null}
        </section>
      </main>
    </div>
  );
}
