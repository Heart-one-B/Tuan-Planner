# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 项目概览

基于 LangGraph 的本地休闲出行规划多智能体系统。系统接收自然语言用户输入（例如"周末带家人去朝阳区玩半天，吃火锅"），执行意图解析、通过高德 API 进行 POI 搜索、基于约束的候选方案规划、规则校验、打分排序，最终向用户展示排名后的方案，并可选地执行预订步骤。

## 常用命令

```bash
# Run tests (run from project root; src/ must be on PYTHONPATH via conftest.py)
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_new_workflow_smoke.py -v

# Run the main entry point (interactive CLI loop)
python src/main.py
```

## 架构

### 工作流引擎（LangGraph StateGraph）

`src/graph/workflow.py` 构建状态图。`src/graph/state.py` 定义 `AgentState`（一个在各节点间累积结果的扁平字典）。`src/graph/routers.py` 包含所有条件边的路由函数。

**节点流水线（新架构）：**

```
START → intent → [clarification] → constraint_build → fact_gathering
       → candidate_planning → rule_validation → [repair_loop] → scoring
       → final_plan → presentation → [confirmation] → [execution] → final_message → END
```

- **intent**：`IntentAgent` 将用户输入解析为结构化意图（场景、时间、位置、偏好、缺失槽位）。非规划类请求路由至 `llm_answer`。
- **clarification**：多轮槽位填充循环（最多 5 轮），一次只追问一个缺失的信息。
- **constraint_build**：将结构化意图转换为声明式约束（时间窗口、预算、饮食偏好等），供下游节点使用。
- **fact_gathering**：调用 `CachedAmapClient` 进行地理编码、获取天气、搜索 POI（活动 + 餐厅）、计算 ETA。结果用于驱动候选方案生成。
- **candidate_planning**：基于收集的事实和约束，生成 3 个候选出行方案（活动 + 可选餐饮）。
- **rule_validation**：按时间段规则、天气约束、ETA 可行性、饮食偏好对方案进行校验，返回 `valid_plans` 和 `violations`。
- **scoring**：按通勤 ETA 对合法方案进行排序（主指标：首个活动的 ETA；次指标：总 ETA）。
- **final_plan**：选取排名第一的方案，格式化为最终输出结构。
- **presentation**：`PresentationAgent` 将最终方案渲染为自然语言 Markdown 展示给用户。
- **confirmation**：CLI 提示，询问用户是否继续执行预订。
- **execution**：`ExecutionAgent` 模拟预订/预约操作。
- **repair_loop**：校验失败时调整候选方案并重试（最多 3 次重规划）。

### 智能体（`src/agent/`）

每个智能体封装一次 LLM 调用，配有特定的系统提示词和输出结构（Pydantic 模型）：

- **IntentAgent**（`intent_agent.py`）：结构化意图提取，含校验失败重试逻辑。
- **PresentationAgent**（`presentation_agent.py`）：方案 → 面向用户的 Markdown。
- **ExecutionAgent**（`execution_agent.py`）：模拟预订操作。
- **RetrievalAgent**（`retrieval_agent.py`）：根据约束生成 POI 搜索关键词。

### 工具（`src/tools/`）

- **CachedAmapClient**（`cached_amap_client.py`）：三级缓存架构 — 内存/文件缓存（基于 TTL）→ 真实高德 MCP API → `poi_detail_cache.json` 离线兜底。支持地理编码、天气、POI 搜索、POI 详情、距离/ETA。由 `tools.yml` 中的 `amap_use_cache_only` 开关控制。
- **AmapMCPClient**（`amap_mcp_client.py`）：高德 API 的底层 MCP 客户端。
- **get_weather**（`get_weather.py`）：封装和风天气 API 的 LangChain 工具，用于获取实时天气。
- **mock_api.py**：用于测试/原型开发的 Mock API 响应。

### 配置（`config/`）

- `model.yml`：LLM 模型选择（提供商、模型名称、temperature 等）。
- `tools.yml`：API 密钥和功能开关（`amap_use_cache_only`、`weather_api_key`）。
- `prompts.yml`：提示词模板引用。

### 工具函数（`src/utils/`）

- `config_handler.py`：将 YAML 配置加载为模块级字典（`model_conf`、`tools_conf`、`prompts_conf`）。
- `state_utils.py`：状态操作的共享辅助函数（错误追加、候选提取、重规划原因解析）。
- `parsing_utils.py`：LLM 输出解析工具。
- `route_utils.py`：路径相关辅助函数。
- `path_tool.py`：相对项目根目录的绝对路径解析。
- `prompts_handler.py` / `logger_handler.py`：提示词加载与日志记录。

### 数据（`data/`）

- `amap_cache.json`：基于 TTL 的高德 API 响应缓存。
- `poi_detail_cache.json`：用于离线兜底的持久化 POI 详情存储。

## 关键约定

- **状态是扁平的 TypedDict**：`AgentState` 即 `dict[str, Any]` — 每个节点返回一个部分字典进行合并。通过 `.get()` 带默认值的方式访问。
- **LLM 提示词中禁止隐含默认值**：用户未提供时间/日期/人数时，LLM 不得自行编造；对应字段保持 `null`。
- **缓存优先模式**：开发时在 `tools.yml` 中设置 `amap_use_cache_only: true`，可避免消耗高德 API 配额。
- **错误累积**：节点通过 `_append_error()` 将错误追加到 `state["errors"]`，而非抛出异常。
