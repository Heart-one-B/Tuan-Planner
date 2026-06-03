# 🐿️ 美团本地生活执行 Agent (Hackathon Demo)

## 1. 项目定位

本项目是一个基于大语言模型（LLM）驱动的**本地场景短时活动规划与执行 Agent**。
区别于传统的搜索推荐系统，本项目实现了从“自然语言意图理解”到“多约束方案规划”，再到“异常自动处理（救场）”及“一键模拟预订”的**全流程闭环**。

### 核心亮点：

- **感知与规约**：自动识别天气风险及用户个性化约束（如减脂、亲子）。
- **LangGraph 编排**：通过状态图串联 Intent、Planning、Presentation、Confirmation、Execution 等节点，确保确认前不执行下单动作。
- **ReAct 思路**：通过“思考-行动-观察”循环，在发现餐厅满座或天气不佳时自动切换备选方案。
- **端到端执行**：生成的方案可直接触发 Mock API 完成下单，并生成可转发的社交话术。

------



## 2. 项目架构与规范

### 2.1 工程结构

codeText

```
AI Hackathon/
├── config/             # 配置文件目录
│   ├── model.yml       # 本地模型参数与 API Key 配置
│   └── model_example.yml # 模型配置示例
├── data/               # 数据层
│   ├── mock_db.json    # 模拟商家、活动数据库
│   └── prompts/        #  外部化的 Prompt 模板
├── docs/
│   └── requirements-checklist.md # 需求状态清单
├── src/                # 源代码根目录 (Source Root)
│   ├── agent/          # Agent 逻辑层 (Intent, Planning, Execution, etc.)
│   ├── graph/          # LangGraph 状态图编排层
│   ├── model/          # 模型工厂层 (统一管理 LLM 实例化)
│   ├── tools/          # 工具集成层 (Mock API 封装)
│   ├── utils/          # 工具类 (路径处理、配置加载)
│   └── main.py         # 项目入口
├── requirements.txt    # 依赖声明文件
└── README.md           # 项目文档
```

### 2.2 开发规范

1. **模型工厂模式**：通过 src/model/factory.py 统一管理模型实例化，目前支持 DashScope/Qwen 与 DeepSeek，可通过 config/model.yml 中的 provider 切换。
2. **LangGraph 主流程**：通过 src/graph/workflow.py 编排 Intent -> Planning -> Presentation -> Confirmation -> Execution/Reject，避免在 main.py 中手动串联复杂状态。
3. **绝对路径管理**：所有文件访问必须通过 src.utils.path_tool 转换，确保在不同操作系统和启动目录下均能准确正确定位。
4. **Prompt 分离标准**：建议将复杂的 System Prompt 存储为独立的文本文件，通过读取加载，以解耦业务逻辑与文本指令。

------



## 3. 快速启动

### 第一步：环境创建与激活

在项目根目录下打开终端，执行以下命令：

codePowershell

```
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境 (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

### 第二步：安装依赖

codePowershell

```
pip install -r requirements.txt
```

当前新增核心依赖包括：

- `langgraph`
- `langchain-openai`

### 第三步：标记源文件根目录

为了确保模块导入不报错（ModuleNotFoundError），请在 IDE 中进行设置：

- **PyCharm**: 右键点击 src 文件夹 -> **Mark Directory as** -> **Sources Root**。

### 第四步：配置 API Key

修改 config/model.yml。当前支持 DashScope/Qwen 与 DeepSeek 两类模型服务。

DeepSeek 示例：

codeYaml

```
provider: deepseek

chat_model_name: qwen3-8b
embedding_model_name: text-embedding-v4
dashscope_api_key:

deepseek_model_name: deepseek-v4-flash
deepseek_api_key: "api-key"
deepseek_base_url: https://api.deepseek.com
```

DashScope/Qwen 示例：

codeYaml

```
provider: dashscope

chat_model_name: qwen3-8b
embedding_model_name: text-embedding-v4
dashscope_api_key: "api-key"

deepseek_model_name: deepseek-v4-flash
deepseek_api_key:
deepseek_base_url: https://api.deepseek.com
```

注意：不要提交真实 API Key。可以参考 config/model_example.yml 创建本地 model.yml。

### 第五步：运行 Demo

```
python src/main.py
```

输出示例如下

```
============================================================
Meituan Local Life Execution Agent - Hackathon Demo
============================================================

请输入您的需求 (回车使用默认场景):
> 今天下午有空，想和老婆孩子出去玩几个小时，帮我安排一下，不要离家太远
[Intent Agent] 正在解析用户自然语言意图...
[OK] 解析结果: {'scenario': 'family', 'child_friendly': True, 'diet_preference': '无'}

[Planning Agent] 开始生成行程并动态调用工具...
[Tool Call] 查询天气: 35℃ 阵雨
[Fallback] 检测到高温阵雨，自动将活动约束为【室内】！
[Tool Call] 锁定活动: 奇幻森林室内亲子乐园
[Tool Call] 检查首选餐厅 (胡桃里音乐酒馆) 余位...
[Presentation Agent] 正在排版最终方案...

==================== 方案详情 ====================
# 下午行程建议（家庭场景）

## 第一部分：行程时间表

| 时间段         | 活动                         | 备注                                       |
|----------------|------------------------------|--------------------------------------------|
| 14:00-17:00    | 奇幻森林室内亲子乐园         | 室内活动，适合炎热天气，安全且有趣         |
| 17:00-18:00    | 休息/自由活动                | 可选择在附近购物或休息                     |
| 18:00-19:30    | 晚餐                         | 胡桃里音乐酒馆 (已预留4人位)               |

---

## 第二部分：方案亮点说明

1. **针对天气的调整**  
   根据天气炎热/有雨的情况，将原定的室外活动切换为室内活动“奇幻森林室内亲子乐园”。

2. **针对减脂/亲子需求的适配**  
   本行程安排了轻松有趣的室内亲子乐园，并匹配适合当前需求的餐厅。

3. **针对餐厅排队情况的优化**  
   已提前检查餐厅余位，避免到店后长时间等待。

---

## 第三部分：结束语

您对这个安排满意吗？如果没问题，我可以为您一键完成预订。
==================================================

[系统提示] 确定按照此方案执行一键下单吗？(y/n): y

[Execution Agent] 用户已确认，正在静默执行并发现预订请求...
[Mock Order] 娱乐门票预订成功！订单号: MT1778487348
[Mock Reservation] 餐厅预订成功！订单号: MT1778487349
[OK] 所有行程凭证已生成。

[DONE] 搞定了！所有订单已处理完成。
[MOCK] 详细凭证已发送至您的手机（模拟），您可以随时出发！
```

