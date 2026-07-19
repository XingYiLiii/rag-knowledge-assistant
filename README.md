# rag-knowledge-assistant

## 项目介绍

`rag-knowledge-assistant` 是一个基于 FastAPI 和 RAG（Retrieval-Augmented Generation）的企业知识库问答系统。它将上传文档转化为可检索的向量知识，并基于检索到的证据生成回答与可追踪引用。

它解决以下知识库使用问题：

- 统一管理知识库与源文档。
- 解析、切片和检索 PDF、DOCX、Markdown、TXT 文档。
- 基于指定知识库中的证据回答问题。
- 为回答提供文档、页码/章节和匹配片段等 Citation 追踪信息。

## 核心功能

- Knowledge Base 创建、查询、更新、删除和统计。
- PDF / DOCX / Markdown / TXT 文档上传与解析。
- 可配置的文本清洗与 Chunk 切片。
- OpenAI-compatible Embedding Provider，支持通过配置切换兼容服务。
- Chroma 持久化向量存储与知识库隔离检索。
- Grounded RAG 问答、上下文预算和 Prompt 边界。
- Citation 引用快照与来源追踪。
- Conversation History：保存用户消息、助手回答和 Citation 快照。
- Jinja2 + 原生 JavaScript/CSS 的知识库管理、聊天和历史 Web UI。
- Docker Compose 单机部署与 SQLite、上传文件、Chroma 数据持久化。
- GitHub Actions CI：Push 和 Pull Request 自动运行 Ruff 与 pytest。

## 技术栈

| Area | Technologies |
| --- | --- |
| Backend | Python, FastAPI, SQLAlchemy, SQLite |
| AI / RAG | LangChain, Chroma, OpenAI-compatible API |
| Frontend | Jinja2, JavaScript, CSS |
| Deployment | Docker Compose, GitHub Actions |

## 快速启动

### 本地运行

```bash
cd backend
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --reload
```

可选地从根目录的 `.env.example` 创建本地 `.env`，并填写兼容服务的 Embedding / LLM 配置。上传文档或调用 Chat API 时需要相应的可用模型配置。

启动后访问：

- Web UI：<http://localhost:8000/>
- Swagger：<http://localhost:8000/docs>
- Health API：<http://localhost:8000/api/v1/health>

### Docker 运行

```bash
docker compose up --build
```

Docker Compose 将 SQLite、上传文件和 Chroma 数据分别保存到命名 Volume。Compose 在根目录 `.env` 存在时加载它；该文件不会进入镜像。

## API示例

### Chat API

```http
POST /api/v1/chat
Content-Type: application/json
```

请求字段：

- `knowledge_base_id`：必填，目标知识库 UUID。
- `question`：必填，最多 2000 个字符。
- `conversation_id`：可选，提供时保存用户消息、回答和 Citation 快照。

```json
{
  "knowledge_base_id": "<knowledge-base-uuid>",
  "question": "How do I deploy the project?"
}
```

响应包含：

- `answer`：基于检索上下文生成的回答。
- `model`：提供回答的模型名称；无检索结果时可以为空。
- `latency_ms`：本次 RAG 请求耗时。
- `citations`：用于当前回答的来源快照。

Citation 每项包含来源文档、可选页码或章节、检索分数和匹配文本片段。完整 Schema 请访问 Swagger。

## 测试

在 `backend/` 目录运行：

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

GitHub Actions 会在每次 Push 和 Pull Request 上执行 Ruff 检查、Ruff 格式检查和 pytest。CI 使用 `APP_ENV=test`，不需要 `.env`、真实 API Key、LLM 或 Embedding 服务。

## RAG评测

项目提供固定的轻量评测集与可重复执行的脚本，详见 [RAG Design and Evaluation](docs/rag-design.md)。当前指标包括：

- Recall@K
- Citation Source Accuracy
- Unanswerable Empty Retrieval Rate

评测只根据提供的结果文件或只读 Retriever 输出计算指标，不预置或虚构模型成绩。

## 安全说明

- API Key 不会被应用日志记录；常见凭据格式会在日志消息中脱敏。
- System Prompt 只包含可信系统规则。
- 用户问题与外部知识都被标记为不可信上下文，不能改变模型角色、系统规则或工具权限。
- Chat 请求限制问题长度，超长输入返回统一验证错误。

## 已知限制

- 当前没有用户认证。
- 当前没有权限系统或多租户隔离。
- 当前没有 SSE / Streaming 输出。
- 当前部署目标为单机 Docker Compose。

## Roadmap

- [ ] PostgreSQL 支持与数据库迁移。
- [ ] Redis / Celery 异步任务处理。
- [ ] 多用户与工作区。
- [ ] 认证与细粒度权限。
- [ ] Streaming Chat 响应。

## 文档

- [Architecture](docs/architecture.md)
- [RAG Design and Evaluation](docs/rag-design.md)