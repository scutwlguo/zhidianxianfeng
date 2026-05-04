# 智电先锋（Streamlit 版）

## 本地运行

```bash
pip install -r requirements.txt
streamlit run code/zhidian_xianfeng_app.py
```

如果当前目录已经是 `code`，请改用：

```bash
streamlit run zhidian_xianfeng_app.py
```

如果需要启用问答接口或阿里云 DashScope 模型能力，请先在本机或部署环境配置：

```bash
DASHSCOPE_API_KEY=your-dashscope-api-key
APP_LLM_PLATFORM=aliyun
APP_LLM_MODEL_NAME=qwen3.6-plus
```

## 云部署（Streamlit Community Cloud）

1. 将仓库推送到 GitHub。
2. 在 Streamlit Community Cloud 选择仓库与入口文件：`code/zhidian_xianfeng_app.py`。
3. 使用根目录 `requirements.txt` 安装依赖。
4. 在云环境中准备两类数据目录：
	- 按日分析 Excel 根目录（例如：`/app/data/output/按日分析结果_全部`）
	- 知识图谱导出目录（例如：`/app/data/output/kg_export`）
5. 在 App 的 `Settings` -> `Secrets` 中填写模型密钥和可选配置：

```toml
DASHSCOPE_API_KEY = "your-dashscope-api-key"
APP_LLM_PLATFORM = "aliyun"
APP_LLM_MODEL_NAME = "qwen3.6-plus"
```

6. 通过环境变量或 Streamlit Secrets 指定目录（推荐）：
	- `APP_DATA_ROOT=/app/data/output/按日分析结果_全部`
	- `APP_KG_ROOT=/app/data/output/kg_export`

## 数据源模式

应用右上角“设置”支持：

- 按日分析结果根目录（Excel）
- 知识图谱文件目录（本地导出）

部署到云端后，这两个目录都应为“云服务器上的路径”。

## Neo4j 配置（可选）

当前前端知识图谱支持“本地导出文件读取”，不必在线连接 Neo4j。

### 1) 先从 Neo4j 导出图谱文件

先在 [code/export_neo4j_kg.py](code/export_neo4j_kg.py) 中修改变量：

- `DATA_ROOT`
- `OUT_ROOT`
- `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DB`
- `ONLY_DATASET` / `ONLY_HOUSE`（可选）

然后执行：

```bash
python code/export_neo4j_kg.py
```

导出后目录示例：

```text
kg_export/
	REDD/
		House1_stats.json
	UK-DALE/
		House2_stats.json
```

### 2) 在页面设置里填写“知识图谱文件目录”

应用右上角“设置” -> `知识图谱文件目录（本地导出）` 填写：

```text
F:/研究生文件/节能减排/kg_export
```

即可保持原展示效果（节点/边/属性面板），但不再依赖前端直连 Neo4j。
