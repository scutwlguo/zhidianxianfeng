# -*- coding: utf-8 -*-
"""
FastAPI + LangChain 多轮对话接口
功能：
1. 从用户自然语言中解析“用户X / 日期 / 分析意图”
2. 行为分析类问题：调用 GPT 用电分析输入拼装逻辑，生成 prompt + package 后送入阿里云模型
3. 普通问答类问题：直接把结构化数据送入模型回答
4. 使用 session 级多轮对话记忆，支持追问

运行：
uvicorn energy_chat_api:app --reload --host 0.0.0.0 --port 8000
"""

import json
import re
import os
import importlib.util
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from model_config import FIXED_MODEL_NAME, FIXED_PLATFORM


BASE_DIR = Path(__file__).resolve().parent

# 你已有的两个模块：一个负责创建 LLM，一个负责生成分析输入包
LLM_MODULE_PATH = BASE_DIR / "LLM_founction_set.py"
PACKER_MODULE_PATH = BASE_DIR / "GPT用电分析输入拼装_v2.py"

DEFAULT_MODEL_NAME = FIXED_MODEL_NAME
DEFAULT_PLATFORM = FIXED_PLATFORM

# ================================
# 阿里云 DashScope 固定模型配置
# ================================
# 公开仓库不要硬编码真实密钥，请通过环境变量 DASHSCOPE_API_KEY 配置。
ALIYUN_BASE_URL_EMBEDDED = "https://dashscope.aliyuncs.com/compatible-mode/v1"

os.environ.setdefault("ALIYUN_BASE_URL", ALIYUN_BASE_URL_EMBEDDED)
os.environ.setdefault("DMXAPI_URL", ALIYUN_BASE_URL_EMBEDDED)

# 与原拼装脚本保持一致；优先使用相对路径，兼容云端部署
DEFAULT_DAILY_JSON_ROOT_CANDIDATES = [
    os.getenv("APP_DAILY_JSON_ROOT", "").strip(),
    str(BASE_DIR.parent / "data" / "用电行为分析_json"),
    str(BASE_DIR / "data" / "用电行为分析_json"),
]
DEFAULT_DATASET = "REDD"
DEFAULT_HOUSE_DIR = "REDD_House6_stats"

# 运行期内存
SESSION_HISTORIES: Dict[str, InMemoryChatMessageHistory] = {}
SESSION_STATE: Dict[str, Dict] = {}
LLM_CACHE: Dict[Tuple[str, str, float, int], object] = {}


def _load_module(module_path: Path, module_name: str):
    if not module_path.exists():
        raise FileNotFoundError(f"模块文件不存在: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


llm_module = _load_module(LLM_MODULE_PATH, "llm_function_set_local")
packer_module = _load_module(PACKER_MODULE_PATH, "gpt_energy_packer_local")

create_llm = llm_module.create_llm
build_gpt_analysis_package = packer_module.build_gpt_analysis_package


def resolve_daily_json_root() -> str:
    for p in DEFAULT_DAILY_JSON_ROOT_CANDIDATES:
        if not p:
            continue
        path = Path(p)
        if path.exists() and path.is_dir():
            return str(path)
    raise FileNotFoundError(
        "未找到用电行为分析 JSON 根目录。请确认 data/用电行为分析_json 已随项目部署，"
        "或通过 APP_DAILY_JSON_ROOT 指定目录。"
    )


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in SESSION_HISTORIES:
        SESSION_HISTORIES[session_id] = InMemoryChatMessageHistory()
    return SESSION_HISTORIES[session_id]


def get_session_state(session_id: str) -> Dict:
    if session_id not in SESSION_STATE:
        SESSION_STATE[session_id] = {
            "dataset": DEFAULT_DATASET,
            "house_dir": DEFAULT_HOUSE_DIR,
            "last_intent": None,
        }
    return SESSION_STATE[session_id]


def normalize_house_dir(user_no: int, dataset: str = DEFAULT_DATASET) -> str:
    # 你现在的目录格式是 REDD_House6_stats
    return f"{dataset}_House{user_no}_stats"


def parse_user_no(text: str) -> Optional[int]:
    patterns = [
        r"用户\s*(\d+)",
        r"house\s*(\d+)",
        r"House\s*(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1))
    return None


def parse_date_from_text(text: str) -> Optional[str]:
    """
    支持：
    - 2026-04-22
    - 2026/4/22
    - 2026年4月22日
    - 26年4月22号
    - 4月22日（默认当前年）
    """
    text = text.strip()

    patterns = [
        r"(?P<y>\d{4})[-/年](?P<m>\d{1,2})[-/月](?P<d>\d{1,2})[日号]?",
        r"(?P<y>\d{2})年(?P<m>\d{1,2})月(?P<d>\d{1,2})[日号]?",
        r"(?P<m>\d{1,2})月(?P<d>\d{1,2})[日号]?",
    ]
    for idx, p in enumerate(patterns):
        m = re.search(p, text)
        if not m:
            continue
        gd = m.groupdict()
        if idx == 2:
            year = datetime.now().year
            month = int(gd["m"])
            day = int(gd["d"])
        else:
            year = int(gd["y"])
            if year < 100:
                year += 2000
            month = int(gd["m"])
            day = int(gd["d"])
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None


def parse_all_dates_from_text(text: str) -> List[str]:
    """
    从文本中提取多个日期，返回去重后按时间升序的 YYYY-MM-DD 列表。
    支持：2026-04-22 / 2026年4月22日 / 4月22日（默认当年）
    """
    text = (text or "").strip()
    if not text:
        return []

    patterns = [
        r"(?P<y>\d{4})[-/年](?P<m>\d{1,2})[-/月](?P<d>\d{1,2})[日号]?",
        r"(?P<y>\d{2})年(?P<m>\d{1,2})月(?P<d>\d{1,2})[日号]?",
        r"(?P<m>\d{1,2})月(?P<d>\d{1,2})[日号]?",
    ]

    out: List[str] = []
    for idx, p in enumerate(patterns):
        for m in re.finditer(p, text):
            gd = m.groupdict()
            try:
                if idx == 2:
                    year = datetime.now().year
                    month = int(gd["m"])
                    day = int(gd["d"])
                else:
                    year = int(gd["y"])
                    if year < 100:
                        year += 2000
                    month = int(gd["m"])
                    day = int(gd["d"])
                dt = datetime(year, month, day)
                out.append(dt.strftime("%Y-%m-%d"))
            except Exception:
                continue

    out = sorted(set(out))
    return out


def normalize_date_list(date_list: Optional[List[str]]) -> List[str]:
    """规范化日期列表为 YYYY-MM-DD，自动去重并排序。"""
    if not date_list:
        return []
    out = []
    for x in date_list:
        try:
            d = datetime.strptime(str(x).strip(), "%Y-%m-%d")
            out.append(d.strftime("%Y-%m-%d"))
        except Exception:
            continue
    return sorted(set(out))


def is_guidance_query(query: str) -> bool:
    q = re.sub(r"[\s\u3000]+", "", (query or "").strip().lower())
    if not q:
        return True

    guidance_keywords = [
        "你好", "您好", "hi", "hello", "在吗", "你是谁", "你能做什么", "有什么功能",
        "怎么用", "如何使用", "帮助", "help", "可以问什么", "能问什么", "提问示例",
    ]
    data_keywords = [
        "用电", "电量", "电费", "设备", "功率", "耗能", "节能", "异常", "风险",
        "趋势", "分析", "建议", "多少", "最大", "最小", "几点", "多久", "最近",
    ]
    return any(k in q for k in guidance_keywords) and not any(k in q for k in data_keywords)


def build_guidance_answer() -> str:
    return (
        "您好，我是智能用电助手，可以帮您快速理解家庭用电情况。\n\n"
        "您可以这样问：\n"
        "- 今天哪些设备最耗电？\n"
        "- 最近一周用电趋势怎么样？\n"
        "- 有没有异常运行或高风险时段？\n"
        "- 空调、热水器有什么节能建议？\n"
        "- 总电量和预估电费是多少？"
    )


def detect_intent(query: str, previous_intent: Optional[str] = None) -> Literal["assistant_guidance", "behavior_analysis", "data_qa"]:
    """
    粗粒度意图识别：
    - 行为分析 / 节能建议 / 报告 => behavior_analysis
    - 其他查数值、查事实 => data_qa
    """
    q = query.strip()
    if is_guidance_query(q):
        return "assistant_guidance"

    behavior_keywords = [
        "分析", "行为", "习惯", "节能", "建议", "报告", "高耗能", "异常运行",
        "画像", "优化", "结论", "能耗洞察", "帮我看看", "用电情况"
    ]
    qa_keywords = [
        "是什么", "多少", "最大", "最小", "哪个", "几点", "多久", "几次", "总电量", "总电费"
    ]

    if any(k in q for k in behavior_keywords):
        return "behavior_analysis"
    if any(k in q for k in qa_keywords):
        return "data_qa"
    return previous_intent or "data_qa"


def resolve_target_from_query(
    query: str,
    session_id: str,
    dataset_override: Optional[str] = None,
    house_dir_override: Optional[str] = None,
) -> Dict:
    state = get_session_state(session_id)

    dataset = dataset_override or state.get("dataset") or DEFAULT_DATASET
    user_no = parse_user_no(query)

    house_dir = house_dir_override or (normalize_house_dir(user_no, dataset) if user_no else state.get("house_dir"))

    if not house_dir:
        raise HTTPException(status_code=400, detail="未能从问题中解析到用户编号，也没有可复用的会话上下文。")

    state["dataset"] = dataset
    state["house_dir"] = house_dir

    return {
        "dataset": dataset,
        "house_dir": house_dir,
    }


@lru_cache(maxsize=96)
def load_single_day_package(dataset: str, house_dir: str, date_str: str) -> Tuple[Dict, str]:
    daily_json_root = resolve_daily_json_root()
    package, prompt_text = build_gpt_analysis_package(
        dataset=dataset,
        house_dir=house_dir,
        pack_type="single-day",
        date=date_str,
        daily_json_root=daily_json_root,
    )
    return package, prompt_text


@lru_cache(maxsize=48)
def load_multi_day_package_range(dataset: str, house_dir: str, start_date: str, end_date: str) -> Tuple[Dict, str]:
    daily_json_root = resolve_daily_json_root()
    package, prompt_text = build_gpt_analysis_package(
        dataset=dataset,
        house_dir=house_dir,
        pack_type="multi-day",
        start_date=start_date,
        end_date=end_date,
        daily_json_root=daily_json_root,
    )
    return package, prompt_text


def load_multi_day_package_dates(dataset: str, house_dir: str, date_list: List[str]) -> Tuple[Dict, str]:
    """按“多个指定日期”打包（可不连续）。"""
    dates = normalize_date_list(date_list)
    if len(dates) == 0:
        raise ValueError("dates 为空或格式不合法，请使用 YYYY-MM-DD")

    day_packages = []
    for d in dates:
        pkg, _ = load_single_day_package(dataset=dataset, house_dir=house_dir, date_str=d)
        day_packages.append(pkg)

    first_pkg = day_packages[0]
    reference_rules = first_pkg.get("reference_rules", {})
    user_info = (((first_pkg.get("record") or {}).get("user")) or {})
    user_label = ((first_pkg.get("target") or {}).get("user_label")) or user_info.get("user_name", house_dir)

    daily_records_with_evidence = []
    daily_records = []
    focus_pool = []
    hint_task = ""
    for p in day_packages:
        rec = (p.get("record") or {})
        day = rec.get("daily_record", {}) or {}
        evd = rec.get("appliance_evidence", []) or []
        daily_records.append(day)
        daily_records_with_evidence.append({
            "date": str(day.get("date", "")),
            "daily_record": day,
            "appliance_evidence": evd,
        })
        htask = ((p.get("task") or {}).get("hint_task") or "").strip()
        if (not hint_task) and htask:
            hint_task = htask
        hfocus = ((p.get("task") or {}).get("hint_focus") or [])
        if isinstance(hfocus, list):
            focus_pool.extend([str(x) for x in hfocus if str(x).strip()])

    dedup_focus = []
    seen = set()
    for x in focus_pool:
        if x not in seen:
            dedup_focus.append(x)
            seen.add(x)

    summarize_fn = getattr(packer_module, "_summarize_days", None)
    derived_summary = summarize_fn(daily_records) if callable(summarize_fn) else {
        "day_count": len(daily_records)
    }

    package = {
        "meta": {
            "schema_version": "2.1",
            "purpose": "for_gpt_energy_behavior_analysis",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "target": {
            "dataset": dataset,
            "scope": "selected_days_single_user",
            "house_dir": house_dir,
            "user_name": user_info.get("user_name", house_dir),
            "user_label": user_label,
        },
        "analysis_period": {
            "start_date": dates[0],
            "end_date": dates[-1],
            "day_count": len(dates),
            "selected_dates": dates,
        },
        "task": {
            "task_type": "selected_days_behavior_analysis",
            "evidence_strategy": "strong_signals_first_weak_signals_assist",
            "require_conservative_wording": True,
            "hint_task": hint_task,
            "hint_focus": dedup_focus,
        },
        "reference_rules": reference_rules,
        "derived_summary": derived_summary,
        "daily_records": daily_records,
        "daily_records_with_evidence": daily_records_with_evidence,
    }

    prompt_builder = getattr(packer_module, "_build_prompt_text", None)
    prompt_text = prompt_builder(package) if callable(prompt_builder) else "请基于给定多天用电数据进行分析并给出节能建议。"
    return package, prompt_text


def json_dumps_cn(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_behavior_system_prompt(prompt_text: str) -> str:
    return (
        f"{prompt_text}\n\n"
        "补充要求：\n"
        "1. 必须严格基于当前家庭用户的用电记录、证据与提示字段回答。\n"
        "2. 如用户后续追问某个设备或某条结论，请结合本轮数据继续解释。\n"
        "3. 若用户问题与节能建议相关，优先引用 strong_signals、risk_hints、suggestion_directions。\n"
        "4. 若证据不足，不要编造；请明确说明依据不足。\n"
        "5. 输出必须简短：总长度控制在 260 字以内，最多 5 条要点。\n"
        "6. 不要输出 Markdown 表格、JSON、代码块、长标题或英文变量名。\n"
        "7. 不要提及 REDD、数据集、JSON、结构化字段等技术或实验来源。"
    )


def build_qa_system_prompt() -> str:
    return (
        "你是一名家庭用电数据问答助手。\n"
        "请严格根据当前家庭用户的用电记录回答用户问题，不要编造数据。\n"
        "规则：\n"
        "1. 先看用户当前问题，再从当前记录中定位答案。\n"
        "2. 若问题只问某个事实（如最大耗电设备、总电量、开启次数），请直接给简洁答案。\n"
        "3. 若问题需要解释原因，可结合 appliance_evidence、risk_hints 做简要解释。\n"
        "4. 若会话中存在上文追问，请在不脱离当前记录的前提下承接上下文。\n"
        "5. 若当前记录中没有答案，请明确说明没有足够数据。\n"
        "6. 输出必须简短：总长度控制在 220 字以内，最多 4 条要点。\n"
        "7. 不要输出 Markdown 表格、JSON、代码块、长标题或英文变量名。\n"
        "8. 不要提及 REDD、数据集、JSON、结构化字段等技术或实验来源。"
    )


def make_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "【当前用电记录】\n{data_context}\n\n【用户问题】\n{user_query}"),
        ]
    )
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="user_query",
        history_messages_key="history",
    )
    return chain_with_history


def answer_with_model(
    llm,
    session_id: str,
    system_prompt: str,
    data_context: Dict,
    user_query: str,
) -> str:
    history = get_session_history(session_id)
    if len(history.messages) > 8:
        history.messages = history.messages[-8:]

    chain = make_chain(llm)
    result = chain.invoke(
        {
            "system_prompt": system_prompt,
            "data_context": json_dumps_cn(data_context),
            "user_query": user_query,
        },
        config={"configurable": {"session_id": session_id}},
    )
    return result.content if hasattr(result, "content") else str(result)


def get_cached_llm(platform: str, model_name: str, temperature: float, max_tokens: int):
    key = (platform, model_name, round(float(temperature), 3), int(max_tokens))
    if key not in LLM_CACHE:
        LLM_CACHE[key] = create_llm(
            platform=platform,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return LLM_CACHE[key]


def clean_model_answer(text: str, max_chars: int = 420) -> str:
    s = (text or "").strip()
    s = re.sub(r"```.*?```", "", s, flags=re.S)
    lines = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if line.startswith("|") and line.endswith("|"):
            continue
        if any(term in line for term in ["REDD", "数据集", "JSON", "结构化字段"]):
            line = line.replace("REDD", "").replace("数据集", "用电记录")
            line = line.replace("JSON", "记录").replace("结构化字段", "记录")
        lines.append(line)

    compact = "\n".join(lines).strip()
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip("，。；、,. ") + "。"
    return compact or "当前信息不足，请换个问题或指定日期后再试。"


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话ID，多轮对话依赖它")
    message: str = Field(..., description="用户自然语言问题")
    dataset: str = Field(default=DEFAULT_DATASET, description="默认 REDD")
    house_dir: str = Field(default=DEFAULT_HOUSE_DIR, description="默认 REDD_House6_stats")
    start_date: str = Field(default="2026-04-16", description="多天分析起始日期 YYYY-MM-DD")
    end_date: str = Field(default="2026-04-22", description="多天分析结束日期 YYYY-MM-DD")
    pack_mode: str = Field(default="auto", description="auto / single-day / multi-day-range")
    platform: str = Field(default=DEFAULT_PLATFORM, description="兼容字段（实际固定 aliyun）")
    model_name: str = Field(default=DEFAULT_MODEL_NAME, description="兼容字段（实际固定 qwen3.6-plus）")
    temperature: float = Field(default=0.2, description="推荐低温度，保证分析稳定")
    max_tokens: int = Field(default=500, description="最大输出 token")


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    resolved_target: Dict
    answer: str


app = FastAPI(title="家庭用电智能分析 API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = get_session_state(req.session_id)
    intent = detect_intent(req.message, state.get("last_intent"))

    start_date = (req.start_date or "").strip()
    end_date = (req.end_date or "").strip()
    if (not start_date) or (not end_date):
        raise HTTPException(status_code=400, detail="请提供 start_date 和 end_date，格式 YYYY-MM-DD")

    target = resolve_target_from_query(
        query=req.message,
        session_id=req.session_id,
        dataset_override=req.dataset,
        house_dir_override=req.house_dir,
    )

    # 固定平台与模型，避免前端传参影响
    fixed_platform = DEFAULT_PLATFORM
    fixed_model_name = DEFAULT_MODEL_NAME

    resolved_target = dict(target)
    resolved_target["platform"] = fixed_platform
    resolved_target["model_name"] = fixed_model_name

    if intent == "assistant_guidance":
        state["last_intent"] = None
        resolved_target["pack_type"] = "no-data"
        return ChatResponse(
            session_id=req.session_id,
            intent=intent,
            resolved_target=resolved_target,
            answer=build_guidance_answer(),
        )

    if fixed_platform == "aliyun" and not os.getenv("DASHSCOPE_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="缺少阿里云密钥：请配置环境变量 DASHSCOPE_API_KEY。",
        )

    llm = get_cached_llm(
        platform=fixed_platform,
        model_name=fixed_model_name,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    if llm is None:
        raise HTTPException(
            status_code=400,
            detail=f"模型初始化失败，请检查配置（platform={fixed_platform}, model={fixed_model_name}）。",
        )

    try:
        pack_mode = (req.pack_mode or "auto").strip().lower()
        if pack_mode == "single-day" or start_date == end_date:
            package, behavior_prompt = load_single_day_package(
                dataset=target["dataset"],
                house_dir=target["house_dir"],
                date_str=start_date,
            )
            resolved_target["pack_type"] = "single-day"
        else:
            package, behavior_prompt = load_multi_day_package_range(
                dataset=target["dataset"],
                house_dir=target["house_dir"],
                start_date=start_date,
                end_date=end_date,
            )
            resolved_target["pack_type"] = "multi-day-range"
        resolved_target["start_date"] = start_date
        resolved_target["end_date"] = end_date
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"加载数据失败: {e}")

    if intent == "behavior_analysis":
        system_prompt = build_behavior_system_prompt(behavior_prompt)
        data_context = package
    else:
        system_prompt = build_qa_system_prompt()
        # 普通问答也建议给 package，而不是只给 raw daily_record，这样模型能直接用 evidence
        data_context = package

    raw_answer = answer_with_model(
        llm=llm,
        session_id=req.session_id,
        system_prompt=system_prompt,
        data_context=data_context,
        user_query=req.message,
    )
    answer = clean_model_answer(raw_answer, max_chars=420 if intent == "behavior_analysis" else 340)

    state["last_intent"] = intent

    return ChatResponse(
        session_id=req.session_id,
        intent=intent,
        resolved_target=resolved_target,
        answer=answer,
    )


@app.get("/session/{session_id}")
def get_session(session_id: str):
    history = get_session_history(session_id)
    state = get_session_state(session_id)
    return {
        "session_id": session_id,
        "state": state,
        "history": [
            {"type": m.type, "content": m.content}
            for m in history.messages
        ],
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    SESSION_HISTORIES.pop(session_id, None)
    SESSION_STATE.pop(session_id, None)
    return {"session_id": session_id, "cleared": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("energy_chat_api:app", host="0.0.0.0", port=8000, reload=True)
