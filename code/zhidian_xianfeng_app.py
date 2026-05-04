import os
import re
import random
import json
import time
import importlib
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# 基础配置
# ============================================================
st.set_page_config(
    page_title="智电先锋",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _sync_streamlit_secrets_to_env() -> None:
    """Let Streamlit Cloud Secrets feed the modules that read os.environ."""
    try:
        secrets = st.secrets
    except Exception:
        return

    keys = (
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "DMX_API_KEY",
        "ALIYUN_BASE_URL",
        "DMXAPI_URL",
        "APP_LLM_PLATFORM",
        "APP_LLM_MODEL_NAME",
        "APP_ENERGY_CHAT_API_URL",
        "APP_DATA_ROOT",
        "APP_KG_ROOT",
        "APP_DAILY_JSON_ROOT",
    )
    sections = ("api_keys", "llm", "app")

    for key in keys:
        value = None
        try:
            if key in secrets:
                value = secrets[key]
        except Exception:
            value = None

        if value is None:
            for section in sections:
                try:
                    if section in secrets and key in secrets[section]:
                        value = secrets[section][key]
                        break
                except Exception:
                    continue

        value_text = str(value).strip() if value is not None else ""
        if value_text and not os.getenv(key):
            os.environ[key] = value_text


_sync_streamlit_secrets_to_env()

from model_config import FIXED_MODEL_NAME, FIXED_PLATFORM


PRICE_PER_KWH = 0.7
SUPPORTED_DATASET_NAMES = ["REDD", "UK-DALE", "REFIT"]
DEFAULT_PASSWORD = "xxxx"
CHAT_PANEL_HEIGHT = 1425
CHAT_MESSAGES_HEIGHT = 1150
DEFAULT_ROOT_CANDIDATES = [
    os.getenv("APP_DATA_ROOT", "").strip(),
    r"F:/研究生文件/节能减排/云端功率分析代码/output/按日分析结果_全部",
]
DEFAULT_KG_ROOT_CANDIDATES = [
    os.getenv("APP_KG_ROOT", "").strip(),
    r"F:/研究生文件/节能减排/云端功率分析代码/output/kg_export",
]
ENERGY_CHAT_API_URL = os.getenv("APP_ENERGY_CHAT_API_URL", "http://127.0.0.1:8000/chat").strip()


# ============================================================
# 数据结构
# ============================================================
@dataclass
class HouseInfo:
    dataset_name: str
    house_id: str
    display_name: str
    address: str = "xxxx"
    account: str = "xxx"
    password: str = DEFAULT_PASSWORD


# ============================================================
# 工具函数
# ============================================================
def ensure_session_defaults() -> None:
    guessed_root = ""
    for p in DEFAULT_ROOT_CANDIDATES:
        if p and Path(p).exists():
            guessed_root = p
            break

    guessed_kg_root = ""
    for p in DEFAULT_KG_ROOT_CANDIDATES:
        if p and Path(p).exists():
            guessed_kg_root = p
            break

    defaults = {
        "configured_root": guessed_root,
        "kg_data_root": guessed_kg_root,
        "dataset_name": "",
        "selected_house": "",
        "logged_in": False,
        "autologin_applied": False,
        "enable_chat_api": True,
        "tier_level": 1,
        "selected_day": None,
        "date_range": (),
        "chat_messages": [
            {
                "role": "assistant",
                "content": "您好，我是智电先锋的用电智能助手，可以为您提供节能建议、异常解释和设备用电分析。",
            }
        ],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def extract_house_number(name: str) -> str:
    match = re.search(r"House[_\-]?(\d+)", name, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", name)
    return digits[-1] if digits else "1"


def resolve_house_by_account(account_input: str, houses: List[Dict[str, str]]) -> Optional[str]:
    account_text = (account_input or "").strip()
    if not account_text:
        return None

    normalized = re.sub(r"\s+", "", account_text).lower()
    digits = re.findall(r"\d+", normalized)
    if digits:
        target_no = digits[-1].lstrip("0") or "0"
        for h in houses:
            house_no = str(h.get("house_no", extract_house_number(h.get("house_key", "1"))))
            if (house_no.lstrip("0") or "0") == target_no:
                return h.get("house_key")

    for h in houses:
        display_name = re.sub(r"\s+", "", str(h.get("display_name", ""))).lower()
        house_key = re.sub(r"\s+", "", str(h.get("house_key", ""))).lower()
        if normalized in {display_name, house_key}:
            return h.get("house_key")

    return None


def _normalize_question(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[\s\u3000]+", "", s)
    s = re.sub(r"[，。！？；：、,.!?;:'\"（）()\[\]【】<>《》]", "", s)
    return s


def _extract_dates_from_text_for_api(text: str) -> List[str]:
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
    return sorted(set(out))


def _is_guidance_query(text: str) -> bool:
    q = _normalize_question(text)
    if not q:
        return True

    guidance_phrases = [
        "你好", "您好", "hi", "hello", "在吗", "你是谁", "你能做什么", "有什么功能",
        "怎么用", "如何使用", "帮助", "help", "可以问什么", "能问什么", "提问示例",
    ]
    data_phrases = [
        "用电", "电量", "电费", "设备", "空调", "热水器", "洗衣机", "烘干机", "电饭煲",
        "冰箱", "照明", "功率", "耗能", "节能", "异常", "风险", "趋势", "分析", "建议",
        "多少", "最大", "最小", "几点", "多久", "最近", "今天", "昨天",
    ]
    return any(p in q for p in guidance_phrases) and not any(p in q for p in data_phrases)


def _build_guidance_response() -> str:
    return (
        "您好，我是智能用电助手，可以帮您快速理解家庭用电情况。\n\n"
        "您可以这样问：\n"
        "- 今天哪些设备最耗电？\n"
        "- 最近一周用电趋势怎么样？\n"
        "- 有没有异常运行或高风险时段？\n"
        "- 空调、热水器这类设备有什么节能建议？\n"
        "- 总电量和预估电费是多少？"
    )


def _query_prefers_multi_day(question: str) -> bool:
    q = _normalize_question(question)
    multi_day_keywords = [
        "最近", "近一周", "一周", "7天", "七天", "这几天", "多天", "本周", "上周",
        "趋势", "变化", "对比", "总览", "整体", "累计", "平均", "波动",
    ]
    return any(k in q for k in multi_day_keywords)


def _resolve_api_date_window(question: str, selected_date, max_date) -> Tuple[str, str, str]:
    dates = _extract_dates_from_text_for_api(question)
    if dates:
        if len(dates) == 1:
            return dates[0], dates[0], "single-day"
        return min(dates), max(dates), "multi-day-range"

    if _query_prefers_multi_day(question):
        end_dt = pd.to_datetime(max_date).normalize()
        start_dt = end_dt - pd.Timedelta(days=6)
        return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), "multi-day-range"

    day_dt = pd.to_datetime(selected_date or max_date).normalize()
    day = day_dt.strftime("%Y-%m-%d")
    return day, day, "single-day"


def _house_dir_for_api_from_house_key(house_key: str) -> str:
    house_no = extract_house_number(house_key)
    try:
        no = int(house_no)
    except Exception:
        no = 1
    no = min(max(no, 1), 6)
    return f"REDD_House{no}_stats"


def _format_date_chip(start_date, end_date=None) -> str:
    try:
        start_dt = pd.to_datetime(start_date)
        if end_date is None:
            return f"{start_dt.month}月{start_dt.day}日"
        end_dt = pd.to_datetime(end_date)
        if start_dt.date() == end_dt.date():
            return f"{start_dt.month}月{start_dt.day}日"
        if start_dt.year == end_dt.year:
            return f"{start_dt.month}月{start_dt.day}日 - {end_dt.month}月{end_dt.day}日"
        return f"{start_dt.strftime('%Y-%m-%d')} - {end_dt.strftime('%Y-%m-%d')}"
    except Exception:
        return str(start_date)


def _call_energy_chat_local_direct(
    user_query: str,
    house_key: str,
    start_date: str,
    end_date: str,
    pack_mode: str,
    session_id: str,
) -> Optional[str]:
    """本地直调 energy_chat_api.chat（无需单独启动 uvicorn）。"""
    try:
        mod = importlib.import_module("energy_chat_api")
        # 避免模块缓存导致配置更新后不生效（如切换阿里云变量名）
        mod = importlib.reload(mod)
        req_cls = getattr(mod, "ChatRequest", None)
        chat_fn = getattr(mod, "chat", None)
        if req_cls is None or chat_fn is None:
            return None

        req_obj = req_cls(
            session_id=session_id,
            message=user_query,
            dataset="REDD",
            house_dir=_house_dir_for_api_from_house_key(house_key),
            start_date=start_date,
            end_date=end_date,
            pack_mode=pack_mode,
            platform=FIXED_PLATFORM,
            model_name=FIXED_MODEL_NAME,
            temperature=0.2,
            max_tokens=800,
        )
        resp = chat_fn(req_obj)

        if isinstance(resp, dict):
            ans = str(resp.get("answer", "")).strip()
            return ans or None
        ans = str(getattr(resp, "answer", "")).strip()
        return ans or None
    except Exception as e:
        st.session_state["last_chat_api_error"] = f"local_direct_failed: {e}"
        return None


def call_energy_chat_api(
    user_query: str,
    house_key: str,
    selected_date,
    max_available_date,
    session_id: str,
) -> str:
    if _is_guidance_query(user_query):
        return _build_guidance_response()

    start_date, end_date, pack_mode = _resolve_api_date_window(user_query, selected_date, max_available_date)
    payload = {
        "session_id": session_id,
        "message": user_query,
        "dataset": "REDD",
        "house_dir": _house_dir_for_api_from_house_key(house_key),
        "start_date": start_date,
        "end_date": end_date,
        "pack_mode": pack_mode,
        "platform": FIXED_PLATFORM,
        "model_name": FIXED_MODEL_NAME,
        "temperature": 0.2,
        "max_tokens": 800,
    }

    if "127.0.0.1" in ENERGY_CHAT_API_URL or "localhost" in ENERGY_CHAT_API_URL:
        local_answer = _call_energy_chat_local_direct(
            user_query=user_query,
            house_key=house_key,
            start_date=start_date,
            end_date=end_date,
            pack_mode=pack_mode,
            session_id=session_id,
        )
        if local_answer:
            return local_answer

    req = urllib.request.Request(
        url=ENERGY_CHAT_API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(body) if body else {}
        answer = str(data.get("answer", "")).strip()
        return answer if answer else "超出问答范围"
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="ignore")
            err_data = json.loads(err_body) if err_body else {}
            err_detail = str(err_data.get("detail", "")).strip()
        except Exception:
            err_detail = ""

        st.session_state["last_chat_api_error"] = f"HTTPError: {e}; detail={err_detail or err_body}"
        # HTTP 有响应但失败，尝试本地直调兜底
        local_answer = _call_energy_chat_local_direct(
            user_query=user_query,
            house_key=house_key,
            start_date=start_date,
            end_date=end_date,
            pack_mode=pack_mode,
            session_id=session_id,
        )
        if local_answer:
            return local_answer
        if err_detail:
            return f"智能助手调用失败：{err_detail}"
        if err_body:
            return f"智能助手调用失败：{err_body}"
        return "智能助手暂时繁忙，请稍后再试。"
    except Exception as e:
        st.session_state["last_chat_api_error"] = str(e)
        # 典型场景：127.0.0.1:8000 未启动；尝试本地直调兜底
        local_answer = _call_energy_chat_local_direct(
            user_query=user_query,
            house_key=house_key,
            start_date=start_date,
            end_date=end_date,
            pack_mode=pack_mode,
            session_id=session_id,
        )
        if local_answer:
            return local_answer
        return "智能助手暂时繁忙，请稍后再试。"


def beautify_assistant_text(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"```.*?```", "", s, flags=re.S)
    # 保持“1、2、3”原样，避免被 Markdown 有序列表自动续号导致编号错乱。
    kept_lines = []
    for raw in s.splitlines():
        line = raw.strip()
        if line.startswith("|") and line.endswith("|"):
            continue
        line = line.replace("REDD", "").replace("数据集", "用电记录")
        line = line.replace("JSON", "记录").replace("结构化字段", "记录")
        kept_lines.append(line)
    s = "\n".join(kept_lines)
    s = re.sub(r"(?m)^\s*[-•]\s*", "- ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    if len(s) > 460:
        s = s[:460].rstrip("，。；、,. ") + "。"
    return s


@st.cache_data(show_spinner=False)
def load_qa_pairs_from_md() -> List[Tuple[str, str]]:
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "问答对设计.md",
        Path.cwd() / "data" / "问答对设计.md",
    ]
    qa_file = next((p for p in candidates if p.exists()), None)
    if qa_file is None:
        return []

    text = qa_file.read_text(encoding="utf-8")
    lines = text.splitlines()

    pairs: List[Tuple[str, str]] = []
    current_q: Optional[str] = None
    current_a_lines: List[str] = []

    def _flush_pair() -> None:
        nonlocal current_q, current_a_lines, pairs
        if current_q:
            ans = "\n".join(current_a_lines).strip()
            if ans:
                pairs.append((current_q.strip(), ans))

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("用户：") or stripped.startswith("用户:"):
            _flush_pair()
            current_q = stripped.split("：", 1)[1].strip() if "：" in stripped else stripped.split(":", 1)[1].strip()
            current_a_lines = []
            continue

        if current_q is None:
            continue

        if stripped.startswith("小智：") or stripped.startswith("小智:"):
            content = stripped.split("：", 1)[1].strip() if "：" in stripped else stripped.split(":", 1)[1].strip()
            current_a_lines.append(content)
            continue

        current_a_lines.append(line)

    _flush_pair()
    return pairs


def match_answer_from_qa(user_question: str) -> Optional[str]:
    pairs = load_qa_pairs_from_md()
    if not pairs:
        return None

    nq = _normalize_question(user_question)
    if not nq:
        return None

    # 1) 归一化后精确匹配
    for q, a in pairs:
        if _normalize_question(q) == nq:
            return a

    # 2) 包含匹配（兼容少量前后缀）
    for q, a in pairs:
        qq = _normalize_question(q)
        if qq and (qq in nq or nq in qq):
            return a

    return None


def stream_text_chunks(text: str, chunk_size: int = 12, delay: float = 0.02):
    content = text or ""
    for i in range(0, len(content), chunk_size):
        yield content[i:i + chunk_size]
        time.sleep(delay)


def _build_demo_price_24h(tier_level: int = 1) -> np.ndarray:
    """
    按“先分时、后阶梯”的规则构造24小时电价（元/kWh）。

    分时：
    - 峰: 0.99496875 元/kWh，时段 10-12、14-19
    - 平: 0.58886875 元/kWh，其余平段
    - 谷: 0.22916875 元/kWh，时段 0-8

    阶梯加价：
    - 第一档 +0.00
    - 第二档 +0.05
    - 第三档 +0.30
    """
    peak = 0.99496875
    flat = 0.58886875
    valley = 0.22916875

    price_24h = np.full(24, flat, dtype=float)
    price_24h[0:8] = valley
    price_24h[10:12] = peak
    price_24h[14:19] = peak

    tier_increment_map = {1: 0.0, 2: 0.05, 3: 0.30}
    tier_increment = tier_increment_map.get(int(tier_level), 0.0)
    return price_24h + tier_increment


def _extract_total_kwh_from_daily_sheets(daily: Dict[str, pd.DataFrame]) -> float:
    total_kwh = 0.0
    if "energy_bar_data" in daily and not daily["energy_bar_data"].empty:
        total_kwh = pd.to_numeric(daily["energy_bar_data"].get("energy_kwh"), errors="coerce").fillna(0).sum()
    elif "event_summary" in daily and not daily["event_summary"].empty:
        total_kwh = pd.to_numeric(daily["event_summary"].get("energy_kwh"), errors="coerce").fillna(0).sum()
    return float(total_kwh)


def _estimate_hourly_kwh_from_total_power_curve(total_df: pd.DataFrame) -> np.ndarray:
    hourly_kwh = np.zeros(24, dtype=float)
    if total_df.empty or "T" not in total_df.columns or "总功率" not in total_df.columns:
        return hourly_kwh

    t = pd.to_numeric(total_df["T"], errors="coerce")
    p = pd.to_numeric(total_df["总功率"], errors="coerce")
    df = pd.DataFrame({"T": t, "P": p}).dropna()
    if df.empty:
        return hourly_kwh

    df = df.sort_values("T").reset_index(drop=True)
    dt = df["T"].shift(-1) - df["T"]
    positive = dt[dt > 0]
    fallback_dt = float(positive.median()) if not positive.empty else 1.0
    dt = dt.fillna(fallback_dt).clip(lower=1, upper=120)

    hour = np.floor(pd.to_numeric(df["T"], errors="coerce") / 60.0).clip(0, 23).astype(int)
    energy = pd.to_numeric(df["P"], errors="coerce").fillna(0).to_numpy() * dt.to_numpy() / 60.0
    tmp = pd.DataFrame({"hour": hour, "kwh": energy})
    grouped = tmp.groupby("hour", as_index=True)["kwh"].sum()
    for h, v in grouped.items():
        if 0 <= int(h) <= 23:
            hourly_kwh[int(h)] = float(v)
    return hourly_kwh


def _estimate_hourly_kwh_from_event_ratio(hour_df: pd.DataFrame, total_kwh: float) -> np.ndarray:
    hourly_kwh = np.zeros(24, dtype=float)
    if hour_df.empty or "hour" not in hour_df.columns or total_kwh <= 0:
        return hourly_kwh

    h = pd.to_numeric(hour_df["hour"], errors="coerce").fillna(-1).astype(int)
    if "event_ratio" in hour_df.columns:
        w = pd.to_numeric(hour_df["event_ratio"], errors="coerce").fillna(0.0)
    elif "event_count" in hour_df.columns:
        w = pd.to_numeric(hour_df["event_count"], errors="coerce").fillna(0.0)
    else:
        w = pd.Series(np.ones(len(hour_df)), index=hour_df.index)

    tmp = pd.DataFrame({"hour": h, "w": w})
    tmp = tmp[(tmp["hour"] >= 0) & (tmp["hour"] <= 23)]
    if tmp.empty:
        return hourly_kwh

    grouped = tmp.groupby("hour", as_index=True)["w"].sum()
    total_w = float(grouped.sum())
    if total_w <= 0:
        return hourly_kwh

    for h_idx, wv in grouped.items():
        hourly_kwh[int(h_idx)] = float(total_kwh) * float(wv) / total_w
    return hourly_kwh


def _compute_daily_tou_cost_from_sheets(daily: Dict[str, pd.DataFrame], tier_level: int = 1) -> float:
    prices = _build_demo_price_24h(tier_level=tier_level)
    total_kwh = _extract_total_kwh_from_daily_sheets(daily)
    if total_kwh <= 0:
        return 0.0

    total_df = daily.get("total_power_curve", pd.DataFrame())
    hour_df = daily.get("hour_event_ratio", pd.DataFrame())

    hourly_kwh = _estimate_hourly_kwh_from_total_power_curve(total_df)
    # 关键修正：total_power_curve 的功率单位/时间单位可能与 kWh 口径不一致，
    # 这里仅把它当作“小时分布权重”，再归一化到当日总电量，避免电费被数量级放大。
    sum_hourly = float(hourly_kwh.sum())
    if sum_hourly > 0:
        hourly_kwh = hourly_kwh * (float(total_kwh) / sum_hourly)

    if float(hourly_kwh.sum()) <= 0:
        hourly_kwh = _estimate_hourly_kwh_from_event_ratio(hour_df, total_kwh)
    if float(hourly_kwh.sum()) <= 0 and total_kwh > 0:
        hourly_kwh = np.full(24, float(total_kwh) / 24.0, dtype=float)

    return float(np.sum(hourly_kwh * prices))


@st.cache_data(show_spinner=False)
def scan_datasets(root_dir: str) -> Dict[str, List[Dict[str, str]]]:
    root = Path(root_dir)
    result: Dict[str, List[Dict[str, str]]] = {}
    if not root.exists() or not root.is_dir():
        return result

    def _collect_houses(dataset_dir: Path) -> List[Dict[str, str]]:
        houses: List[Dict[str, str]] = []
        for house_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            excel_files = sorted(house_dir.glob("*.xlsx"))
            if not excel_files:
                continue
            house_no = extract_house_number(house_dir.name)
            houses.append(
                {
                    "house_key": house_dir.name,
                    "house_no": house_no,
                    "display_name": f"用户{house_no}",
                    "house_path": str(house_dir),
                }
            )
        return houses

    # 情况1：root 为“按日分析结果总根目录”（下级是 REDD/UK-DALE/REFIT）
    dataset_dirs = [p for p in root.iterdir() if p.is_dir() and p.name in SUPPORTED_DATASET_NAMES]
    if dataset_dirs:
        for dataset_dir in sorted(dataset_dirs):
            houses = _collect_houses(dataset_dir)
            if houses:
                result[dataset_dir.name] = houses
        return result

    # 情况2：root 直接是某个数据集目录（如 .../按日分析结果_全部/REDD）
    if root.name in SUPPORTED_DATASET_NAMES:
        houses = _collect_houses(root)
        if houses:
            result[root.name] = houses
        return result

    # 情况3：root 直接是某个用户目录（目录内就是每日xlsx）
    house_excels = sorted(root.glob("*.xlsx"))
    if house_excels:
        house_no = extract_house_number(root.name)
        dataset_name = root.parent.name if root.parent.name in SUPPORTED_DATASET_NAMES else "CUSTOM"
        result[dataset_name] = [
            {
                "house_key": root.name,
                "house_no": house_no,
                "display_name": f"用户{house_no}",
                "house_path": str(root),
            }
        ]

    return result


@st.cache_data(show_spinner=False)
def scan_house_dates(house_dir: str) -> List[pd.Timestamp]:
    path = Path(house_dir)
    dates: List[pd.Timestamp] = []
    if not path.exists():
        return dates
    for file in sorted(path.glob("*.xlsx")):
        try:
            dt = pd.to_datetime(file.stem)
            if not pd.isna(dt):
                dates.append(pd.Timestamp(dt).normalize())
        except Exception:
            continue
    dates = sorted(set(dates))
    return dates


@st.cache_data(show_spinner=False)
def load_daily_excel(file_path: str) -> Dict[str, pd.DataFrame]:
    excel = pd.ExcelFile(file_path)
    sheets = {}
    for sheet in excel.sheet_names:
        sheets[sheet] = pd.read_excel(file_path, sheet_name=sheet)
    return sheets


@st.cache_data(show_spinner=False)
def load_range_summary(house_dir: str, start_date: str, end_date: str, tier_level: int = 1) -> pd.DataFrame:
    path = Path(house_dir)
    rows = []
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()

    for file in sorted(path.glob("*.xlsx")):
        try:
            day = pd.to_datetime(file.stem).normalize()
        except Exception:
            continue
        if day < start_dt or day > end_dt:
            continue

        daily = load_daily_excel(str(file))
        total_kwh = _extract_total_kwh_from_daily_sheets(daily)
        total_cost = _compute_daily_tou_cost_from_sheets(daily, tier_level=tier_level)

        rows.append(
            {
                "date": day,
                "date_str": day.strftime("%Y-%m-%d"),
                "total_kwh": float(total_kwh),
                "cost": float(total_cost),
                "file_path": str(file),
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["date", "date_str", "total_kwh", "cost", "file_path"]
    )


@st.cache_data(show_spinner=False)
def build_alert_records(house_dir: str) -> pd.DataFrame:
    # 按业务要求给出固定风险洞察（并按最新日期优先展示）
    rows = [
        {
            "date": "2026-04-21",
            "time": "07:30",
            "level": "高",
            "device": "电热水器",
            "message": "检测到早上和傍晚时段连续高功率运行超过 4 小时，建议优化加热时段与温度设定，优先采用“集中加热+及时关闭”的方式。",
        },
        {
            "date": "2026-04-20",
            "time": "21:10",
            "level": "中",
            "device": "电饭煲",
            "message": "检测到晚间长时间运行，可能处于保温或使用后未及时关闭状态，建议缩短保温时长并在饭后及时断电。",
        },
        {
            "date": "2026-04-19",
            "time": "23:20",
            "level": "高",
            "device": "空调",
            "message": "检测到单日累计运行时长超过 12 小时，持续高负荷运行会显著增加能耗，建议开启睡眠模式并设置分时停机策略。",
        },
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "time", "level", "device", "message"])

    # 最新在最上面
    dt = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.assign(_dt=dt).sort_values("_dt", ascending=False).drop(columns=["_dt"]).reset_index(drop=True)
    return df


def get_house_info(dataset_name: str, house_key: str) -> HouseInfo:
    house_no = extract_house_number(house_key)
    return HouseInfo(
        dataset_name=dataset_name,
        house_id=house_key,
        display_name=f"用户{house_no}",
        account=f"用户{house_no}",
    )


@st.cache_data(show_spinner=False, ttl=120)
def load_house_kg_data_for_ui(user_name: str, dataset_name: str = "", house_key: str = "", kg_root: str = "") -> Dict[str, object]:
    """从本地导出文件读取用户图谱（JSON/CSV），不依赖 Neo4j 连接。"""

    def _pick(d: Dict, keys: List[str], default=""):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    def _as_str(v):
        return "" if v is None else str(v)

    candidate_roots: List[Path] = []
    if kg_root:
        candidate_roots.append(Path(kg_root))
    if st.session_state.get("kg_data_root"):
        candidate_roots.append(Path(str(st.session_state.get("kg_data_root"))))
    if st.session_state.get("configured_root"):
        base = Path(str(st.session_state.get("configured_root")))
        candidate_roots.append(base / "kg_export")
        candidate_roots.append(base / "knowledge_graph")

    # 去重且仅保留存在目录
    dedup_roots: List[Path] = []
    seen = set()
    for p in candidate_roots:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            dedup_roots.append(p)

    def _build_result(user_row: Dict, apps: List[Dict], edges: List[Dict]) -> Dict[str, object]:
        return {
            "ok": True,
            "user": {
                "name": _as_str(_pick(user_row, ["name", "名称", "user", "用户", "display_name"], user_name)),
                "avg_kwh": _as_str(_pick(user_row, ["avg_kwh", "平均日用电量（kwh）", "平均日用电量(kwh)", "平均日用电量", "average_kwh"], "")),
                "days": _as_str(_pick(user_row, ["days", "统计天数", "day_count"], "")),
            },
            "appliances": [
                {
                    "name": _as_str(_pick(a, ["name", "名称", "appliance", "device"])),
                    "weekday_periods": _as_str(_pick(a, ["weekday_periods", "工作日开启时段", "workday_periods"])),
                    "weekend_periods": _as_str(_pick(a, ["weekend_periods", "周末开启时段"])),
                    "rated_power": _as_str(_pick(a, ["rated_power", "额定功率", "power"])),
                }
                for a in apps
                if _as_str(_pick(a, ["name", "名称", "appliance", "device"]))
            ],
            "edges": [
                {
                    "source": _as_str(_pick(e, ["source", "from", "起点", "源"], "")),
                    "target": _as_str(_pick(e, ["target", "to", "终点", "目标"], "")),
                    "weekday_overlap": _as_str(_pick(e, ["weekday_overlap", "工作日重叠时段文本", "工作日重叠"])),
                    "weekend_overlap": _as_str(_pick(e, ["weekend_overlap", "周末重叠时段文本", "周末重叠"])),
                }
                for e in edges
                if _as_str(_pick(e, ["source", "from", "起点", "源"], "")) and _as_str(_pick(e, ["target", "to", "终点", "目标"], ""))
            ],
        }

    for root in dedup_roots:
        # 1) JSON 单文件模式
        json_candidates = [
            root / dataset_name / f"{house_key}.json",
            root / dataset_name / f"{user_name}.json",
            root / f"{house_key}.json",
            root / f"{user_name}.json",
        ]
        for jf in json_candidates:
            if not jf.exists():
                continue
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if isinstance(data, dict) and {"user", "appliances", "edges"}.issubset(set(data.keys())):
                    return _build_result(data.get("user", {}), data.get("appliances", []), data.get("edges", []))

                # 兼容 nodes/edges 结构
                nodes = data.get("nodes", []) if isinstance(data, dict) else []
                edges = data.get("edges", []) if isinstance(data, dict) else []
                user_node = {}
                app_nodes: List[Dict] = []
                for n in nodes:
                    ntype = _as_str(_pick(n, ["type", "类型", "label_type"], "")).lower()
                    name = _as_str(_pick(n, ["name", "名称", "label"], ""))
                    props = n.get("props", {}) if isinstance(n.get("props", {}), dict) else {}
                    merged = {**props, **n}
                    if "用户" in ntype or ntype == "user" or name == user_name:
                        user_node = merged
                    else:
                        app_nodes.append(merged)
                if user_node or app_nodes:
                    return _build_result(user_node or {"name": user_name}, app_nodes, edges)
            except Exception:
                continue

        # 2) CSV 目录模式
        dir_candidates = [
            root / dataset_name / house_key,
            root / house_key,
            root / user_name,
            root,
        ]
        for d in dir_candidates:
            if not d.exists() or not d.is_dir():
                continue
            user_csv = d / "user.csv"
            apps_csv = d / "appliances.csv"
            edges_csv = d / "edges.csv"
            if user_csv.exists() and apps_csv.exists() and edges_csv.exists():
                try:
                    user_df = pd.read_csv(user_csv)
                    apps_df = pd.read_csv(apps_csv)
                    edges_df = pd.read_csv(edges_csv)
                    user_row = user_df.iloc[0].to_dict() if not user_df.empty else {"name": user_name}
                    return _build_result(user_row, apps_df.to_dict("records"), edges_df.to_dict("records"))
                except Exception:
                    continue

    return {
        "ok": False,
        "error": (
            "未找到本地图谱文件。请先导出并放置："
            "1) <kg_root>/<dataset>/<house>.json（或 <kg_root>/<house>.json），"
            "或 2) user.csv + appliances.csv + edges.csv。"
        ),
    }


def render_house_kg_panel(user_name: str, dataset_name: str = "", house_key: str = "", height: int = 548) -> None:
    """在 Streamlit 中渲染动态知识图谱（Neo4j 风格：标签居中、点击弹出属性面板）。"""
    data = load_house_kg_data_for_ui(
        user_name=user_name,
        dataset_name=dataset_name,
        house_key=house_key,
        kg_root=str(st.session_state.get("kg_data_root", "")),
    )
    if not data.get("ok"):
        st.info(str(data.get("error", "知识图谱暂不可用")))
        return

    import json as _json

    user = data["user"]
    appliances = data["appliances"]
    overlaps = data["edges"]

    # 构建 vis.js 节点和边数据
    nodes_list = []
    edges_list = []

    user_id = f"USER::{user['name']}"
    nodes_list.append({
        "id": user_id,
        "label": str(user["name"]),
        "color": {"background": "#c4a4d6", "border": "#b08cc2",
                  "highlight": {"background": "#d4b8e3", "border": "#c4a4d6"}},
        "shape": "circle",
        "size": 38,
        "font": {"color": "#ffffff", "size": 14, "face": "Microsoft YaHei, sans-serif", "bold": "true"},
        "props": {
            "类型": "用户",
            "名称": user["name"],
            "平均日用电量(kWh)": str(user.get("avg_kwh", "")),
            "统计天数": str(user.get("days", "")),
        },
    })

    app_ids = set()
    for a in appliances:
        app_name = str(a.get("name", "未知电器"))
        app_id = f"APP::{app_name}"
        app_ids.add(app_id)
        nodes_list.append({
            "id": app_id,
            "label": app_name,
            "color": {"background": "#e8824a", "border": "#d4703a",
                      "highlight": {"background": "#f09060", "border": "#e8824a"}},
            "shape": "circle",
            "size": 26,
            "font": {"color": "#ffffff", "size": 12, "face": "Microsoft YaHei, sans-serif"},
            "props": {
                "类型": "电器",
                "名称": app_name,
                "工作日开启时段": str(a.get("weekday_periods", "")),
                "周末开启时段": str(a.get("weekend_periods", "")),
                "额定功率": str(a.get("rated_power", "")),
            },
        })
        edges_list.append({
            "from": user_id, "to": app_id, "label": "",
            "color": {"color": "#8e9aaf", "highlight": "#b0bac9"},
            "width": 1.2, "arrows": "to",
            "font": {"color": "#9ca3af", "size": 11, "strokeWidth": 0},
            "title": f"{user.get('name', '')} 拥有 {app_name}",
            "props": {
                "类型": "关系",
                "关系": "拥有",
                "起点": str(user.get("name", "")),
                "终点": app_name,
            },
        })

    for r in overlaps:
        s = f"APP::{r.get('source')}"
        t = f"APP::{r.get('target')}"
        if s not in app_ids or t not in app_ids:
            continue
        wd = r.get("weekday_overlap") or ""
        we = r.get("weekend_overlap") or ""
        edges_list.append({
            "from": s, "to": t, "label": "",
            "color": {"color": "#7a8599", "highlight": "#a0aab8"},
            "width": 1.4, "arrows": "to",
            "font": {"color": "#9ca3af", "size": 10, "strokeWidth": 0},
            "title": f"工作日重叠：{wd}\n周末重叠：{we}",
            "props": {
                "类型": "关系",
                "关系": "同时开启",
                "起点": str(r.get("source", "")),
                "终点": str(r.get("target", "")),
                "工作日重叠": str(wd),
                "周末重叠": str(we),
            },
        })

    # 为每条边分配稳定 id，便于点击后展示边属性
    for i, e in enumerate(edges_list):
        e["id"] = f"EDGE::{i}"

    nodes_json = _json.dumps(nodes_list, ensure_ascii=False)
    edges_json = _json.dumps(edges_list, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.6/dist/vis-network.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#07142a; overflow:hidden; font-family:'Microsoft YaHei',sans-serif; }}
#graph {{ width:100%; height:{height}px; }}
/* 属性面板 - Neo4j 风格 */
#props-panel {{
    display:none; position:absolute; top:10px; right:10px;
    width:260px; max-height:{height - 30}px; overflow-y:auto;
    background:rgba(35,37,45,0.95); border:1px solid rgba(160,170,185,0.3);
    border-radius:8px; color:#e8f0ff; font-size:13px;
    box-shadow:0 4px 20px rgba(0,0,0,0.5);
}}
#props-panel .panel-header {{
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 14px; border-bottom:1px solid rgba(160,170,185,0.2);
    font-weight:bold; font-size:14px;
}}
#props-panel .panel-header .close-btn {{
    cursor:pointer; color:#9cb1d9; font-size:18px; line-height:1;
}}
#props-panel .panel-header .close-btn:hover {{ color:#fff; }}
#props-panel .node-badge {{
    display:inline-block; padding:2px 10px; border-radius:10px;
    font-size:11px; font-weight:bold; margin-right:8px;
}}
#props-panel .badge-user {{ background:#c4a4d6; color:#1a1a2e; }}
#props-panel .badge-device {{ background:#e8824a; color:#1a1a2e; }}
#props-panel .badge-edge {{ background:#8ea3c6; color:#1a1a2e; }}
#props-panel table {{ width:100%; border-collapse:collapse; }}
#props-panel table tr {{ border-bottom:1px solid rgba(160,170,185,0.1); }}
#props-panel table td {{ padding:8px 14px; vertical-align:top; }}
#props-panel table td:first-child {{
    color:#9cb1d9; white-space:nowrap; width:40%; font-size:12px;
}}
#props-panel table td:last-child {{ color:#e8f0ff; word-break:break-all; }}
</style>
</head><body>
<div id="graph"></div>
<div id="props-panel">
    <div class="panel-header">
        <span id="panel-title">Node properties</span>
        <span class="close-btn" onclick="document.getElementById('props-panel').style.display='none'">&times;</span>
    </div>
    <div id="panel-body"></div>
</div>
<script>
var nodesData = {nodes_json};
var edgesData = {edges_json};

// 构建属性查找表
var propsMap = {{}};
nodesData.forEach(function(n) {{ propsMap[n.id] = n; }});
edgesData.forEach(function(e) {{ propsMap[e.id] = e; }});

var nodes = new vis.DataSet(nodesData);
var edges = new vis.DataSet(edgesData);

var container = document.getElementById('graph');
var data = {{ nodes: nodes, edges: edges }};
var options = {{
    interaction: {{
        hover: true,
        navigationButtons: true,
        keyboard: true,
        tooltipDelay: 200
    }},
    physics: {{
        enabled: true,
        forceAtlas2Based: {{
            gravitationalConstant: -50,
            springLength: 160,
            springConstant: 0.04,
            damping: 0.5
        }},
        minVelocity: 0.75,
        solver: 'forceAtlas2Based'
    }},
    edges: {{
        smooth: {{ type: 'continuous', roundness: 0.15 }},
        arrows: {{ to: {{ scaleFactor: 0.55 }} }},
        font: {{ size: 10, color: '#b8c7dd', strokeWidth: 0 }}
    }},
    nodes: {{
        borderWidth: 2,
        borderWidthSelected: 3,
        chosen: {{
            node: function(values, id, selected, hovering) {{
                if (selected) {{ values.shadowColor = 'rgba(255,255,255,0.3)'; values.shadow = true; values.shadowSize = 15; }}
            }}
        }}
    }}
}};

var network = new vis.Network(container, data, options);

network.on('hoverEdge', function(params) {{
    var edgeInfo = propsMap[params.edge];
    if (edgeInfo && edgeInfo.props && edgeInfo.props['关系']) {{
        edges.update({{ id: params.edge, label: edgeInfo.props['关系'] }});
    }}
}});

network.on('blurEdge', function(params) {{
    edges.update({{ id: params.edge, label: '' }});
}});

// 点击节点弹出属性面板
network.on('click', function(params) {{
    var panel = document.getElementById('props-panel');
    if (params.nodes.length > 0) {{
        var nodeId = params.nodes[0];
        var nodeInfo = propsMap[nodeId];
        if (!nodeInfo || !nodeInfo.props) {{ panel.style.display='none'; return; }}

        var isUser = nodeId.startsWith('USER::');
        var badgeClass = isUser ? 'badge-user' : 'badge-device';
        var badgeText = isUser ? '用户' : '电器';

        var title = document.getElementById('panel-title');
        title.innerHTML = '<span class="node-badge ' + badgeClass + '">' + badgeText + '</span>节点属性';

        var tbody = '<table>';
        var props = nodeInfo.props;
        for (var key in props) {{
            if (props[key] && props[key] !== 'None' && props[key] !== '') {{
                tbody += '<tr><td>' + key + '</td><td>' + props[key] + '</td></tr>';
            }}
        }}
        tbody += '</table>';
        document.getElementById('panel-body').innerHTML = tbody;
        panel.style.display = 'block';
    }} else if (params.edges.length > 0) {{
        var edgeId = params.edges[0];
        var edgeInfo = propsMap[edgeId];
        if (!edgeInfo || !edgeInfo.props) {{ panel.style.display='none'; return; }}

        var title = document.getElementById('panel-title');
        title.innerHTML = '<span class="node-badge badge-edge">关系</span>边属性';

        var tbody = '<table>';
        var props = edgeInfo.props;
        for (var key in props) {{
            if (props[key] && props[key] !== 'None' && props[key] !== '') {{
                tbody += '<tr><td>' + key + '</td><td>' + props[key] + '</td></tr>';
            }}
        }}
        tbody += '</table>';
        document.getElementById('panel-body').innerHTML = tbody;
        panel.style.display = 'block';
    }} else {{
        panel.style.display = 'none';
    }}
}});
</script>
</body></html>"""

    components.html(html, height=height, scrolling=False)


def build_logo_html() -> str:
    return """
    <div class="brand-wrap">
        <div class="brand-logo">⚡</div>
        <div class="brand-meta">
            <div class="brand-title">智电先锋</div>
            <div class="brand-subtitle">智电 · 让每一度电都充满智慧</div>
        </div>
    </div>
    """


def render_page_header() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-main">
                <div class="hero-kicker">Smart Energy Insight</div>
                <div class="hero-headline">多维家庭用能分析与风险洞察平台</div>
                <div class="hero-desc">聚合日用电量趋势、设备画像、安全预警与智能问答，支持家庭用能分析的可视化展示与交互讲解。</div>
                <div class="hero-tags">
                    <span>多用户</span>
                    <span>风险预警</span>
                    <span>设备画像</span>
                    <span>智能助手</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_global_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #040b18;
            --bg-1: #07142a;
            --bg-2: #0a1d3f;
            --bg-3: #10264d;
            --line-1: rgba(110, 168, 255, 0.18);
            --line-2: rgba(110, 168, 255, 0.30);
            --text-1: #e8f0ff;
            --text-2: #9cb1d9;
            --accent-1: #38bdf8;
            --accent-2: #22c55e;
            --accent-3: #2f81f7;
        }

        .stApp {
            background:
                radial-gradient(900px 380px at 12% -4%, rgba(56, 189, 248, 0.10), transparent 58%),
                radial-gradient(860px 360px at 100% 0%, rgba(47, 129, 247, 0.12), transparent 54%),
                linear-gradient(180deg, #06101f 0%, #050d1c 42%, #040914 100%);
            color: var(--text-1);
        }

        header[data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0) !important;
            height: 0 !important;
        }

        div[data-testid="stToolbar"] {
            top: 0.5rem;
            right: 0.75rem;
        }

        .block-container {
            padding-top: 0.55rem !important;
            padding-bottom: 1.1rem !important;
            max-width: 97% !important;
        }

        hr {
            border: none !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent, rgba(110,168,255,.28), transparent) !important;
            margin: 0.55rem 0 0.9rem 0 !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, label, div {
            color: var(--text-1);
        }

        .stCaption {
            color: var(--text-2) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--line-1) !important;
            border-radius: 20px !important;
            background: linear-gradient(180deg, rgba(8, 20, 42, 0.94), rgba(5, 13, 28, 0.98)) !important;
            box-shadow: 0 14px 36px rgba(1, 6, 18, 0.38) !important;
            overflow: hidden !important;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"] input,
        div[data-testid="stDateInputField"] {
            background: rgba(10, 18, 36, 0.92) !important;
            border: 1px solid var(--line-1) !important;
            color: var(--text-1) !important;
            border-radius: 14px !important;
        }

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {
            border-color: var(--line-2) !important;
            box-shadow: 0 0 0 1px rgba(56,189,248,0.18) !important;
        }

        [data-testid="stPopover"] button,
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            border-radius: 12px !important;
            border: 1px solid var(--line-1) !important;
            background: linear-gradient(180deg, rgba(16, 38, 77, 0.98), rgba(9, 25, 52, 0.98)) !important;
            color: var(--text-1) !important;
            box-shadow: none !important;
        }

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover,
        [data-testid="stPopover"] button:hover {
            border-color: var(--line-2) !important;
            transform: translateY(-1px);
        }

        div[data-testid="stChatMessage"] {
            background: linear-gradient(180deg, rgba(10, 18, 34, 0.96), rgba(7, 13, 25, 0.96));
            border: 1px solid rgba(122, 162, 247, 0.14);
            border-radius: 16px;
            padding: 10px 12px;
            margin-bottom: 10px;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-1);
            line-height: 1.72;
            font-size: 0.95rem;
        }

        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 14px;
            min-height: 56px;
        }

        .brand-logo {
            width: 52px;
            height: 52px;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 26px;
            font-weight: 800;
            box-shadow: 0 10px 28px rgba(34, 197, 94, 0.26);
            flex: 0 0 52px;
        }

        .brand-title {
            font-size: 30px;
            font-weight: 800;
            line-height: 1.05;
            letter-spacing: 0.3px;
        }

        .brand-subtitle {
            margin-top: 4px;
            color: var(--text-2);
            font-size: 12px;
            letter-spacing: 0.18px;
        }

        .hero-shell {
            display: flex;
            align-items: stretch;
            min-height: 102px;
            padding: 6px 0 2px 0;
        }

        .hero-main {
            width: 100%;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.10);
            border: 1px solid rgba(56, 189, 248, 0.16);
            color: #8ed8ff;
            font-size: 12px;
            margin-bottom: 8px;
        }

        .hero-headline {
            font-size: 24px;
            font-weight: 800;
            line-height: 1.2;
            letter-spacing: 0.2px;
            margin-bottom: 8px;
        }

        .hero-desc {
            color: var(--text-2);
            font-size: 13px;
            line-height: 1.72;
            max-width: 920px;
        }

        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }

        .hero-tags span {
            display: inline-flex;
            align-items: center;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 12px;
            color: #dcecff;
            background: rgba(16, 38, 77, 0.72);
            border: 1px solid rgba(110,168,255,.15);
        }

        .header-title-shell {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 74px;
            width: 100%;
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(110, 168, 255, 0.24);
            background:
                radial-gradient(500px 160px at 14% 0%, rgba(56, 189, 248, 0.18), transparent 62%),
                radial-gradient(560px 180px at 88% 100%, rgba(47, 129, 247, 0.18), transparent 62%),
                linear-gradient(135deg, rgba(8, 24, 50, 0.92), rgba(7, 20, 42, 0.95));
            box-shadow: 0 10px 28px rgba(1, 6, 18, 0.32), inset 0 1px 0 rgba(255,255,255,0.04);
        }

        .header-title-shell::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(148, 201, 255, 0.09), transparent);
            pointer-events: none;
        }

        .header-title-text {
            position: relative;
            text-align: center;
            font-size: 40px;
            font-weight: 900;
            line-height: 1.1;
            letter-spacing: 0.3px;
            padding: 10px 18px;
            background: linear-gradient(90deg, #ecf4ff 0%, #94c9ff 50%, #ecf4ff 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 0 30px rgba(111, 193, 255, 0.24);
        }

        @media (max-width: 1200px) {
            .header-title-text {
                font-size: 34px;
            }
        }

        @media (max-width: 900px) {
            .header-title-text {
                font-size: 28px;
            }
        }

        .header-setting-pad {
            height: 8px;
        }

        .chat-toolbar-note {
            color: var(--text-2);
            font-size: 12px;
            line-height: 1.65;
            margin-top: 2px;
            margin-bottom: 10px;
        }

        .chat-scroll-wrap {
            border: none;
            border-radius: 0;
            background: transparent;
            padding: 2px 0 0 0;
            margin-bottom: 8px !important;
        }

        .quick-question-wrap {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin: 2px 2px 6px;
        }
        .quick-question-label {
            color: var(--text-2);
            font-size: 12px;
            white-space: nowrap;
        }
        .date-context-chip {
            display: inline-flex;
            align-items: center;
            height: 26px;
            padding: 0 10px;
            border-radius: 999px;
            border: 1px solid rgba(110, 168, 255, 0.18);
            background: rgba(13, 29, 57, 0.76);
            color: #a9c8f6;
            font-size: 12px;
            line-height: 1;
            margin-top: -4px;
            margin-bottom: 6px;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="column"] .stButton button {
            white-space: nowrap;
            min-height: 34px !important;
            height: 34px !important;
            padding: 0 10px !important;
            border-radius: 12px !important;
            background: linear-gradient(180deg, rgba(20, 49, 94, 0.95), rgba(11, 31, 65, 0.98)) !important;
            border-color: rgba(110, 168, 255, 0.20) !important;
            color: #dcecff !important;
            font-size: 12px !important;
            font-weight: 700 !important;
        }

        div[data-testid="stChatInput"] {
            margin-top: 10px !important;
            border: none !important;
            background: transparent !important;
        }
        div[data-testid="stChatInput"] > div {
            background: linear-gradient(180deg, rgba(14, 22, 38, 0.92), rgba(8, 15, 29, 0.96)) !important;
            border: 1px solid rgba(110, 168, 255, 0.16) !important;
            border-radius: 18px !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 14px 34px rgba(1, 6, 18, 0.24) !important;
        }
        div[data-testid="stChatInput"] textarea {
            color: var(--text-1) !important;
            font-size: 14px !important;
            line-height: 1.55 !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #8394b8 !important;
        }
        div[data-testid="stChatInput"] button {
            border-radius: 12px !important;
            background: linear-gradient(135deg, var(--accent-3), var(--accent-2)) !important;
            color: #ffffff !important;
            box-shadow: 0 10px 24px rgba(47, 129, 247, 0.30) !important;
        }

        /* 用户资料卡片 */
        .user-profile-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 16px 8px 10px;
            gap: 6px;
        }
        .user-avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent-3), var(--accent-1));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: 800;
            color: #fff;
            box-shadow: 0 6px 20px rgba(47, 129, 247, 0.30);
        }
        .user-name {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-1);
        }
        .user-badge {
            font-size: 11px;
            padding: 2px 10px;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.10);
            border: 1px solid rgba(56, 189, 248, 0.18);
            color: #8ed8ff;
        }
        .user-info-rows {
            display: flex;
            flex-direction: column;
            gap: 0;
            padding: 4px 12px 8px;
        }
        .user-info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 7px 0;
            border-bottom: 1px solid rgba(110, 168, 255, 0.08);
        }
        .user-info-row:last-child {
            border-bottom: none;
        }
        .user-info-label {
            font-size: 12px;
            color: var(--text-2);
            flex-shrink: 0;
        }
        .user-info-value {
            font-size: 12px;
            color: var(--text-1);
            text-align: right;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 65%;
        }

        .user-password-wrap {
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .pwd-toggle-checkbox {
            display: none;
        }
        .user-password-mask,
        .user-password-plain {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 72px;
            height: 30px;
            padding: 0 12px;
            border-radius: 10px;
            border: 1px solid rgba(110, 168, 255, 0.24);
            background: rgba(10, 18, 36, 0.92);
            color: var(--text-1);
            font-size: 12px;
            line-height: 1;
        }
        .user-password-plain {
            display: none;
        }
        .pwd-toggle-checkbox:checked ~ .user-password-mask {
            display: none;
        }
        .pwd-toggle-checkbox:checked ~ .user-password-plain {
            display: inline-flex;
        }
        .user-password-eye {
            width: 34px;
            height: 30px;
            border-radius: 10px;
            border: 1px solid rgba(110, 168, 255, 0.24);
            background: rgba(10, 18, 36, 0.92);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            user-select: none;
            font-size: 14px;
        }
        .user-status-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 64px;
            height: 28px;
            padding: 0 10px;
            border-radius: 999px;
            border: 1px solid rgba(34, 197, 94, 0.28);
            background: rgba(22, 163, 74, 0.12);
            color: #8ff0b0;
            font-size: 12px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def section_title(title: str) -> None:
    icon_map = {
        "日用电量趋势": "📈",
        "用能总览": "📊",
        "安全风险洞察": "🛡️",
        "能耗风险洞察": "⚠️",
        "家庭用能知识图谱": "🧠",
        "单日设备画像": "🔍",
        "智能用电助手": "🤖",
    }
    icon = icon_map.get(title, "✨")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;font-size:22px;font-weight:800;letter-spacing:0.25px;line-height:1.2;margin:2px 0 10px 0;">
            <span style="font-size:18px;opacity:.95;">{icon}</span>
            <span style="background:linear-gradient(90deg,#eef5ff,#8cc8ff);-webkit-background-clip:text;background-clip:text;color:transparent;">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def meter_card(title: str, value: float, unit: str, color: str, compact: bool = False) -> None:
    pad = "14px 14px" if compact else "20px 22px"
    radius = "16px" if compact else "20px"
    title_size = "12px" if compact else "14px"
    value_size = "29px" if compact else "40px"
    unit_size = "12px" if compact else "14px"
    title_gap = "6px" if compact else "10px"
    unit_gap = "4px" if compact else "8px"
    st.markdown(
        f"""
        <div style="background:{color};padding:{pad};border-radius:{radius};color:white;box-shadow:0 10px 26px rgba(15,23,42,0.12);">
            <div style="font-size:{title_size};opacity:.9;margin-bottom:{title_gap};">{title}</div>
            <div style="font-size:{value_size};font-weight:800;line-height:1;">{value:,.2f}</div>
            <div style="font-size:{unit_size};margin-top:{unit_gap};opacity:.92;">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_figure(title: str, text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#dbeafe"))
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_dark",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,23,49,0.72)",
        font=dict(color="#dbeafe"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


CHART_PALETTE = [
    "#35c7ff", "#6f8bff", "#22d3ee", "#2dd4bf", "#f59e0b", "#fb7185", "#a78bfa", "#84cc16", "#38bdf8", "#60a5fa"
]

# 更丰富、偏柔和的柱状图配色（参考竞赛大屏风格，避免全浅蓝导致单调）
RICH_BAR_PALETTE = [
    "#34C6FF",  # 天空蓝
    "#5DA9FF",  # 亮蓝
    "#7BC8FF",  # 冰蓝
    "#1ED4C8",  # 青绿
    "#66D3B0",  # 薄荷绿
    "#E9C35B",  # 柔黄
    "#E7A25D",  # 杏橙
    "#F08BA3",  # 粉玫
    "#AE8EEB",  # 淡紫
    "#7BC6E8",  # 雾蓝
    "#9ED47A",  # 黄绿
]


def apply_chart_theme(fig: go.Figure, height: int = 320, b_margin: int = 24) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=60, b=b_margin),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,23,49,0.72)",
        font=dict(color="#dbeafe", size=12),
        title=dict(font=dict(size=18, color="#eaf2ff"), x=0.02, xanchor="left"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(7,16,34,0.45)",
            bordercolor="rgba(110,168,255,.24)",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor="rgba(7,16,34,0.92)", bordercolor="rgba(110,168,255,.35)", font=dict(color="#e6f0ff")),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(110,168,255,.12)",
        zeroline=False,
        tickfont=dict(color="#c8dbf8"),
        linecolor="rgba(110,168,255,.22)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(110,168,255,.12)",
        zeroline=False,
        tickfont=dict(color="#c8dbf8"),
        linecolor="rgba(110,168,255,.22)",
    )
    return fig

def build_daily_energy_bar(summary_df: pd.DataFrame) -> go.Figure:
    if summary_df.empty:
        return empty_figure("日用电量趋势", "当前范围暂无数据")

    trend_df = summary_df.copy()
    date_series = pd.to_datetime(trend_df["date"], errors="coerce") if "date" in trend_df.columns else pd.to_datetime(trend_df["date_str"], errors="coerce")
    trend_df["date_md"] = date_series.dt.strftime("%-m-%-d") if os.name != "nt" else date_series.dt.strftime("%m-%d").str.lstrip("0").str.replace("-0", "-", regex=False)
    trend_df["date_md"] = trend_df["date_md"].fillna("未知")

    fig = px.line(
        trend_df,
        x="date_md",
        y="total_kwh",
        markers=True,
        labels={"date_md": "日期（月-日）", "total_kwh": "kWh"},
        title="日用电量趋势",
    )
    fig.update_traces(
        mode="lines+markers",
        line=dict(width=3.4, color="#38bdf8", shape="spline", smoothing=0.55),
        marker=dict(size=7, color="#b8e0ff", line=dict(width=1.2, color="#e6f7ff")),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.10)",
        hovertemplate="日期：%{x}<br>用电量：%{y:.2f} kWh<extra></extra>",
    )
    apply_chart_theme(fig, height=248, b_margin=16)

    day_count = int(trend_df["date_md"].nunique())
    tick_angle = 0
    if day_count > 14:
        tick_angle = -35
    if day_count > 24:
        tick_angle = -90

    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=trend_df["date_md"].tolist(),
        tickangle=tick_angle,
        automargin=True,
    )
    fig.update_yaxes(title_text="kWh", title_standoff=2, automargin=True)
    fig.update_layout(margin=dict(l=14, r=8, t=52, b=16))
    return fig

def build_hour_event_figure(hour_df: pd.DataFrame) -> go.Figure:
    if hour_df.empty:
        return empty_figure("小时事件分布", "没有可展示的小时事件数据")

    bar_df = hour_df.copy()
    bar_df["hour_label"] = pd.to_numeric(bar_df["hour"], errors="coerce").fillna(0).astype(int).astype(str) + "时"
    fig = px.bar(
        bar_df,
        x="hour_label",
        y="event_count",
        text_auto=True,
        labels={"hour_label": "小时", "event_count": "事件数"},
        title="小时事件分布",
    )
    cartoon_colors = RICH_BAR_PALETTE
    fig.update_traces(
        marker=dict(
            color=[cartoon_colors[i % len(cartoon_colors)] for i in range(len(bar_df))],
            line=dict(color="rgba(232,245,255,.78)", width=1.25),
        ),
        textfont=dict(color="#eaf2ff", size=11),
        hovertemplate="小时：%{x}<br>事件数：%{y}<extra></extra>",
        opacity=0.96,
    )
    apply_chart_theme(fig, height=320, b_margin=20)
    fig.update_layout(barcornerradius=10, bargap=0.22)
    fig.update_xaxes(type="category", tickangle=0)
    return fig

def build_device_pie_figure(pie_df: pd.DataFrame) -> go.Figure:
    if pie_df.empty:
        return empty_figure("设备能耗构成", "没有设备能耗占比数据")
    fig = px.pie(
        pie_df,
        names="device_or_group",
        values="energy_kwh",
        hole=0.42,
        title="设备能耗构成",
        color_discrete_sequence=CHART_PALETTE,
    )
    max_pos = int(pd.to_numeric(pie_df["energy_kwh"], errors="coerce").fillna(0).values.argmax()) if len(pie_df) else 0
    fig.update_traces(
        domain=dict(x=[0.02, 0.72], y=[0.02, 0.98]),
        textposition="outside",
        textinfo="percent",
        textfont=dict(size=11, color="#dbeafe"),
        marker=dict(line=dict(color="rgba(230,245,255,.45)", width=1.2)),
        pull=[0.05 if i == max_pos else 0 for i in range(len(pie_df))],
        hovertemplate="设备：%{label}<br>电量：%{value:.2f} kWh<br>占比：%{percent}<extra></extra>",
    )
    apply_chart_theme(fig, height=320, b_margin=20)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            x=0.84,
            y=0.50,
            xanchor="left",
            yanchor="middle",
            font=dict(size=14),
            bgcolor="rgba(7,16,34,0.38)",
            bordercolor="rgba(110,168,255,.22)",
            borderwidth=1,
        ),
        margin=dict(l=10, r=34, t=60, b=20),
    )
    return fig

def build_device_energy_bar_figure(bar_df: pd.DataFrame) -> go.Figure:
    if bar_df.empty:
        return empty_figure("设备能耗对比", "没有设备级能耗数据")
    fig = px.bar(
        bar_df,
        x="device_or_group",
        y="energy_kwh",
        text_auto=".2f",
        labels={"device_or_group": "设备", "energy_kwh": "kWh"},
        title="设备能耗对比",
    )
    light_bar_colors = RICH_BAR_PALETTE
    fig.update_traces(
        marker=dict(
            color=[light_bar_colors[i % len(light_bar_colors)] for i in range(len(bar_df))],
            line=dict(color="rgba(230,244,255,.75)", width=1.2),
        ),
        textfont=dict(color="#eaf2ff", size=11),
        hovertemplate="设备：%{x}<br>电量：%{y:.2f} kWh<extra></extra>",
        opacity=0.97,
    )
    apply_chart_theme(fig, height=320, b_margin=56)
    fig.update_layout(barcornerradius=12, bargap=0.28)
    device_count = int(bar_df["device_or_group"].nunique()) if "device_or_group" in bar_df.columns else len(bar_df)
    fig.update_xaxes(tickangle=-22 if device_count > 6 else 0)
    fig.update_yaxes(title_text="kWh")
    return fig

def build_power_curve_figure(curve_df: pd.DataFrame, total_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    color_idx = 0
    if not curve_df.empty:
        for col in curve_df.columns:
            if col == "minute_of_day":
                continue
            fig.add_trace(
                go.Scatter(
                    x=curve_df["minute_of_day"] / 60.0,
                    y=curve_df[col],
                    mode="lines",
                    name=str(col),
                    line=dict(width=2.2, color=CHART_PALETTE[color_idx % len(CHART_PALETTE)], shape="spline", smoothing=0.45),
                    hovertemplate="时间：%{x:.2f} 时<br>%{fullData.name}：%{y:.2f}<extra></extra>",
                )
            )
            color_idx += 1

    if not total_df.empty and "T" in total_df.columns and "总功率" in total_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(total_df["T"], errors="coerce") / 60.0,
                y=pd.to_numeric(total_df["总功率"], errors="coerce"),
                mode="lines",
                name="总功率",
                line=dict(width=3.2, dash="dash", color="#f59e0b"),
                hovertemplate="时间：%{x:.2f} 时<br>总功率：%{y:.2f}<extra></extra>",
            )
        )

    if len(fig.data) == 0:
        return empty_figure("设备功率曲线", "没有功率曲线数据")

    fig.update_layout(
        title="设备功率曲线",
        xaxis_title="时间（小时）",
        yaxis_title="功率",
        legend_title="设备",
        hovermode="x unified",
    )
    apply_chart_theme(fig, height=320, b_margin=22)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.14,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
            bgcolor="rgba(7,16,34,0.40)",
            bordercolor="rgba(110,168,255,.20)",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=94, b=22),
        legend_title_text="",
    )
    tick_hours = list(range(0, 25, 2))
    fig.update_xaxes(range=[0, 24], tickvals=tick_hours, ticktext=[f"{h:02d}:00" for h in tick_hours])
    return fig

def read_single_day_visual_data(file_path: str) -> Dict[str, pd.DataFrame]:
    sheets = load_daily_excel(file_path)
    expected = {
        "hour_event_ratio": pd.DataFrame(columns=["hour", "event_count", "event_ratio"]),
        "energy_pie_data": pd.DataFrame(columns=["device_or_group", "energy_kwh", "ratio"]),
        "energy_bar_data": pd.DataFrame(columns=["device_or_group", "energy_kwh"]),
        "heating_power_curve": pd.DataFrame(columns=["minute_of_day"]),
        "total_power_curve": pd.DataFrame(columns=["datatime", "总功率", "T"]),
        "event_summary": pd.DataFrame(),
        "result_M": pd.DataFrame(),
    }
    for key, default_df in expected.items():
        if key not in sheets:
            sheets[key] = default_df
    return sheets


def do_logout() -> None:
    st.session_state.logged_in = False
    st.session_state.selected_day = None
    st.session_state.autologin_applied = True
    try:
        st.query_params.clear()
    except Exception:
        pass


def try_autologin_from_query() -> None:
    if bool(st.session_state.get("autologin_applied", False)):
        return

    params = st.query_params
    autologin = str(params.get("autologin", "0")).strip().lower() in {"1", "true", "yes"}
    if not autologin:
        return

    dataset = str(params.get("dataset", "")).strip()
    house = str(params.get("house", "")).strip()
    if not dataset or not house:
        return

    if not st.session_state.configured_root:
        guessed_root = ""
        for p in DEFAULT_ROOT_CANDIDATES:
            if Path(p).exists():
                guessed_root = p
                break
        if guessed_root:
            st.session_state.configured_root = guessed_root

    datasets = scan_datasets(st.session_state.configured_root) if st.session_state.configured_root else {}
    if dataset in datasets:
        house_keys = {h["house_key"] for h in datasets[dataset]}
        if house in house_keys:
            st.session_state.dataset_name = dataset
            st.session_state.selected_house = house
            st.session_state.logged_in = True
            st.session_state.autologin_applied = True


def sidebar_settings() -> Tuple[Dict[str, List[Dict[str, str]]], Optional[str], Optional[str]]:
    with st.popover("☰ 设置", use_container_width=False):
        st.markdown("### 平台设置")
        root_dir = st.text_input(
            "按日分析结果根目录",
            value=st.session_state.configured_root,
            placeholder=r"例如：F:/研究生文件/节能减排/云端功率分析代码/output/按日分析结果_全部",
        )
        kg_root_dir = st.text_input(
            "知识图谱文件目录（本地导出）",
            value=st.session_state.kg_data_root,
            placeholder=r"例如：F:/研究生文件/节能减排/kg_export",
        )
        tier_options = [1, 2, 3]
        current_tier = int(st.session_state.get("tier_level", 1))
        if current_tier not in tier_options:
            current_tier = 1
        tier_level = st.selectbox("电价阶梯档位", tier_options, index=tier_options.index(current_tier))
        enable_chat_api = st.toggle(
            "启用智能助手 API 回答（非问答对问题）",
            value=bool(st.session_state.get("enable_chat_api", False)),
            help="开启后：命中问答对仍走本地；未命中时调用后端 API。",
        )
        st.caption("智能助手会根据当前登录用户自动匹配用电记录。")

        # 开关即时生效，不依赖“保存路径”按钮
        st.session_state.enable_chat_api = bool(enable_chat_api)

        if st.button("保存路径", use_container_width=True):
            normalized = root_dir.strip().strip('"').strip("'")
            kg_normalized = kg_root_dir.strip().strip('"').strip("'")
            st.session_state.configured_root = normalized
            st.session_state.kg_data_root = kg_normalized
            st.session_state.tier_level = int(tier_level)
            st.session_state.enable_chat_api = bool(enable_chat_api)
            st.cache_data.clear()
            st.success("数据路径已保存")
            st.rerun()

        datasets = scan_datasets(st.session_state.configured_root) if st.session_state.configured_root else {}
        if not datasets:
            st.info("可设置三种路径：总根目录 / 单个分组目录 / 单个用户目录。")
            return {}, None, None

        dataset_names = list(datasets.keys())
        dataset_index = dataset_names.index(st.session_state.dataset_name) if st.session_state.dataset_name in dataset_names else 0
        dataset_name = st.selectbox("选择分组", dataset_names, index=dataset_index)
        st.session_state.dataset_name = dataset_name

        houses = datasets[dataset_name]
        house_labels = [f"{h['display_name']}（{h['house_key']}）" for h in houses]
        house_keys = [h["house_key"] for h in houses]
        selected_house_idx = house_keys.index(st.session_state.selected_house) if st.session_state.selected_house in house_keys else 0
        chosen_label = st.selectbox("可选用户", house_labels, index=selected_house_idx)
        chosen_house = houses[house_labels.index(chosen_label)]["house_key"]
        st.session_state.selected_house = chosen_house
        return datasets, dataset_name, chosen_house


def render_login_panel(dataset_name: str, house_key: str, datasets: Dict[str, List[Dict[str, str]]]) -> HouseInfo:
    house_info = get_house_info(dataset_name, house_key)
    st.markdown(f"""
    <div class="user-profile-card">
        <div class="user-avatar">{''.join([c for c in house_info.display_name if c])[0] if house_info.display_name else 'U'}</div>
        <div class="user-name">{house_info.display_name}</div>
        <div class="user-badge">在线</div>
    </div>
    <div class="user-info-rows">
        <div class="user-info-row">
            <span class="user-info-label">住址</span>
            <span class="user-info-value">{house_info.address}</span>
        </div>
        <div class="user-info-row">
            <span class="user-info-label">账号</span>
            <span class="user-info-value">{house_info.account}</span>
        </div>
        <div class="user-info-row">
            <span class="user-info-label">账户状态</span>
            <span class="user-info-value"><span class="user-status-pill">已认证</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return house_info


def render_total_card(summary_df: pd.DataFrame, compact: bool = False) -> None:
    total_kwh = float(summary_df["total_kwh"].sum()) if not summary_df.empty else 0.0
    total_cost = float(summary_df["cost"].sum()) if (not summary_df.empty and "cost" in summary_df.columns) else 0.0
    potential_save_kwh = total_kwh * 0.25827
    potential_save_cost = total_cost * 0.3996

    c1, c2 = st.columns(2)
    with c1:
        meter_card("总用电量", total_kwh, "kWh", "linear-gradient(135deg,#2563eb,#38bdf8)", compact=compact)
    with c2:
        meter_card("总电费", total_cost, "元（分时+阶梯）", "linear-gradient(135deg,#16a34a,#84cc16)", compact=compact)

    row_gap = "4px" if compact else "6px"
    st.markdown(f"<div style='height:{row_gap}'></div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        meter_card("潜在节能电量", potential_save_kwh, "kWh", "linear-gradient(135deg,#0ea5e9,#06b6d4)", compact=compact)
    with c4:
        meter_card("潜在节能费用", potential_save_cost, "元", "linear-gradient(135deg,#22c55e,#14b8a6)", compact=compact)


def render_alert_panel(alert_df: pd.DataFrame) -> None:
    section_title("能耗风险洞察")
    if alert_df.empty:
        st.info("当前没有能耗风险提示数据。")
        return
    for _, row in alert_df.iterrows():
        level_color = "#ef4444" if row["level"] == "高" else "#f59e0b"
        st.markdown(
            f"""
            <div style="border-left:4px solid {level_color};background:linear-gradient(180deg,rgba(16,37,73,.82),rgba(9,22,45,.90));padding:11px 12px;border-radius:12px;margin-bottom:9px;border:1px solid rgba(110,168,255,.18);">
                <div style="font-size:12px;color:#95b4df;">{row['date']} {row['time']} · {row['level']}风险</div>
                <div style="font-size:16px;font-weight:700;margin:3px 0;color:#e6f0ff;">{row['device']}</div>
                <div style="font-size:13px;color:#c2d5f5;line-height:1.55;">{row['message']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_chat_panel(house_key: str, selected_date, max_available_date) -> None:
    title_l, title_r = st.columns([3.1, 1.45])
    with title_l:
        section_title("智能用电助手")
    with title_r:
        with st.popover("导出报告", use_container_width=True):
            report_candidates = [
                Path(__file__).resolve().parents[1] / "data" / "个性化节能分析报告.pdf",
                Path.cwd() / "data" / "个性化节能分析报告.pdf",
            ]
            report_file = next((p for p in report_candidates if p.exists()), None)

            if report_file is None:
                st.warning("未找到报告文件：data/个性化节能分析报告.pdf")
            else:
                report_content = report_file.read_bytes()
                st.caption(f"当前导出源文件：{report_file.name}")
                default_save_path = r"C:/个性化节能分析报告.pdf"
                save_path_text = st.text_input(
                    "建议分析报告保存路径",
                    value=default_save_path,
                    placeholder=r"例如：C:/个性化节能分析报告.pdf",
                    key="chat_report_save_path",
                )

                if st.button("保存到指定路径", use_container_width=True):
                    try:
                        save_path = Path((save_path_text or "").strip().strip('"').strip("'"))
                        if not str(save_path):
                            st.warning("请先输入保存路径。")
                        else:
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                            save_path.write_bytes(report_content)
                            st.success(f"报告已保存到：{save_path}")
                    except Exception as e:
                        st.error(f"保存失败：{e}")

                st.download_button(
                    "下载建议分析报告",
                    data=report_content,
                    file_name="个性化节能分析报告.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    st.markdown(
        """
        <div class="chat-scroll-wrap">
        """,
        unsafe_allow_html=True,
    )
    chat_log_container = st.container(height=CHAT_MESSAGES_HEIGHT)
    with chat_log_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    st.markdown("</div>", unsafe_allow_html=True)

    quick_prompt = ""
    st.markdown(
        '<div class="quick-question-wrap"><span class="quick-question-label">快捷提问</span></div>',
        unsafe_allow_html=True,
    )
    quick_questions = [
        ("今日用电情况", "今天用电情况怎么样？"),
        ("耗电设备分析", "最近几天哪些设备最耗电？"),
        ("能耗风险分析", "有没有异常运行或高风险时段？"),
        ("用电节能分析", "请给我几条节能建议。"),
    ]
    quick_cols = st.columns(2)
    for idx, (label, question) in enumerate(quick_questions):
        with quick_cols[idx % 2]:
            if st.button(label, key=f"quick_question_{idx}", use_container_width=True):
                quick_prompt = question

    user_prompt = st.chat_input(
        "请输入您的问题，例如：最近用电情况怎么样？",
        key="assistant_chat_input",
    )

    prompt_to_send = quick_prompt or (user_prompt.strip() if user_prompt else "")

    if prompt_to_send:
        prompt = prompt_to_send.strip()
        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            # 将“本轮新增消息”也渲染在聊天记录容器内，避免流式输出把输入框顶上去
            with chat_log_container:
                with st.chat_message("user"):
                    st.write(prompt)

            answer = match_answer_from_qa(prompt)
            if answer:
                with chat_log_container:
                    with st.chat_message("assistant"):
                        thinking_holder = st.empty()
                        thinking_holder.markdown("<span style='color:#9cb1d9;'>思考中...</span>", unsafe_allow_html=True)
                        time.sleep(5)
                        thinking_holder.empty()
            else:
                if bool(st.session_state.get("enable_chat_api", False)):
                    session_id = f"web-{house_key}"
                    with chat_log_container:
                        with st.chat_message("assistant"):
                            thinking_holder = st.empty()
                            thinking_holder.markdown("<span style='color:#9cb1d9;'>思考中...</span>", unsafe_allow_html=True)
                            with st.spinner("正在生成回复..."):
                                answer = call_energy_chat_api(
                                    user_query=prompt,
                                    house_key=house_key,
                                    selected_date=selected_date,
                                    max_available_date=max_available_date,
                                    session_id=session_id,
                                )
                            thinking_holder.empty()
                else:
                    answer = "超出问答范围"

            answer = beautify_assistant_text(answer)

            with chat_log_container:
                with st.chat_message("assistant"):
                    streamed = st.write_stream(stream_text_chunks(answer))

            st.session_state.chat_messages.append({"role": "assistant", "content": str(streamed) if streamed is not None else answer})
            st.rerun()

def login_view(datasets: Dict[str, List[Dict[str, str]]], dataset_name: Optional[str], house_key: Optional[str]) -> None:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.6, 1])
    with center:
        st.markdown("### 登录到智电先锋")

        has_data = bool(datasets and dataset_name)
        houses: List[Dict[str, str]] = []
        if has_data:
            houses = datasets[dataset_name]

        account = st.text_input("账号", placeholder="请输入账号，例如：用户1", disabled=not has_data)
        password = st.text_input("密码", type="password", disabled=not has_data)
        if st.button("登录", type="primary", use_container_width=True, disabled=not has_data):
            target_house = resolve_house_by_account(account, houses)
            if not target_house:
                st.error("账号不存在，请检查后重试")
            elif password != DEFAULT_PASSWORD:
                st.error("密码错误，请重试")
            else:
                st.session_state.logged_in = True
                st.session_state.selected_house = target_house
                st.rerun()


def main() -> None:
    ensure_session_defaults()
    try_autologin_from_query()
    apply_global_theme()

    # 顶部一行：品牌 + 平台标题 + 设置
    head_left, head_mid, head_right = st.columns([2.1, 4.7, 1.2], gap="large")
    with head_left:
        st.markdown(build_logo_html(), unsafe_allow_html=True)
    with head_mid:
        st.markdown(
            """
            <div class="header-title-shell">
                <div class="header-title-text">智能家庭用能与节能分析平台</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with head_right:
        st.markdown("<div class='header-setting-pad'></div>", unsafe_allow_html=True)
        hdr_c1, hdr_c2 = st.columns([1, 1])
        with hdr_c1:
            datasets, dataset_name, house_key = sidebar_settings()
        with hdr_c2:
            if st.session_state.logged_in:
                if st.button("↩ 退出", use_container_width=True, type="secondary"):
                    do_logout()
                    st.rerun()

    st.markdown("---")

    if not st.session_state.logged_in:
        login_view(datasets, dataset_name, house_key)
        return

    if not st.session_state.configured_root:
        st.warning("请先通过右上角设置按钮配置数据路径。")
        return

    datasets = scan_datasets(st.session_state.configured_root)
    dataset_name = st.session_state.dataset_name or (list(datasets.keys())[0] if datasets else "")
    if not dataset_name or dataset_name not in datasets:
        st.warning("当前没有可用分组，请先检查路径。")
        return

    houses = {h["house_key"]: h for h in datasets[dataset_name]}
    if not houses:
        st.warning("当前分组下没有可用用户。")
        return

    if st.session_state.selected_house not in houses:
        st.session_state.selected_house = next(iter(houses.keys()))

    house_key = st.session_state.selected_house
    house_dir = houses[house_key]["house_path"]
    house_info = get_house_info(dataset_name, house_key)

    available_dates = scan_house_dates(house_dir)
    if not available_dates:
        st.error("当前用户目录下没有按日 Excel 数据。")
        return

    min_date = min(available_dates).date()
    max_date = max(available_dates).date()

    if not st.session_state.date_range:
        st.session_state.date_range = (min_date, max_date)
    if st.session_state.selected_day is None:
        st.session_state.selected_day = max_date

    left_area, right_chat = st.columns([5.1, 1.35], gap="large")

    with left_area:
        layout_left, layout_mid, layout_right = st.columns([1.15, 1.45, 2.50], gap="large")

        with layout_left:
            with st.container(border=True, height=300):
                render_login_panel(dataset_name, house_key, datasets)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            with st.container(border=True, height=300):
                top_l, top_r = st.columns([3, 1])
                with top_l:
                    section_title("用能总览")
                    if st.session_state.date_range and len(st.session_state.date_range) == 2:
                        st.markdown(
                            f"<div class='date-context-chip'>当前范围：{_format_date_chip(st.session_state.date_range[0], st.session_state.date_range[1])}</div>",
                            unsafe_allow_html=True,
                        )
                selected_range = st.session_state.date_range
                with top_r:
                    with st.popover("📅", use_container_width=True):
                        selected_range = st.date_input(
                            "选择日期范围（总览）",
                            value=st.session_state.date_range,
                            min_value=min_date,
                            max_value=max_date,
                            format="YYYY-MM-DD",
                        )
                if isinstance(selected_range, tuple) and len(selected_range) == 2:
                    st.session_state.date_range = selected_range
                elif isinstance(selected_range, list) and len(selected_range) == 2:
                    st.session_state.date_range = tuple(selected_range)

                start_date, end_date = st.session_state.date_range
                summary_df = load_range_summary(
                    house_dir,
                    str(start_date),
                    str(end_date),
                    tier_level=int(st.session_state.get("tier_level", 1)),
                )
                render_total_card(summary_df, compact=True)

        with layout_mid:
            with st.container(border=True, height=300):
                render_alert_panel(build_alert_records(house_dir))

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            with st.container(border=True, height=300):
                title_l, title_r = st.columns([5, 1])
                with title_l:
                    section_title("日用电量趋势")
                    if st.session_state.date_range and len(st.session_state.date_range) == 2:
                        st.markdown(
                            f"<div class='date-context-chip'>趋势范围：{_format_date_chip(st.session_state.date_range[0], st.session_state.date_range[1])}</div>",
                            unsafe_allow_html=True,
                        )
                selected_range = st.session_state.date_range
                with title_r:
                    with st.popover("📅", use_container_width=True):
                        selected_range = st.date_input(
                            "选择日期范围（趋势）",
                            value=st.session_state.date_range,
                            min_value=min_date,
                            max_value=max_date,
                            format="YYYY-MM-DD",
                        )
                if isinstance(selected_range, tuple) and len(selected_range) == 2:
                    st.session_state.date_range = selected_range
                elif isinstance(selected_range, list) and len(selected_range) == 2:
                    st.session_state.date_range = tuple(selected_range)

                start_date, end_date = st.session_state.date_range
                summary_df = load_range_summary(
                    house_dir,
                    str(start_date),
                    str(end_date),
                    tier_level=int(st.session_state.get("tier_level", 1)),
                )
                fig = build_daily_energy_bar(summary_df)
                st.plotly_chart(fig, use_container_width=True)

        with layout_right:
            with st.container(border=True, height=612):
                section_title("家庭用能知识图谱")
                render_house_kg_panel(
                    user_name=house_info.display_name,
                    dataset_name=dataset_name,
                    house_key=house_key,
                    height=548,
                )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            c_title, c_picker = st.columns([3, 1])
            with c_title:
                section_title("单日设备画像")
                st.markdown(
                    f"<div class='date-context-chip'>当前单日：{_format_date_chip(st.session_state.selected_day)}</div>",
                    unsafe_allow_html=True,
                )
            with c_picker:
                with st.popover("📅", use_container_width=True):
                    chosen_day = st.date_input(
                        "选择单日",
                        value=st.session_state.selected_day,
                        min_value=min_date,
                        max_value=max_date,
                        format="YYYY-MM-DD",
                        key="single_day_picker",
                    )
                st.session_state.selected_day = chosen_day

            target_file = os.path.join(house_dir, f"{pd.to_datetime(st.session_state.selected_day).strftime('%Y-%m-%d')}.xlsx")
            if not os.path.exists(target_file):
                st.warning("该日期暂无导出的按日分析结果 Excel。")
            else:
                day_data = read_single_day_visual_data(target_file)
                fig_col1, fig_col2 = st.columns(2, gap="large")
                with fig_col1:
                    st.plotly_chart(build_hour_event_figure(day_data["hour_event_ratio"]), use_container_width=True)
                with fig_col2:
                    st.plotly_chart(build_device_pie_figure(day_data["energy_pie_data"]), use_container_width=True)

                fig_col3, fig_col4 = st.columns(2, gap="large")
                with fig_col3:
                    st.plotly_chart(build_device_energy_bar_figure(day_data["energy_bar_data"]), use_container_width=True)
                with fig_col4:
                    st.plotly_chart(
                        build_power_curve_figure(day_data["heating_power_curve"], day_data["total_power_curve"]),
                        use_container_width=True,
                    )

    with right_chat:
        with st.container(border=True, height=CHAT_PANEL_HEIGHT):
            render_chat_panel(
                house_key=house_key,
                selected_date=st.session_state.selected_day,
                max_available_date=max_date,
            )

    st.caption(
        f"当前用户：{house_info.display_name}｜ "
        f"电价模式：分时+阶梯（当前第 {int(st.session_state.get('tier_level', 1))} 档）"
    )


if __name__ == "__main__":
    main()
