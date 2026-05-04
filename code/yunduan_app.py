import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from urllib.parse import urlencode
from typing import Dict, List, Optional


st.set_page_config(
    page_title="智电先锋云端",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUPPORTED_DATASET_NAMES = ["REDD", "UK-DALE", "REFIT"]
DEFAULT_ROOT_CANDIDATES = [
    os.getenv("APP_DATA_ROOT", "").strip(),
    str((Path(__file__).resolve().parent.parent / "data" / "按日分析结果_全部")),
    str((Path(__file__).resolve().parent.parent / "data" / "按日分析结果")),
    str((Path(__file__).resolve().parent.parent / "data")),
    r"F:/研究生文件/节能减排/云端功率分析代码/output/按日分析结果_全部",
]
DEFAULT_ROOT = ""
LOCATION_MAP = {
    "REDD": "广州市天河区",
    "UK-DALE": "上海市浦东新区",
    "REFIT": "杭州市西湖区",
    "CUSTOM": "深圳市南山区",
}


def resolve_initial_root() -> str:
    for p in DEFAULT_ROOT_CANDIDATES:
        if p and Path(p).exists() and Path(p).is_dir():
            return str(Path(p))
    return ""


def resolve_dataset_dirs(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []

    # 情况1：总根目录（下级是 REDD/UK-DALE/REFIT）
    dataset_dirs = [p for p in root.iterdir() if p.is_dir() and p.name in SUPPORTED_DATASET_NAMES]
    if dataset_dirs:
        return sorted(dataset_dirs)

    # 情况2：root 本身就是某个数据集目录
    if root.name in SUPPORTED_DATASET_NAMES:
        return [root]

    return []


def ensure_session_defaults() -> None:
    guessed_root = resolve_initial_root()
    defaults = {
        "configured_root": guessed_root,
        "selected_dataset": "",
        "sim_anchor_day": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "zhidian_app_url": os.getenv("APP_ZHIDIAN_URL", "https://zhi-dian-xian-feng-user.streamlit.app").strip(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_global_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #05113a 0%, #040d30 100%);
            color: #eaf2ff;
        }
        
        /* ==================== 品牌区 ==================== */
        .brand-wrap {display:flex; align-items:center; gap:12px;}
        .brand-logo {
            width:52px; height:52px; border-radius:14px;
            display:flex; align-items:center; justify-content:center;
            background: linear-gradient(135deg, #22d3ee, #3b82f6);
            color:#fff; font-size:32px;
        }
        .brand-title {font-size:34px; font-weight:900; letter-spacing:1px;}
        .brand-subtitle {font-size:15px; color:#9ec3ff;}

        /* ==================== 监测卡片 ==================== */
        .monitor-card {
            border: 1px solid rgba(77, 138, 255, 0.40);
            border-radius: 16px;
            padding: 20px 18px 18px;
            background: linear-gradient(180deg, rgba(24,62,150,0.55), rgba(14,38,106,0.60));
            margin-bottom: 16px;
            min-height: 158px;
            display: flex;
            flex-direction: column;
        }
        .monitor-card.offline {
            background: linear-gradient(180deg, rgba(75,84,122,0.48), rgba(40,46,80,0.48));
        }
        .monitor-link {
            display: block;
            text-decoration: none !important;
            color: inherit !important;
        }
        .monitor-link:hover .monitor-card {
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(32, 100, 220, 0.24);
            border-color: rgba(119, 171, 255, 0.62);
            transition: all .18s ease;
        }

        .monitor-top {
            display:flex; 
            align-items: center; 
            justify-content: space-between; 
            gap: 12px;
            min-height: 42px;
            margin-bottom: 14px;
        }
        
        .monitor-user {
            display: flex;
            align-items: center;
            font-size: 22px;
            font-weight: 800;
            color: #e9f2ff;
            line-height: 1.2;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 32px;
            font-size: 14px;
            font-weight: 700;
            border-radius: 999px;
            padding: 4px 14px;
            border: 1px solid rgba(255,255,255,0.25);
        }
        .pill.online {background: rgba(8, 182, 93, 0.30); color: #62f4b1;}
        .pill.offline {background: rgba(180, 189, 207, 0.22); color: #dbe7ff;}

        .monitor-label {
            font-size: 14.5px;
            color: #9fb8e8;
            margin-bottom: 6px;
        }
        
        .monitor-power {
            font-size: 36px;
            font-weight: 900;
            color: #a7d3ff;
            line-height: 1.05;
        }
        .monitor-unit {
            font-size: 16px; 
            color: #9fb8e8; 
            margin-left: 6px;
        }

        /* ==================== 设备控制页面表格 - 新增优化 ==================== */
        .device-control-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
        }
        
        .device-control-table th,
        .device-control-table td {
            text-align: center !important;
            vertical-align: middle !important;
            padding: 14px 8px;
            font-size: 15.5px;
        }
        
        .device-control-table th {
            background: rgba(30, 70, 160, 0.65);
            color: #c5d9ff;
            font-weight: 700;
            border-bottom: 2px solid rgba(77, 138, 255, 0.6);
        }
        
        .device-control-table td {
            border-bottom: 1px solid rgba(77, 138, 255, 0.25);
            color: #eaf2ff;
        }
        
        /* 当前负荷列突出显示 */
        .device-control-table td:nth-child(4) {
            font-weight: 600;
            color: #a7d3ff;
        }
        
        /* 状态标签美化 */
        .status-tag {
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 700;
        }
        .status-online {background: rgba(8, 182, 93, 0.25); color: #62f4b1;}
        .status-offline {background: rgba(180, 189, 207, 0.22); color: #dbe7ff;}
        
        /* 远程控制 toggle 居中 */
        .stToggle {
            display: flex !important;
            justify-content: center !important;
            margin: 0 auto !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_logo_html() -> str:
    return """
    <div class="brand-wrap">
        <div class="brand-logo">⚡</div>
        <div>
            <div class="brand-title">智电先锋云端</div>
            <div class="brand-subtitle">智电 · 让每一度电都充满智慧</div>
        </div>
    </div>
    """


@st.cache_data(show_spinner=False)
def scan_available_datasets(root_dir: str) -> List[str]:
    root = Path(root_dir)
    dataset_dirs = resolve_dataset_dirs(root)
    return sorted([p.name for p in dataset_dirs])


def extract_house_number(name: str) -> str:
    match = re.search(r"House[_\-]?(\d+)", name, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", name)
    return digits[-1] if digits else "1"


@st.cache_data(show_spinner=False)
def scan_users(root_dir: str, selected_dataset: Optional[str] = None) -> List[Dict[str, str]]:
    root = Path(root_dir)
    users: List[Dict[str, str]] = []
    if not root.exists() or not root.is_dir():
        return users

    dataset_dirs = resolve_dataset_dirs(root)
    if selected_dataset:
        dataset_dirs = [p for p in dataset_dirs if p.name == selected_dataset]

    for ds in sorted(dataset_dirs):
        for house in sorted([p for p in ds.iterdir() if p.is_dir()]):
            excel_files = sorted(house.glob("*.xlsx"))
            if not excel_files:
                continue
            no = extract_house_number(house.name)
            users.append(
                {
                    "uid": f"{ds.name}_{house.name}",
                    "dataset": ds.name,
                    "location": LOCATION_MAP.get(ds.name, LOCATION_MAP["CUSTOM"]),
                    "user_name": f"用户{no}",
                    "house_key": house.name,
                    "house_path": str(house),
                }
            )
    return users


@st.cache_data(show_spinner=False)
def load_daily_series(house_dir: str) -> List[Dict[str, float]]:
    path = Path(house_dir)
    rows: List[Dict[str, float]] = []
    if not path.exists():
        return rows

    for file in sorted(path.glob("*.xlsx")):
        try:
            day = pd.to_datetime(file.stem).normalize()
        except Exception:
            continue

        total_kwh = 0.0
        try:
            excel = pd.ExcelFile(file)
            if "energy_bar_data" in excel.sheet_names:
                df = pd.read_excel(file, sheet_name="energy_bar_data")
                total_kwh = pd.to_numeric(df.get("energy_kwh"), errors="coerce").fillna(0).sum()
            elif "event_summary" in excel.sheet_names:
                df = pd.read_excel(file, sheet_name="event_summary")
                total_kwh = pd.to_numeric(df.get("energy_kwh"), errors="coerce").fillna(0).sum()
        except Exception:
            total_kwh = 0.0

        rows.append({"date": day, "total_kwh": float(total_kwh)})

    return rows


def compute_realtime_snapshot(users: List[Dict[str, str]]) -> List[Dict[str, object]]:
    now = pd.Timestamp.now()
    now_day = now.normalize()
    anchor_day = pd.to_datetime(st.session_state.get("sim_anchor_day", now.strftime("%Y-%m-%d"))).normalize()
    elapsed_days = max(0, int((now_day - anchor_day).days))

    quarter_idx = now.hour * 4 + now.minute // 15
    snap: List[Dict[str, object]] = []

    for u in users:
        series = load_daily_series(u["house_path"])
        if not series:
            continue

        day_item = series[elapsed_days % len(series)]

        avg_kw = float(day_item["total_kwh"]) / 24.0
        seed = abs(hash(u["uid"])) % 10000
        phase = (seed % 96) / 96.0 * 2.0 * np.pi
        factor = 0.78 + 0.32 * np.sin(2.0 * np.pi * quarter_idx / 96.0 + phase) + 0.1 * np.cos(
            4.0 * np.pi * quarter_idx / 96.0 + phase * 0.5
        )
        factor = float(np.clip(factor, 0.20, 1.55))
        realtime_kw = max(0.0, avg_kw * factor)

        default_online = ((quarter_idx + seed) % 11 != 0) and realtime_kw > 0.001
        online_state_key = f"device_online_{u['uid']}"
        if online_state_key not in st.session_state:
            st.session_state[online_state_key] = bool(default_online)
        online = bool(st.session_state[online_state_key])
        if not online:
            realtime_kw = 0.0

        snap.append(
            {
                **u,
                "online": online,
                "status": "在线" if online else "离线",
                "power_kw": realtime_kw,
                "power_w": realtime_kw * 1000.0,
            }
        )

    return snap


def enable_quarter_refresh() -> None:
    now = pd.Timestamp.now()
    remain_seconds = (15 - now.minute % 15) * 60 - now.second
    remain_seconds = 900 if remain_seconds <= 0 else remain_seconds
    components.html(
        f"""
        <script>
            const t = {int(remain_seconds)} * 1000;
            setTimeout(function() {{
                window.parent.location.reload();
            }}, t);
        </script>
        """,
        height=0,
    )


def render_runtime_monitor(snapshot: List[Dict[str, object]]) -> None:
    st.subheader("📊 运行监测页面")
    if not snapshot:
        st.warning("未扫描到用户数据，请先检查数据目录。")
        return

    st.caption("实时低频功率 · 每 15 分钟自动刷新一次 · 点击用户卡片可进入对应用户展示页")

    target_base = st.session_state.get("zhidian_app_url", "").strip()

    cols = st.columns(4, gap="medium")
    
    for idx, row in enumerate(snapshot):
        with cols[idx % 4]:
            card_cls = "monitor-card" if row["online"] else "monitor-card offline"
            pill_cls = "pill online" if row["online"] else "pill offline"
            
            power_display = f"{row['power_kw']:.3f}" if row["online"] else "0.000"
            
            params = urlencode(
                {
                    "autologin": "1",
                    "dataset": str(row.get("dataset", "")),
                    "house": str(row.get("house_key", "")),
                }
            )
            target_url = f"{target_base}?{params}" if target_base else ""

            if target_url:
                st.markdown(
                    f"""
                    <a class="monitor-link" href="{target_url}" target="_blank">
                        <div class="{card_cls}">
                            <div class="monitor-top">
                                <div class="monitor-user">{row['user_name']}</div>
                                <div class="{pill_cls}">{row['status']}</div>
                            </div>
                            <div class="monitor-label">实时低频用电功率</div>
                            <div class="monitor-power">{power_display}<span class="monitor-unit">kW</span></div>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        <div class="monitor-top">
                            <div class="monitor-user">{row['user_name']}</div>
                            <div class="{pill_cls}">{row['status']}</div>
                        </div>
                        <div class="monitor-label">实时低频用电功率</div>
                        <div class="monitor-power">{power_display}<span class="monitor-unit">kW</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_top_settings() -> None:
    with st.popover("☰ 设置", use_container_width=False):
        st.markdown("### 云端设置")
        zhidian_url = st.text_input("用户展示页地址", value=st.session_state.zhidian_app_url, placeholder="例如：https://zhi-dian-xian-feng-user.streamlit.app")
        st.session_state.zhidian_app_url = zhidian_url.strip()

        root_dir = st.text_input("数据目录", value=st.session_state.configured_root)
        if st.button("保存目录", use_container_width=True):
            st.session_state.configured_root = root_dir.strip().strip('"').strip("'")
            st.session_state.selected_dataset = ""
            st.session_state.sim_anchor_day = pd.Timestamp.now().strftime("%Y-%m-%d")
            st.cache_data.clear()
            st.rerun()

        dataset_names = scan_available_datasets(st.session_state.configured_root)
        if not dataset_names:
            st.info("当前目录未检测到可用数据集。")
            return

        if st.session_state.selected_dataset not in dataset_names:
            st.session_state.selected_dataset = dataset_names[0]

        idx = dataset_names.index(st.session_state.selected_dataset)
        selected = st.selectbox("当前展示数据集", dataset_names, index=idx)
        if selected != st.session_state.selected_dataset:
            st.session_state.selected_dataset = selected
            st.session_state.sim_anchor_day = pd.Timestamp.now().strftime("%Y-%m-%d")
            st.rerun()


def render_device_control(snapshot: List[Dict[str, object]]) -> None:
    st.subheader("⚙️ 设备控制页面")
    if not snapshot:
        st.warning("未扫描到用户数据，请先检查数据目录。")
        return

    # 表头
    head_cols = st.columns([1.1, 1.6, 1.8, 1.2, 1.0, 1.0, 1.3])
    headers = ["用户", "所属位置", "设备名称", "当前负荷", "设备状态", "开关状态", "远程控制"]
    for c, h in zip(head_cols, headers):
        c.markdown(f"**{h}**", unsafe_allow_html=True)

    # 数据行
    for row in snapshot:
        uid = str(row["uid"])
        online_state_key = f"device_online_{uid}"
        toggle_key = f"remote_toggle_{uid}"
        
        if online_state_key not in st.session_state:
            st.session_state[online_state_key] = bool(row.get("online", True))
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = bool(st.session_state[online_state_key])

        current_online = bool(st.session_state[online_state_key])
        power_display = f"{(row['power_kw'] if current_online else 0.0):.3f} kW"
        status_cls = "status-online" if current_online else "status-offline"
        status_text = "在线" if current_online else "离线"
        switch_text = "开" if current_online else "关"

        cols = st.columns([1.1, 1.6, 1.8, 1.2, 1.0, 1.0, 1.3])

        cols[0].write(row["user_name"])
        cols[1].write(row["location"])
        cols[2].write("智电先锋智能插座")
        cols[3].write(power_display)

        # 设备状态（带颜色标签）
        cols[4].markdown(f'<span class="status-tag {status_cls}">{status_text}</span>', unsafe_allow_html=True)
        
        cols[5].write(switch_text)
        
        # 远程控制 toggle（强制居中）
        with cols[6]:
            toggled = st.toggle(
                "远程控制", 
                key=toggle_key, 
                value=current_online,
                label_visibility="collapsed"
            )
            if toggled != current_online:
                st.session_state[online_state_key] = bool(toggled)
                st.rerun()

    # 底部说明
    st.caption("💡 点击远程控制开关可实时切换设备在线状态 · 每15分钟自动刷新")


def main() -> None:
    ensure_session_defaults()
    apply_global_theme()

    head_left, head_right = st.columns([8.5, 1.5])
    with head_left:
        st.markdown(build_logo_html(), unsafe_allow_html=True)
    with head_right:
        render_top_settings()

    st.markdown("---")

    with st.sidebar:
        st.markdown("### 导航")
        page = st.radio("选择页面", ["运行监测页面", "设备控制页面"], label_visibility="collapsed")

    if not st.session_state.configured_root:
        st.warning("请先通过右上角设置配置数据目录。")
        return

    if not st.session_state.selected_dataset:
        available = scan_available_datasets(st.session_state.configured_root)
        if not available:
            st.warning("当前目录下未找到可用数据集。")
            return
        st.session_state.selected_dataset = available[0]

    users = scan_users(st.session_state.configured_root, st.session_state.selected_dataset)
    # st.caption(f"当前数据集：{st.session_state.selected_dataset}")
    snapshot = compute_realtime_snapshot(users)

    enable_quarter_refresh()

    if page == "运行监测页面":
        render_runtime_monitor(snapshot)
    else:
        render_device_control(snapshot)


if __name__ == "__main__":
    main()