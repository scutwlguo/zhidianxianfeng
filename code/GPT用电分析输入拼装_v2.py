# -*- coding: utf-8 -*-
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_DAILY_JSON_ROOT = r"F:\研究生文件\节能减排\桌面程序代码\data\用电行为分析_json"
DEFAULT_OUTPUT_ROOT = r"F:\研究生文件\节能减排\桌面程序代码\output\gpt_analysis_inputs"

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RULE_CANDIDATES = [
    _SCRIPT_DIR / "电器健康使用参考表.json",
    _SCRIPT_DIR / "电器健康使用参考表.json",
    _SCRIPT_DIR / "电器健康使用参考表.json",
]
DEFAULT_RULE_FILE = next((p for p in _DEFAULT_RULE_CANDIDATES if p.exists()), _DEFAULT_RULE_CANDIDATES[-1])

def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _house_dir_to_user_label(house_dir: str, fallback: str = "用户") -> str:
    m = re.search(r"House(\d+)", str(house_dir), flags=re.IGNORECASE)
    return f"用户{int(m.group(1))}" if m else fallback

def _scan_target_files(root: Path, dataset: str, house_dir: str) -> List[Path]:
    user_dir = root / dataset / house_dir
    if not user_dir.exists() or (not user_dir.is_dir()):
        return []
    return sorted(user_dir.glob("*.json"))

def _guess_first_dataset_and_house(root: Path) -> Tuple[str, str]:
    if (not root.exists()) or (not root.is_dir()):
        return "", ""
    for ds in sorted([p for p in root.iterdir() if p.is_dir()]):
        houses = sorted([p for p in ds.iterdir() if p.is_dir()])
        for h in houses:
            if len(list(h.glob("*.json"))) > 0:
                return ds.name, h.name
    return "", ""

def _in_date_range(date_str: str, start_date: str, end_date: str) -> bool:
    d = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(d):
        return False
    s = pd.to_datetime(start_date, errors="coerce")
    e = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return True
    return s.normalize() <= d.normalize() <= e.normalize()

def _parse_hhmm_to_minute(text: str) -> int:
    s = str(text).strip()
    if ":" not in s:
        return 0
    hh, mm = s.split(":", 1)
    h = max(0, min(24, int(hh)))
    m = max(0, min(59, int(mm)))
    return max(0, min(1440, h * 60 + m))

def _split_window_to_ranges(start_m: int, end_m: int) -> List[Tuple[int, int]]:
    if start_m == end_m:
        return [(0, 1440)]
    if start_m < end_m:
        return [(start_m, end_m)]
    return [(start_m, 1440), (0, end_m)]

def _calc_overlap_minutes(interval: Tuple[int, int], window: Tuple[int, int]) -> int:
    s = max(int(interval[0]), int(window[0]))
    e = min(int(interval[1]), int(window[1]))
    return max(0, e - s)

def _signal_to_text(sig: Dict) -> str:
    t = sig.get("type", "")
    actual = sig.get("actual", "")
    threshold = sig.get("threshold", "")
    sev = sig.get("severity", "")
    if t == "single_duration_exceed":
        return f"单次运行 {actual} 分钟 > 阈值 {threshold} 分钟（{sev}）"
    if t == "daily_duration_exceed":
        return f"单日累计运行 {actual} 分钟 > 阈值 {threshold} 分钟（{sev}）"
    if t == "daily_energy_exceed_hint":
        return f"单日电量 {actual} kWh > 提示阈值 {threshold} kWh（{sev}）"
    return json.dumps(sig, ensure_ascii=False)

def _build_appliance_evidence(day_payload: Dict, rules: Dict) -> List[Dict]:
    app_rules = rules.get("appliance_rules", {}) if isinstance(rules, dict) else {}
    evidences = []
    for app in day_payload.get("appliances", []):
        code = str(app.get("appliance_code", ""))
        name = str(app.get("appliance_name_cn", code))
        r = app_rules.get(code, {}) if isinstance(app_rules, dict) else {}
        periods = app.get("on_off_periods", []) or []
        durations = [float(p.get("duration_min", 0) or 0) for p in periods]
        total_on_minutes = float(np.sum(durations)) if durations else 0.0
        max_single_on_minutes = float(np.max(durations)) if durations else 0.0
        switch_on_count = int(app.get("switch_on_count", 0) or 0)
        energy_kwh = float(app.get("energy_kwh", 0.0) or 0.0)

        th_single = float(r.get("max_single_on_minutes", 0) or 0)
        th_daily = float(r.get("max_daily_on_minutes", 0) or 0)
        th_energy = float(r.get("abnormal_daily_energy_kwh_hint", 0) or 0)

        strong_signals = []
        if th_single > 0 and max_single_on_minutes > th_single:
            strong_signals.append({"type":"single_duration_exceed","actual":round(max_single_on_minutes,3),"threshold":th_single,"severity":"high" if max_single_on_minutes > th_single * 1.5 else "medium"})
        if th_daily > 0 and total_on_minutes > th_daily:
            strong_signals.append({"type":"daily_duration_exceed","actual":round(total_on_minutes,3),"threshold":th_daily,"severity":"high" if total_on_minutes > th_daily * 1.5 else "medium"})
        if th_energy > 0 and energy_kwh > th_energy:
            strong_signals.append({"type":"daily_energy_exceed_hint","actual":round(energy_kwh,6),"threshold":th_energy,"severity":"high" if energy_kwh > th_energy * 1.5 else "medium"})

        weak_signals = []
        typical_windows = r.get("typical_windows", []) if isinstance(r, dict) else []
        if isinstance(typical_windows, list) and len(typical_windows) > 0 and len(periods) > 0:
            windows = []
            for w in typical_windows:
                if isinstance(w, list) and len(w) == 2:
                    windows.extend(_split_window_to_ranges(_parse_hhmm_to_minute(str(w[0])), _parse_hhmm_to_minute(str(w[1]))))
            total_m = 0
            in_typical_m = 0
            for p in periods:
                s = int(p.get("start_minute", 0) or 0)
                e = int(p.get("end_minute", s) or s)
                if e <= s:
                    e = min(1440, s + 1)
                seg = (max(0, min(1440, s)), max(0, min(1440, e)))
                seg_len = max(0, seg[1] - seg[0])
                total_m += seg_len
                ov = 0
                for w in windows:
                    ov += _calc_overlap_minutes(seg, w)
                in_typical_m += min(seg_len, ov)
            if total_m > 0:
                outside_ratio = 1.0 - (float(in_typical_m) / float(total_m))
                if outside_ratio >= 0.6:
                    weak_signals.append({"type":"mostly_outside_typical_windows","outside_ratio":round(outside_ratio,4),"note":"仅作辅助信号，不能单独判定异常"})

        evidence_summary = [_signal_to_text(sig) for sig in strong_signals]
        for ws in weak_signals:
            if ws.get("type") == "mostly_outside_typical_windows":
                evidence_summary.append(f"运行时段有 {round(float(ws.get('outside_ratio',0))*100,1)}% 落在典型时段外（弱信号）")

        evidences.append({
            "appliance_code": code,
            "appliance_name_cn": name,
            "energy_kwh": round(energy_kwh, 6),
            "switch_on_count": switch_on_count,
            "total_on_minutes": round(total_on_minutes, 3),
            "max_single_on_minutes": round(max_single_on_minutes, 3),
            "rule_tier": r.get("rule_tier", "unknown"),
            "strong_signals": strong_signals,
            "weak_signals": weak_signals,
            "analysis_focus": r.get("analysis_focus", []),
            "risk_hints": r.get("risk_hints", []),
            "suggestion_directions": r.get("suggestion_directions", []),
            "evidence_summary": evidence_summary,
        })
    evidences.sort(key=lambda x: x.get("energy_kwh", 0.0), reverse=True)
    return evidences

def _summarize_days(day_payloads: List[Dict]) -> Dict:
    if len(day_payloads) == 0:
        return {"day_count":0,"avg_daily_total_kwh":0.0,"avg_daily_cost_cny":0.0,"top_energy_appliances":[],"appliance_stats":[]}
    daily_kwh, daily_cost = [], []
    app_energy, app_switch_count = defaultdict(float), defaultdict(int)
    app_duration_total, app_duration_samples = defaultdict(float), defaultdict(int)
    for p in day_payloads:
        ds = p.get("daily_summary", {})
        daily_kwh.append(float(ds.get("total_energy_kwh", 0.0) or 0.0))
        daily_cost.append(float(ds.get("total_cost_cny", 0.0) or 0.0))
        for a in p.get("appliances", []):
            code = str(a.get("appliance_code", "UNK"))
            app_energy[code] += float(a.get("energy_kwh", 0.0) or 0.0)
            app_switch_count[code] += int(a.get("switch_on_count", 0) or 0)
            for seg in a.get("on_off_periods", []):
                dur = float(seg.get("duration_min", 0.0) or 0.0)
                if np.isfinite(dur) and dur > 0:
                    app_duration_total[code] += dur
                    app_duration_samples[code] += 1
    app_stats = []
    for code in sorted(app_energy.keys(), key=lambda x: app_energy[x], reverse=True):
        avg_dur = app_duration_total[code] / app_duration_samples[code] if app_duration_samples[code] > 0 else 0.0
        app_stats.append({
            "appliance_code": code,
            "total_energy_kwh": round(float(app_energy[code]), 6),
            "avg_daily_energy_kwh": round(float(app_energy[code]) / max(1, len(day_payloads)), 6),
            "total_switch_on_count": int(app_switch_count[code]),
            "avg_single_on_duration_min": round(float(avg_dur), 3),
        })
    return {
        "day_count": int(len(day_payloads)),
        "avg_daily_total_kwh": round(float(np.mean(daily_kwh)), 6),
        "avg_daily_cost_cny": round(float(np.mean(daily_cost)), 6),
        "top_energy_appliances": app_stats[:8],
        "appliance_stats": app_stats,
    }

def _extract_single_day_hint(pkg: Dict) -> Dict:
    return (((pkg.get("record") or {}).get("daily_record") or {}).get("gpt_ready_hint") or {})

def _extract_report_style_rules(pkg: Dict) -> List[str]:
    return (((pkg.get("reference_rules") or {}).get("global_rules") or {}).get("report_style_rules") or [])

def _build_day_key_facts(day_payload: Dict, evidence_list: List[Dict]) -> List[str]:
    facts = []
    ds = day_payload.get("daily_summary", {}) or {}
    facts.append(f"当日总用电量 {round(float(ds.get('total_energy_kwh',0.0) or 0.0),3)} kWh，总电费 {round(float(ds.get('total_cost_cny',0.0) or 0.0),3)} 元。")
    top_apps = [x for x in evidence_list if float(x.get("energy_kwh", 0)) > 0][:3]
    if top_apps:
        facts.append("当日主要耗能设备为：" + "、".join([f"{x['appliance_name_cn']} {round(float(x['energy_kwh']),3)} kWh" for x in top_apps]) + "。")
    strong_items = []
    for x in evidence_list:
        if x.get("strong_signals"):
            sig_desc = "；".join(x.get("evidence_summary", [])[:2])
            strong_items.append(f"{x['appliance_name_cn']}：{sig_desc}")
    if strong_items:
        facts.append("优先关注的强证据设备包括：" + "；".join(strong_items[:4]) + "。")
    return facts

def _build_focus_block(hint_focus: List[str]) -> str:
    if not hint_focus:
        return "- 无"
    return "\n".join([f"- {x}" for x in hint_focus])

def _build_prompt_text(pkg: Dict) -> str:
    task = pkg.get("task", {}) or {}
    target = pkg.get("target", {}) or {}
    ap = pkg.get("analysis_period", {}) or {}
    principles = (pkg.get("reference_rules", {}) or {}).get("usage_principles", []) or []
    report_style_rules = _extract_report_style_rules(pkg)
    user_label = target.get("user_label", "用户")
    hint = _extract_single_day_hint(pkg)
    hint_task = str(hint.get("task", "") or "").strip()
    hint_focus = hint.get("focus", []) if isinstance(hint.get("focus", []), list) else []
    principles_text = "\n".join([f"- {x}" for x in principles]) if principles else "- 无"
    report_style_text = "\n".join([f"- {x}" for x in report_style_rules]) if report_style_rules else "- 无"
    key_facts = []
    if target.get("scope") == "single_day_single_user":
        day_payload = (((pkg.get("record") or {}).get("daily_record")) or {})
        evidence_list = (((pkg.get("record") or {}).get("appliance_evidence")) or [])
        key_facts = _build_day_key_facts(day_payload, evidence_list)
    key_facts_text = "\n".join([f"- {x}" for x in key_facts]) if key_facts else "- 无"

    return f'''你是一名资深家庭能源管理顾问，请严格基于输入JSON完成分析，不要编造数据，不要脱离输入数据做过度推断。

【分析对象】
- 分析范围：{target.get("scope", "")}
- 用户：{user_label}
- 日期：{ap.get("start_date", "")} ~ {ap.get("end_date", "")}

【本次分析任务（优先遵循）】
{hint_task if hint_task else "请基于输入数据分析主要用电习惯、潜在高耗能行为，并给出可执行节能建议。"}

【重点关注维度】
{_build_focus_block(hint_focus)}

【关键事实摘要】
{key_facts_text}

【证据使用规则】
1) 优先依据 strong_signals 判断问题，包括：
   - 单次运行时间超阈值
   - 单日累计运行时间超阈值
   - 单日电量超过提示阈值
2) weak_signals（如典型时段外运行、夜间运行、负荷集中）只能作为辅助证据，不能单独判定异常。
3) 若某设备为常开型设备，应重点关注日电量异常抬升，而不是运行时段本身。
4) 若输入 JSON 中 analysis_focus、risk_hints、suggestion_directions 非空，应优先吸收后再生成设备建议。

【参考规则使用原则】
{principles_text}

【写作风格约束】
{report_style_text}

【输出要求】
1) 先给出“总体结论”3-5条，必须先点名最值得优化的 1-3 个设备。
2) 再输出“逐设备分析表格”，列为：
   设备 | 关键证据 | 用电习惯/潜在问题 | 可执行建议 | 预期节能方向
3) 最后输出“Top3 优先建议”，按“节能收益 × 可执行性”综合排序。
4) 建议必须具体到设备和行为，不要只写“减少使用”“注意节能”这类空泛表述。
5) 若证据不足，请使用保守措辞，例如“可能存在”“建议关注”“可进一步确认”。

【任务备注】
- 任务类型：{task.get("task_type", "")}
- 证据策略：{task.get("evidence_strategy", "")}
- 保守措辞：{task.get("require_conservative_wording", True)}
'''

def _build_single_day_package(dataset: str, house_dir: str, date_str: str, day_payload: Dict, rules: Dict) -> Dict:
    user = day_payload.get("user", {})
    user_label = str(user.get("user_name", "")).strip() or _house_dir_to_user_label(house_dir, fallback=house_dir)
    evidence = _build_appliance_evidence(day_payload, rules)
    day_hint = day_payload.get("gpt_ready_hint", {}) if isinstance(day_payload.get("gpt_ready_hint", {}), dict) else {}
    record = {"user": user, "daily_record": day_payload, "appliance_evidence": evidence}
    return {
        "meta": {"schema_version": "2.1", "purpose": "for_gpt_energy_behavior_analysis", "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")},
        "target": {"dataset": dataset, "scope": "single_day_single_user", "house_dir": house_dir, "user_name": user.get("user_name", house_dir), "user_label": user_label},
        "analysis_period": {"start_date": date_str, "end_date": date_str, "day_count": 1},
        "task": {
            "task_type": "single_day_behavior_analysis",
            "evidence_strategy": "strong_signals_first_weak_signals_assist",
            "require_conservative_wording": True,
            "hint_task": day_hint.get("task", ""),
            "hint_focus": day_hint.get("focus", []),
        },
        "reference_rules": rules,
        "record": record,
    }

def _build_date_range_package(dataset: str, house_dir: str, day_payloads: List[Dict], rules: Dict, req_start: str, req_end: str) -> Dict:
    user_info = day_payloads[0].get("user", {})
    user_label = str(user_info.get("user_name", "")).strip() or _house_dir_to_user_label(house_dir, fallback=house_dir)
    dates = [str(x.get("date", "")) for x in day_payloads]
    daily_records_with_evidence = [{"date": str(p.get("date", "")), "daily_record": p, "appliance_evidence": _build_appliance_evidence(p, rules)} for p in day_payloads]

    focus_pool, task_pool = [], []
    for p in day_payloads:
        hint = p.get("gpt_ready_hint", {})
        if isinstance(hint, dict):
            if hint.get("task"):
                task_pool.append(str(hint.get("task")))
            if isinstance(hint.get("focus", []), list):
                focus_pool.extend([str(x) for x in hint.get("focus", []) if str(x).strip()])
    dedup_focus, seen = [], set()
    for x in focus_pool:
        if x not in seen:
            dedup_focus.append(x)
            seen.add(x)

    return {
        "meta": {"schema_version": "2.1", "purpose": "for_gpt_energy_behavior_analysis", "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")},
        "target": {"dataset": dataset, "scope": "date_range_single_user", "house_dir": house_dir, "user_name": user_info.get("user_name", house_dir), "user_label": user_label},
        "analysis_period": {"start_date": dates[0], "end_date": dates[-1], "day_count": len(day_payloads), "requested_start_date": req_start, "requested_end_date": req_end},
        "task": {
            "task_type": "date_range_habit_analysis",
            "evidence_strategy": "strong_signals_first_weak_signals_assist",
            "require_conservative_wording": True,
            "hint_task": task_pool[0] if task_pool else "",
            "hint_focus": dedup_focus,
        },
        "reference_rules": rules,
        "derived_summary": _summarize_days(day_payloads),
        "daily_records": day_payloads,
        "daily_records_with_evidence": daily_records_with_evidence,
    }

def build_gpt_analysis_package(dataset: str, house_dir: str, pack_type: str = "single-day", date: str = "", start_date: str = "", end_date: str = "", daily_json_root: str = DEFAULT_DAILY_JSON_ROOT, rules_file: str = str(DEFAULT_RULE_FILE), max_days: int = 31):
    if not dataset:
        raise ValueError("dataset 不能为空")
    if not house_dir:
        raise ValueError("house_dir 不能为空")
    if pack_type not in {"single-day", "multi-day"}:
        raise ValueError("pack_type 仅支持 single-day 或 multi-day")
    daily_root = Path(daily_json_root)
    rule_json = _load_json(Path(rules_file))
    files = _scan_target_files(daily_root, dataset, house_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"未找到日级JSON：{daily_root / dataset / house_dir}")

    if pack_type == "single-day":
        target_date = date.strip() or sorted([fp.stem for fp in files])[-1]
        fp = daily_root / dataset / house_dir / f"{target_date}.json"
        if not fp.exists():
            raise FileNotFoundError(f"未找到指定日期数据：{fp}")
        payload = _load_json(fp)
        package = _build_single_day_package(dataset, house_dir, target_date, payload, rule_json)
    else:
        if (not start_date) or (not end_date):
            raise ValueError("multi-day 模式必须提供 start_date 与 end_date")
        day_payloads = []
        for fp in files:
            try:
                payload = _load_json(fp)
            except Exception:
                continue
            if _in_date_range(str(payload.get("date", "")), start_date, end_date):
                day_payloads.append(payload)
        day_payloads = sorted(day_payloads, key=lambda x: str(x.get("date", "")))
        if max_days > 0 and len(day_payloads) > max_days:
            day_payloads = day_payloads[-max_days:]
        if len(day_payloads) == 0:
            raise ValueError("筛选后没有可用日期数据，请检查日期范围")
        package = _build_date_range_package(dataset, house_dir, day_payloads, rule_json, start_date, end_date)

    prompt_text = _build_prompt_text(package)
    return package, prompt_text

def main() -> None:
    parser = argparse.ArgumentParser(description="拼装GPT用电分析输入")
    parser.add_argument("--pack-type", type=str, default="single-day", choices=["single-day", "multi-day"], help="打包类型")
    parser.add_argument("--daily-json-root", type=str, default=DEFAULT_DAILY_JSON_ROOT, help="日级JSON根目录")
    parser.add_argument("--rules-file", type=str, default=str(DEFAULT_RULE_FILE), help="电器健康参考表JSON")
    parser.add_argument("--dataset", type=str, default="REDD", help="数据集名，如 REDD")
    parser.add_argument("--house-dir", type=str, default="REDD_House6_stats", help="房屋目录名，如 REDD_House1_stats")
    parser.add_argument("--date", type=str, default="", help="single-day：指定日期 YYYY-MM-DD，不填则取最新一天")
    parser.add_argument("--start-date", type=str, default="", help="起始日期 YYYY-MM-DD，可选")
    parser.add_argument("--end-date", type=str, default="", help="结束日期 YYYY-MM-DD，可选")
    parser.add_argument("--max-days", type=int, default=31, help="最多打包天数，默认31")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="输出目录")
    args = parser.parse_args()

    dataset = str(args.dataset or "").strip()
    house_dir = str(args.house_dir or "").strip()
    if not dataset or not house_dir or len(_scan_target_files(Path(args.daily_json_root), dataset, house_dir)) == 0:
        guessed_dataset, guessed_house = _guess_first_dataset_and_house(Path(args.daily_json_root))
        if guessed_dataset and guessed_house:
            dataset, house_dir = guessed_dataset, guessed_house
            print(f"[INFO] 自动选择数据：dataset={dataset}, house_dir={house_dir}")

    package, prompt_text = build_gpt_analysis_package(
        dataset=dataset,
        house_dir=house_dir,
        pack_type=args.pack_type,
        date=args.date,
        start_date=args.start_date,
        end_date=args.end_date,
        daily_json_root=args.daily_json_root,
        rules_file=args.rules_file,
        max_days=args.max_days,
    )

    ap = package.get("analysis_period", {})
    if args.pack_type == "single-day":
        out_root = Path(args.output_root) / dataset / house_dir / str(ap.get("start_date", "single_day"))
    else:
        out_root = Path(args.output_root) / dataset / house_dir / f"{ap.get('start_date', 'start')}_to_{ap.get('end_date', 'end')}"
    out_root.mkdir(parents=True, exist_ok=True)

    json_out = out_root / "analysis_input.json"
    prompt_out = out_root / "analysis_prompt.txt"
    json_out.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_out.write_text(prompt_text, encoding="utf-8")
    print(f"[OK] 结构化输入：{json_out}")
    print(f"[OK] 提示词模板：{prompt_out}")

if __name__ == "__main__":
    main()
