#!/usr/bin/env python3
"""Fleet & training board collector. Probes the four boxes over ssh (read-only),
writes data/fleet.json (latest) and data/curves.json (training series), then commits
and pushes. Runs from cron every 10 minutes; safe to run by hand."""
import json, os, re, shlex, inspect, subprocess, time, datetime as dt, pathlib
from active_jobs import collect_active_jobs, science_counts

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles")   # was a fixed UTC-7, wrong after the November DST change

def ssh(host, cmd, t=40):
    try:
        r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", f"ConnectTimeout=10", host, cmd],
                           capture_output=True, text=True, timeout=t)
        return r.stdout if r.returncode in (0, 1) else None
    except Exception:
        return None

def gpus(host):
    out = ssh(host, "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits")
    if not out: return None
    cards = []
    for line in out.strip().splitlines():
        try:
            i, u, tot, ut = [x.strip() for x in line.split(",")]
            # a card whose utilization reads [N/A] (driver error state, seen on
            # 4090-jm card 0 on 2026-09-21) must still show on the board, flagged
            card = {"idx": int(i), "mem_used": int(u), "mem_total": int(tot)}
            try:
                card["util"] = int(ut)
            except ValueError:
                card["util"] = -1
                card["error"] = ut
            cards.append(card)
        except ValueError:
            pass
    return cards

def tail(host, path, n=3):
    out = ssh(host, f"tail -n {n} {path} 2>/dev/null")
    return (out or "").strip().splitlines()

def count_lines(host, glob):
    out = ssh(host, f"wc -l {glob} 2>/dev/null | grep -v total")
    res = {}
    for line in (out or "").splitlines():
        try:
            n, p = line.split()
            res[os.path.basename(p)] = int(n)
        except ValueError:
            pass
    return res

# ---------------- A800: WWW stage7 (GRPO training) ----------------
METRIC_KEYS = {"critic/rewards/mean": "reward", "response_length/mean": "resp_len",
               "actor/entropy_loss": "entropy", "actor/kl_loss": "kl", "actor/pg_loss": "pg_loss"}

def stage7(curves):
    host = "A800"
    logs = ssh(host, "ls -t /data0/xyf/www2027-1/logs/train_*full.log 2>/dev/null")
    if logs is None: return None
    arms = []
    for path in logs.split():
        arm = re.sub(r".*/train_(.*)\.log$", r"\1", path)
        out = ssh(host, f"grep -oE 'step:[0-9]+ - .*' {path} | sed 's/\\x1b\\[[0-9;]*m//g' | tail -n 400", t=60) or ""
        series = []
        for line in out.splitlines():
            m = re.match(r"step:(\d+) - (.*)", line)
            if not m: continue
            step = int(m.group(1)); rec = {"step": step}
            for k, v in re.findall(r"([\w/]+):(-?[\d.]+)", m.group(2)):
                if k in METRIC_KEYS:
                    try: rec[METRIC_KEYS[k]] = float(v)
                    except ValueError: pass
            if len(rec) > 1: series.append(rec)
        # merge with stored series (log tail may not cover the start)
        old = {r["step"]: r for r in curves.get("stage7", {}).get(arm, [])}
        for r in series: old[r["step"]] = r
        merged = [old[k] for k in sorted(old)]
        curves.setdefault("stage7", {})[arm] = merged
        last = merged[-1] if merged else {}
        arms.append({"arm": arm, "steps": last.get("step", 0), "last": last})
    q = tail(host, "/data0/xyf/www2027-1/logs/queue_runner_stage7.out", 2)
    running = None
    for line in q:
        m = re.search(r"(?:LAUNCH|waiting for) ([\w-]+)", line)
        if m: running = m.group(1)
    total_arms = 6
    FULL = 149  # steps per full arm (T0 arms ran 149)
    s7 = [a for a in arms if "envonly" in a["arm"]]
    done = len([a for a in s7 if a["steps"] >= FULL and a["arm"] != running])
    for a in arms: a["tier"] = "stage7" if "envonly" in a["arm"] else "T0"
    qdone = any("QUEUE DONE" in line for line in q)
    return {"id": "www-stage7", "repo": "WWW2027-1", "title": "stage7 六臂修正版 GRPO", "box": "A800",
            "cards": [] if qdone else [0, 1, 2, 3], "kind": "train",
            "status": "done" if qdone else ("running" if running else "unknown"),
            "progress": {"done": max(0, min(done, total_arms)), "total": total_arms, "unit": "臂"},
            "detail": ("训练队列 QUEUE DONE 09-20 10:08 LA；T0 评测 09-21 完，S*/T1/噪声底 09-23 手动跑完" if qdone else f"当前臂 {running}") + (f"，step {[a for a in arms if a['arm']==running][0]['steps']}/{FULL}" if running and any(a['arm']==running for a in arms) else ""),
            "arms": arms}

# ---------------- 3090: ICLR2027-9 ladder ----------------
def ladder():
    host = "3090"
    s1 = ssh(host, "grep -hE '^END|^SCORED|LAUNCH|ABORT|FAIL' /data/xyf/iclr9_ladder_status 2>/dev/null") or ""
    s2 = ssh(host, "grep -hE '^END|^SCORED|ABORT|FAIL' /data/xyf/iclr9_ladder_status_lane2 2>/dev/null") or ""
    ends = re.findall(r"^END (\w+) ([\d.]+) rc=(\d+) wall_s=(\d+)", s1 + "\n" + s2, re.M)
    done = {(a, t): int(w) for a, t, rc, w in ends if rc == "0"}
    # in-flight: newest gen log per lane
    inflight = []
    for lane, pat in (("lane1", "ladder_[br][a-z]*_t*.log"), ("lane2", "ladder_lane2_[a-z]*_t*.log")):
        out = ssh(host, f"f=$(ls -t /data/xyf/ICLR2027-9/logs/{pat} 2>/dev/null | head -1); echo $f; tail -c 3000 $f 2>/dev/null | tr '\\r' '\\n' | grep -o 'Processed prompts: *[0-9]*/[0-9]*' | tail -1")
        if not out: continue
        lines = out.strip().splitlines()
        if not lines: continue
        name = os.path.basename(lines[0]).replace(".log", "")
        m = re.match(r"ladder_(?:lane2_)?(\w+)_t([\d.]+)", name)
        if not m: continue
        arm, T = m.group(1), m.group(2)
        if (arm, T) in done: continue
        prog = re.search(r"(\d+)/(\d+)", lines[-1]) if len(lines) > 1 else None
        inflight.append({"lane": lane, "arm": arm, "T": T,
                         "prompts": [int(prog.group(1)), int(prog.group(2))] if prog else None})
    fails = re.findall(r"(ABORT|FAIL)[^\n]*", s1 + s2)
    return {"id": "iclr9-ladder", "repo": "ICLR2027-9", "title": "14B 温度阶梯（TP2，两路）", "box": "3090",
            "cards": [] if len(done) >= 12 else [0, 1, 6, 7], "kind": "gen",
            "status": "done" if len(done) >= 12 else ("failed" if fails else "running"),
            "progress": {"done": len(done), "total": 12, "unit": "run"},
            "detail": "；".join(f"{x['lane']} {x['arm']} T={x['T']}" + (f" {x['prompts'][0]}/{x['prompts'][1]} 题" if x['prompts'] else "") for x in inflight) or "等待下一对",
            "runs": sorted([{"arm": a, "T": float(t), "wall_h": round(w/3600, 2)} for (a, t), w in done.items()], key=lambda r: (r["T"], r["arm"])),
            "alerts": fails[-2:]}

# ---------------- chronocheck-style status files (fuxin / new105) ----------------
def chrono(host, status, outdir, title, cards, total_rows, arms, repo="ICLR2027-6"):
    st = tail(host, status, 1)
    if not st: return None
    last = st[-1]
    status_word = "running"
    if last.startswith("DONE"): status_word = "done"
    elif last.startswith("FAILED"): status_word = "failed"
    counts = count_lines(host, f"{outdir}/*.jsonl")
    done_rows = sum(counts.get(f"{a}.jsonl", 0) for a in arms)
    m = re.search(r"(\d{4}-\d\d-\d\dT[\d:]+)", last)
    return {"id": os.path.basename(status), "repo": repo, "title": title, "box": host, "cards": cards,
            "kind": "gen", "status": status_word,
            "progress": {"done": done_rows, "total": total_rows * len(arms), "unit": "行"},
            "detail": last[:110], "per_arm": {a: counts.get(f"{a}.jsonl", 0) for a in arms}}

# ---------------- 194: ICLR2027-3 B7-TP2 (72B-AWQ, cards 2/3) ----------------
def b7tp2():
    host = "194-yyd"
    st = ssh(host, "tail -n 3 /data/yyd/b7_status 2>/dev/null; echo ==; wc -l /data/yyd/ICLR2027-3/experiments/credo/results/raw_72b_awq/qwen2.5-vl-72b-awq/*/responses.jsonl 2>/dev/null | grep -v total; echo ==; tail -n 2 /data/yyd/provision_status 2>/dev/null")
    if st is None: return None
    parts = st.split("==")
    lines = [l for l in parts[0].strip().splitlines() if l.strip()]
    if not lines: return None
    last = lines[-1]
    rows = sum(int(l.split()[0]) for l in parts[1].strip().splitlines() if l.strip() and l.split()[0].isdigit())
    status = "running"
    if last.startswith("END b7tp2 0"): status = "done"
    elif last.startswith("END b7tp2") or "DOES_NOT_FIT" in last:
        # closed 09-19 by PREREG outcome (gate failed legitimately, ea7912b): terminal, not an alert
        status = "done"; last = "已关闭(门禁不过,PREREG_B7_72B outcome) " + last
    return {"id": "iclr3-b7tp2", "repo": "ICLR2027-3", "title": "B7-TP2 72B-AWQ 第四规模点（194 双卡）", "box": host,
            "cards": [] if status == "done" else [2, 3], "kind": "gen", "status": status,
            "progress": {"done": rows, "total": 6294, "unit": "题"},
            "detail": last[:110]}

# ---------------- generic status-file jobs (terminal lines drive the board) ----------------
STATUS_JOBS = [
    ("3090", "/data/xyf/iclr2_uitars_seed1_status", "ICLR2027-2", "UI-TARS 二次抽样（3090 卡 0/1 TP2 m2w test_domain；android 09-21 03:37 CST 已落地）", [0, 1]),
    ("A800", "/data0/xyf/www2027-1/logs/eval_matrix_t0_corrected.out", "WWW2027-1", "修正版六臂 T0 评测（30 作业，四卡；完了自动接 S* → outcome T1 → 噪声底）", [0, 1, 2, 3]),
    ("A800", "/data0/xyf/imwut_B_status", "IMWUT2027-1", "ORAL_GAP B1 max-q 门控 / B2 覆盖率扫 / B3 seeds 45-46（四道，排在 WWW 链后）", [0, 1, 2, 3]),
    ("194-yyd", "/data/yyd/iclr9_ladder_seed2_status_laneA", "ICLR2027-9", "14B 阶梯 seed 2 base 臂 lane A（T 0.6 → 1.0 0.2 0.8；09-21 00:20 LA 拆成四道）", [0]),
    ("194-yyd", "/data/yyd/iclr9_ladder_seed2_status_laneB", "ICLR2027-9", "14B 阶梯 seed 2 rlvr 臂 lane B（T 0.6 → 1.0 0.2 0.4）", [1]),
    ("194-yyd", "/data/yyd/iclr9_ladder_seed2_status_laneD", "ICLR2027-9", "14B 阶梯 seed 2 base 臂 lane D（T 1.2 0.4）", [2]),
    ("194-yyd", "/data/yyd/iclr9_ladder_seed2_status_laneE", "ICLR2027-9", "14B 阶梯 seed 2 rlvr 臂 lane E（T 0.8 1.2）", [3]),
    ("fuxin", "/data/xyf/iclr9_ladder_seed2_fuxin_status_laneF", "ICLR2027-9", "14B 阶梯 seed 2 base 臂 lane F（fuxin；T 0.4 于 09-22 00:36 LA 被 root 杀掉不重发；T 1.2 在卡 6）", [6]),
    ("fuxin", "/data/xyf/iclr9_ladder_seed2_fuxin_status_laneG", "ICLR2027-9", "14B 阶梯 seed 2 rlvr 臂 lane G（fuxin；等 rlvr 权重落地，卡 6 忙则卡 4）", []),
    ("3090", "/data/xyf/science/logs/full_eval.log", "Science", "Phase-Trans 盲态全量评测（≤14B 30 模型 × 17 任务，跟随 pin 下载；指标隔离在 _blind/ 未读，只记 rc/秒）", [6, 7]),
    ("new105", "/home/xyf/science/logs/full_eval.log", "Science", "Phase-Trans 盲态全量评测（≤4B 子集 20 模型 × 17 任务；box-floor 行）", [1]),
    ("4090-jm", "/home/yxy/science/logs/full_eval.log", "Science", "Phase-Trans 盲态全量评测（1.4B–8B 段 9 模型；09-24 21:00 LA 按用户指示撤下，yxy 在用；九模型清单待另派机器）", [1]),
    ("A800", "/data0/xyf/science/logs/full_eval.log", "Science", "Phase-Trans A800 腿评测：Qwen2.5-32B（卡 3）/72B+OPT-66b（卡 1,2）/14B+OLMo-13B（卡 0）；只记 rc/秒", []),
    ("A800", "/data0/xyf/science/logs/dl_models.log", "Science", "Phase-Trans A800 腿：Qwen2.5-32B/72B + OPT-30b/66b pin 下载（≈400 GB，4.7 MB/s；只占盘不占卡）", []),
]

# ---------------- card ownership by process account (2026-09-27) ----------------
# fleet_scan.py was fixed on 09-26 to attribute by the account that owns the process; the board still
# called a card "ours" only when a registered job listed it, so our unregistered chains showed as "other".
OUR_USERS = {"A800": {"xyf"}, "3090": {"xyf"}, "new105": {"xyf"}, "fuxin": {"xyf"}, "194-yyd": {"yyd"}, "4090-jm": set()}

def card_users(host):
    out = ssh(host, "nvidia-smi --query-gpu=index,uuid --format=csv,noheader; echo ---; "
                    "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader; echo ---; ps -eo pid=,user=")
    if not out or out.count("---") < 2: return {}
    a, b, c = out.split("---", 2)
    idx = {}
    for l in a.strip().splitlines():
        try: i, u = [x.strip() for x in l.split(",")]; idx[u] = int(i)
        except ValueError: pass
    user = {}
    for l in c.strip().splitlines():
        f = l.split()
        if len(f) == 2 and f[0].isdigit(): user[f[0]] = f[1]
    res = {}
    for l in b.strip().splitlines():
        try: u, pid = [x.strip() for x in l.split(",")]
        except ValueError: continue
        if u in idx and pid.isdigit(): res.setdefault(idx[u], set()).add(user.get(pid, "?"))
    return res

# ---------------- current work (2026-09-27) ----------------
def camco_jobs():
    h, W = "3090", "/data/xyf/scratch/camco"
    cmd = (f"cd {W}; "
           "echo E3 $(grep -acE '\\[final\\] card[0-9] done score_' e3/logs/e3.log) $(ps -eo args | grep -cE '^bash chain_card[0-9](_v2)?.sh') $(grep -ac FAIL e3/logs/e3.log); "
           "echo E4 $(ls AAAI2027-4/results/e4/chair_*.json 2>/dev/null | wc -l) $(ps -eo args | grep -cE '^bash (experiments/camco/)?run_e4_3090.sh') $(grep -ac FAIL e4/logs/e4.log); "
           "echo E4LAST $(tail -n 1 e4/logs/e4.log | cut -c1-110); "
           "echo V2 $(ls v2/state/score_*.done 2>/dev/null | wc -l) $(ps -eo args | grep -cE '^bash run_v2_3090.sh') "
           "$(for x in $(grep -aoE 'FAIL [a-z0-9_]+' v2/logs/v2.log 2>/dev/null | cut -d' ' -f2 | sort -u); do [ -e v2/state/$x.done ] || echo $x; done | wc -l) "
           "$(ls -d v2/state/claim_* 2>/dev/null | wc -l); "
           "echo V2CARDS $(grep -aoE 'card[0-9] start' v2/logs/v2.log 2>/dev/null | tail -n 16 | sort -u | tr -dc '0-9 '); "
           "echo V2LAST $(tail -n 1 v2/logs/v2.log 2>/dev/null | cut -c1-110)")
    out = ssh(h, cmd)
    if not out: return []
    kv = {l.split(" ", 1)[0]: (l.split(" ", 1)[1] if " " in l else "") for l in out.strip().splitlines()}
    jobs = []
    try:
        d, ch, fl = (int(x) for x in kv.get("E3", "0 0 0").split()[:3])
        jobs.append({"id": "camco-e3", "repo": "AAAI2027-4", "title": "CaMCo E3 梯度范数归一化 λ（9 个终格）", "box": h, "cards": [],
                     "kind": "gen", "status": "running" if ch else "done", "progress": {"done": d, "total": 9, "unit": "格"},
                     "detail": "09-27 读出：主判据 FAIL（Qwen2-VL CHAIR_s +4.04 pp，描述变长所致待 E7 确认）" if not ch else "终格训练/评测中", "alerts": []})
    except ValueError: pass
    try:
        n, run, fl = (int(x) for x in kv.get("E4", "0 0 0").split()[:3])
        jobs.append({"id": "camco-e4", "repo": "AAAI2027-4", "title": "CaMCo E4 数据规模 400/800/1500（24 个 CHAIR）", "box": h,
                     "cards": [6] if run else [], "kind": "gen", "status": "failed" if fl else ("running" if run or n < 24 else "done"),
                     "progress": {"done": n, "total": 24, "unit": "文件"}, "detail": kv.get("E4LAST", "")[:110], "alerts": []})
    except ValueError: pass
    try:
        dn, run, fl, tot = (int(x or 0) for x in (kv.get("V2", "0 0 0 0").split() + ["0", "0", "0", "0"])[:4])
        if dn or run:
            cards = sorted({int(c) for c in kv.get("V2CARDS", "").split() if c.isdigit()})
            jobs.append({"id": "camco-v2", "repo": "AAAI2027-4", "title": "CaMCo 大修 E6（CaMCo+PAI 叠加）/ E6-A1（官方 PAI）/ E7（Qwen 控长度留出集）", "box": h,
                         "cards": cards if run else [], "kind": "gen", "status": "failed" if fl else ("running" if run else "done"),
                         "progress": {"done": dn, "total": tot or dn, "unit": "格"},
                         "detail": "E6/E6-A1/E7 已读出（均 FAIL，见 REVISION_V2.md）" if not run and not fl else kv.get("V2LAST", "")[:110], "alerts": []})
    except ValueError: pass
    return jobs

def cvpr3_job():
    # 2026-09-27: after the K400 download the chain trains six runs (8-card DDP) and then 36 evals; show the chain's last
    # log line and the current run's step and ETA from its training log.
    h, R = "3090", "/data/xyf/CVPR2027-1"
    out = ssh(h, f"tail -1 {R}/logs/adapt_chain.log; ls -t {R}/logs/adapt_train_*.log 2>/dev/null | head -1; "
                 f"f=$(ls -t {R}/logs/adapt_train_*.log 2>/dev/null | head -1); [ -n \"$f\" ] && tail -c 600 $f | tr '\\r' '\\n' | grep step | tail -1; "
                 f"echo @alive=$(pgrep -fc '^bash scripts/(chain_adapt|chain_resume_supervisor).sh'); ls {R}/stage/adapt/train_*.done 2>/dev/null | wc -l")
    if not out: return None
    lines = out.strip().splitlines()
    alive = next((l.split("=", 1)[1] for l in lines if l.startswith("@alive=")), "0")
    ndone = int(lines[-1]) if lines[-1].strip().isdigit() else 0
    last = lines[0].strip() if lines else ""
    step = next((l for l in lines if " step " in l), "")
    m = re.search(r"step (\d+)/(\d+).*eta (\d+) min", step)
    run = next((l.rsplit("adapt_train_", 1)[1].replace(".log", "") for l in lines if "adapt_train_" in l), "")
    det = f"训练 run {ndone}/6 已完成;当前 {run}" + (f" 第 {m.group(1)}/{m.group(2)} 步,约 {m.group(3)} 分钟" if m else "") + f";链日志:{last[-80:]}"
    return {"id": "cvpr3-adapt", "repo": "CVPR2027-1", "title": "CVPR-3 P-3-5 适配腿(6 个 8 卡训练 run → 36 评测)", "box": h,
            "cards": [], "kind": "gen", "status": "running" if alive != "0" else "done",
            "progress": {"done": ndone, "total": 6, "unit": "run"}, "detail": det,
            "alerts": [] if alive != "0" or ndone == 6 else ["链与监护脚本都不在跑"]}


def pheromones_job():
    h, R = "A800", "/data0/xyf/AAAI2027-7/results/resubmit"
    out = ssh(h, f"for e in e1 e2 e3 e1ra; do printf '%s ' $(cat {R}/${{e}}_runs.jsonl 2>/dev/null | wc -l); done; ps -eo args | grep -cE 'run_e(1ra)?.py'")
    if not out: return None
    try: e1, e2, e3, e1ra, run = (int(x) for x in out.split()[:5])
    except ValueError: return None
    return {"id": "pheromones-resubmit", "repo": "AAAI2027-7", "title": "Pheromones 转投实验 E1–E4 + E1-RA（A800 CPU，GPU 隐藏）", "box": h, "cards": [],
            "kind": "gen", "status": "running" if run > 1 else "done", "progress": {"done": e1 + e2 + e3 + e1ra, "total": 1890, "unit": "run"},
            "detail": f"E1 {e1}/450 · E2 {e2}/540 · E3 {e3}/270 · E1-RA {e1ra}/630；09-27 已读出，转大修（REVISION_V2）", "alerts": []}

def pheromones_v2_job():
    # 2026-09-27: V2 major revision on the 3090 GPUs (REVISION_V2.md P1/P3; P2's model-free arms ran on the CPU)
    h, R = "3090", "/data/xyf/scratch/pheromones/results/v2"
    out = ssh(h, f"for e in p1 p2 p3; do printf '%s ' $(cat {R}/${{e}}_runs.jsonl 2>/dev/null | wc -l); done; "
                 f"printf '%s ' $(pgrep -fc '^/data/xyf/envs/gavel-3090/bin/python experiments/v2/run_v2.py'); grep -c FAIL {R}/p_launch.log")
    if not out: return None
    try: p1, p2, p3, run, fail = (int(x) for x in out.split()[:5])
    except ValueError: return None
    return {"id": "pheromones-v2", "repo": "AAAI2027-7", "title": "Pheromones 大修 P1 内容消融 / P3 委托曲线（3090 GPU，Qwen2.5-3B）", "box": h, "cards": [],
            "kind": "gen", "status": "running" if run > 0 else "done", "progress": {"done": p1 + p3, "total": 660, "unit": "run"},
            "detail": f"P1 {p1}/600 · P3 {p3}/60 · P2 无模型臂 {p2} 行（CPU 已跑完）；{run} 个分片在跑",
            "alerts": [f"p_launch.log 有 {fail} 行 FAIL"] if fail else []}

SCIENCE_PROBE = inspect.getsource(science_counts) + r'''
import json, os, re
from pathlib import Path
# Operational ledger only. No file under results/_blind is opened.
log = Path('/data0/xyf/science/logs/full_eval.log')
lines = log.read_text().splitlines() if log.exists() else []
completed = science_counts(lines)
consumers = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        if p.stat().st_uid != os.getuid(): continue
        args = p.joinpath('cmdline').read_bytes().split(b'\0')
        if not any(Path(a.decode()).name == 'full_eval.py' for a in args[1:3] if a): continue
        allowed = {}
        for field in p.joinpath('environ').read_bytes().split(b'\0'):
            key, sep, value = field.partition(b'=')
            if key in (b'EVAL_ONLY', b'EVAL_CARDS'): allowed[key.decode()] = value.decode()
        consumers.append(allowed)
    except (OSError, ValueError): pass
out = []
for model in sorted(set(completed) | {m for c in consumers for m in c.get('EVAL_ONLY','').split(',') if m}):
    ok = completed.get(model, 0)
    live = [c for c in consumers if model in c.get('EVAL_ONLY','').split(',')]
    if model != 'Qwen/Qwen2.5-32B' and not live: continue
    out.append(dict(model=model, done=ok, running=bool(live),
                    cards=sorted({int(k) for c in live for k in c.get('EVAL_CARDS','').split(',') if k.isdigit()})))
print(json.dumps(out))
'''


def science_a800_jobs():
    out = ssh("A800", "python3 -c " + shlex.quote(SCIENCE_PROBE))
    try:
        snapshots = json.loads(out)
    except (TypeError, ValueError):
        return []
    jobs = []
    for s in snapshots:
        model = s["model"].rsplit("/", 1)[-1]
        status = "running" if s["running"] else ("done" if s["done"] == 17 else "unknown")
        jobs.append({"id": "science-a800-" + model.lower().replace("qwen2.5-", ""), "repo": "Science",
                     "title": f"Phase-Trans A800：{model} 离散评测", "box": "A800", "cards": s["cards"],
                     "kind": "gen", "status": status, "progress": {"done": s["done"], "total": 17, "unit": "任务"},
                     "detail": "盲态：只读状态台账和进程配置，不读指标；" + ("进程在线" if s["running"] else "17 项任务 rc=0" if status == "done" else "未确认在跑"),
                     "alerts": []})
    return jobs

def pheromones_v3_job():
    # 2026-09-27: V3 relative-direction pheromones, 7B fp16 eval on new105 card 1 (REVISION_V3.md; machine change recorded there)
    h, R = "new105", "/home/xyf/AAAI2027-7/results/v3"
    out = ssh(h, f"wc -l < {R}/v3_runs.jsonl; ps -eo args | grep -c '^[p].*run_v3.py'; tail -n 1 {R}/eval.log")
    if not out: return None
    l = out.strip().splitlines()
    try: rows, run = int(l[0]), int(l[1])
    except (ValueError, IndexError): return None
    return {"id": "pheromones-v3", "repo": "AAAI2027-7", "title": "Pheromones V3 相对方向信息素：7B 评测 600 局（new105 卡 1）", "box": h,
            "cards": [], "kind": "gen", "status": "done",
            "progress": {"done": 600, "total": 600, "unit": "局"}, "detail": "09-27 23:39 LA 读出：V3-P1/P2 均 FAIL，走路线 B（REVISION_V3.md）",
            "alerts": []}

def pheromones_d1_job():
    # 2026-09-28: D1 (STAY removed) on new105 card 1, REVISION_V3.md; the chain holds /home/xyf/.gpu1_d1.lock while it runs
    h, R = "new105", "/home/xyf/AAAI2027-7/results/v3"
    out = ssh(h, f"grep -c '\"arm\": \"rel.*_ns\"' {R}/v3_runs.jsonl; (test -f /home/xyf/.gpu1_d1.lock && kill -0 $(cat /home/xyf/.gpu1_d1.lock) 2>/dev/null) && echo up || echo down; tail -n 1 {R}/d1.log")
    if not out: return None
    l = out.strip().splitlines()
    try: rows, up = int(l[0]), l[1] == "up"
    except (ValueError, IndexError): return None
    last = l[2] if len(l) > 2 else ""
    stop = "STOP" in last or "FAIL" in last
    return {"id": "pheromones-d1", "repo": "AAAI2027-7", "title": "Pheromones D1 禁 STAY 诊断：G4 解码门 + 7B 400 局（new105 卡 1）", "box": h,
            # the GPU-1 lock file is reused by the DCAS early-7B chain, so a held lock means D1 only while rows < 400
            "cards": [1] if up and rows < 400 else [], "kind": "gen", "status": "done" if rows >= 400 else ("running" if up else ("failed" if stop else "pending")),
            "progress": {"done": rows, "total": 400, "unit": "局"}, "detail": last[:110], "alerts": [last[:110]] if stop else []}

def dcas_e1_job():
    # 2026-09-28: DCAS E1 on new105 card 1 (PREREG_E1 placement amendment): setup retry -> gate (waits for the D1 lock) -> two workers,
    # 9 dumps then 18 judge runs
    h, R = "new105", "/home/xyf/AAAI2027-1/results/e1"
    out = ssh(h, f"ls {R}/state/*.done 2>/dev/null | wc -l; ls {R}/state/*.fail 2>/dev/null | wc -l; pgrep -u xyf -fc 'run_e1_new105.sh|e1_gate_new105.sh|dcas_e1_setup'; tail -n 1 {R}/logs/e1.log 2>/dev/null || tail -n 1 {R}/logs/gate.log")
    if not out: return None
    l = out.strip().splitlines()
    try: done, fail, procs = int(l[0]), int(l[1]), int(l[2])
    except (ValueError, IndexError): return None
    last = l[3] if len(l) > 3 else ""
    status = "failed" if fail or ("STOP" in last and not procs) else ("done" if done >= 27 else ("running" if procs else "pending"))
    if procs and "STOP" in last: last = "装环境重试中（hf-mirror 的 xet 返回 401，改普通 HTTP 重下 Qwen）；之后门控等 D1 释放卡 1"
    return {"id": "dcas-e1", "repo": "AAAI2027-1", "title": "DCAS E1：9 个 dump + 18 个判分（new105 卡 1，装环境后接 D1）", "box": h,
            "cards": [1] if status == "running" else [], "kind": "gen", "status": status,
            "progress": {"done": done, "total": 27, "unit": "步"}, "detail": last[:110], "alerts": [last[:110]] if status == "failed" else []}

def cpu_ctrl_job():
    # 2026-09-28: a private CPU batch on A800 (28 tasks, 4 workers x 8 cores, GPUs hidden). Read through the neutral link
    # ~/fleet_jobs/cpu-ctrl-1; count markers and processes only, never task logs or result files (they hold metrics)
    h = "A800"
    out = ssh(h, "D=~/fleet_jobs/cpu-ctrl-1; R=$(readlink -f $D/../..); P=$D/progress.log; "
                 "wc -l < $D/tasks.txt; ls $D/*.json 2>/dev/null | wc -l; grep -cE ' END .* rc=[1-9]' $P; grep -c ALL_DONE $P; "
                 "date -d \"$(head -1 $P | cut -c1-19)\" +%s; date +%s; "
                 "pids=$(for p in $(pgrep -u xyf python); do [ \"$(readlink /proc/$p/cwd)\" = \"$R\" ] && echo $p; done); "
                 "echo $pids | wc -w; ([ -n \"$pids\" ] && ps -o %cpu= -p $(echo $pids | tr ' ' ,) | awk '{s+=$1} END {print int(s/100+0.5)}') || echo 0; "
                 # mean wall of finished tasks, from TAKE/END pairs
                 "awk '{d=$1; gsub(/-/,\" \",d); t=$2; gsub(/:/,\" \",t); u=mktime(d\" \"t); "
                 "if($4==\"TAKE\") s[$5\" \"$6]=u; else if($4==\"END\") {w+=u-s[$5\" \"$6]; n++}} END {print (n ? int(w/n) : 0)}' $P")
    if not out: return None
    try: total, done, bad, fin, t0, now, procs, cores, wall = (int(x) for x in out.split()[:9])
    except ValueError: return None
    status = "failed" if bad else ("done" if fin else ("running" if procs else "failed"))
    start = dt.datetime.fromtimestamp(t0, LA).strftime("%m-%d %H:%M LA")
    if status == "running":
        # remaining tasks x mean task wall / parallel workers; ignores the progress of tasks now running
        eta = (" · 每格均 " + f"{wall / 60:.0f} 分钟，约 " + dt.datetime.fromtimestamp(now + (total - done) * wall / max(procs, 1), LA).strftime("%m-%d %H:%M LA") + " 跑完") if done and wall else " · 首格未完成，暂无 ETA"
        detail = f"{procs} 个进程 · 约 {cores} 核 · {start} 起跑{eta}"
    else:
        detail = f"{start} 起跑 · " + ("全部完成" if status == "done" else (f"{bad} 格 rc≠0" if bad else "进程已退出但没有 ALL_DONE"))
    return {"id": "cpu-ctrl-1", "repo": "private", "title": "对照实验批 cpu-ctrl-1（A800 CPU，7 组 × 4 种子，GPU 已屏蔽）", "box": h,
            "cards": [], "cpu": f"CPU {cores} 核" if procs else "CPU", "kind": "gen", "status": status,
            "progress": {"done": done, "total": total, "unit": "格"}, "detail": detail, "alerts": [detail] if status == "failed" else []}

def cpu_grid_job(jid, total, title):
    # Private CPU grids on A800 (GPUs hidden). Neutral link ~/fleet_jobs/<jid> -> <run dir>/results/<grid>; counts markers
    # and processes only. Several grids share one run dir, so a process belongs to a grid when its stdout log is inside it.
    h = "A800"
    out = ssh(h, f"D=$(readlink -f ~/fleet_jobs/{jid}); ls $D/*.json 2>/dev/null | wc -l; "
                 "cat $D/*.log 2>/dev/null | grep -c '^FAIL '; "
                 "pids=$(for p in $(pgrep -u xyf python); do case \"$(readlink /proc/$p/fd/1)\" in $D/*) echo $p;; esac; done); "
                 "echo $pids | wc -w; [ -n \"$pids\" ] && ps -o %cpu= -p $(echo $pids | tr ' ' ,) | awk '{s+=$1} END {print int(s/100+0.5)}' || echo 0")
    if not out: return None
    try: done, bad, procs, cores = (int(x) for x in out.split()[:4])
    except ValueError: return None
    status = "failed" if bad else ("done" if done >= total and not procs else ("running" if procs else "pending"))
    detail = (f"{procs} 个进程 · 约 {cores} 核 · 网格 {done}/{total} 格" if procs else
              ("全部完成" if status == "done" else (f"{bad} 格 FAIL" if bad else f"网格 {done}/{total}，等 A800 CPU 空出后续跑")))
    return {"id": jid, "repo": "private", "title": title, "box": h,
            "cards": [], "cpu": f"CPU {cores} 核" if procs else "CPU", "kind": "gen", "status": status,
            "progress": {"done": done, "total": total, "unit": "格"}, "detail": detail, "alerts": [detail] if status == "failed" else []}

def cpu_ctrl2_job():
    return cpu_grid_job("cpu-ctrl-2", 36, "对照实验批 cpu-ctrl-2（A800 CPU，36 格网格，GPU 已屏蔽）")

def cpu_ctrl3_job():
    return cpu_grid_job("cpu-ctrl-3", 24, "对照实验批 cpu-ctrl-3（A800 CPU，24 格网格，GPU 已屏蔽）")

def sysload(host):
    # CPU side of each box, so GPU cards and CPU batches sit in one scheduling view (user 2026-09-28)
    out = ssh(host, "nproc; cut -d' ' -f1 /proc/loadavg; free -g | awk '/^Mem:/{print $2, $7}'")
    try:
        cores, load, mt, ma = out.split()[:4]
        return {"cores": int(cores), "load": float(load), "mem_total": int(mt), "mem_avail": int(ma)}
    except (AttributeError, ValueError):
        return None

def status_job(host, path, repo, title, cards):
    out = ssh(host, f"tail -n 40 {path} 2>/dev/null")
    if not out or not out.strip(): return None
    lines = [l for l in out.strip().splitlines() if l.strip()]
    last = lines[-1]
    n_done = sum(1 for l in lines if l.startswith(("SCORED", "FULL ")) and "rc=0" in l or l.startswith("SCORED") or " END " in l and ("rc=0" in l or "greedy=" in l) or l.startswith("END ") and "rc=0" in l
                 or (" wall=" in l and " rc=0 " in l))
    # NOTE/RELAUNCH lines are annotations and may quote a failure; only real status lines count
    fails = [l for l in lines if ("FAILED" in l or "ABORT" in l) and not l.startswith(("NOTE", "RELAUNCH", "LANEB_DEFERRED", "DRIVER_STOPPED"))]
    status = "running"
    # 2026-09-25: terminal lines may carry a timestamp or lane prefix ("[m2w] ... DONE lane=", "... DONE all lanes"),
    # so match DONE as a word anywhere; a non-zero rc on the last line is a failure, not "running"
    if "CHAIN_DONE" in last or re.search(r"\b(DONE|finished)\b", last) or "ALL_DONE" in last or "SMOKE_DONE" in last or "FULL_DONE" in last: status = "done"
    elif fails and (fails[-1] == last): status = "failed"
    elif re.search(r"\brc=[1-9]\d*\b", last) and not last.startswith(("NOTE", "RELAUNCH")): status = "failed"; fails = [last]
    elif last.startswith("DRIVER_STOPPED"): status = "failed"; fails = [last]
    jid = os.path.basename(path)
    if jid == "full_eval.log":
        jid = f"science-{host}"  # three boxes share the basename; history.jsonl keys by id
        # consumers come and go per card set; the card list is the union of every cards=... in the
        # recent tail (FULL_START/SMOKE/FULL lines), not the static list (2026-09-25: 3090 grew 6,7 -> 0..7)
        seen = {int(c) for l in lines for c in re.findall(r"cards=([0-9,]+)", l)[-1:] for c in c.split(",") if c.isdigit()}
        if seen: cards = sorted(seen)
    return {"id": jid, "repo": repo, "title": title, "box": host, "cards": cards, "kind": "gen",
            "status": status, "progress": {"done": n_done, "total": 0, "unit": "步"}, "detail": last[:110],
            "alerts": [f[:110] for f in fails[-1:]] if status == "failed" else []}

def main():
    t0 = time.time()
    curves = json.loads((DATA / "curves.json").read_text()) if (DATA / "curves.json").exists() else {}
    boxes, jobs, alerts = [], [], []
    for name, label in (("A800", "A800 ×4 (80 GB；共享卡按当前进程归属)"), ("3090", "3090 ×8 (24 GB, .110)"),
                        ("fuxin", "fuxin 4090 ×8 (48 GB, 公司)"), ("new105", "new105 4090D ×2 (48 GB, 公司)"), ("194-yyd", "194 4090D ×4 (48 GB, 公司)"),
                        ("4090-jm", "4090-jm（共享；不启动新任务）")):
        cards = gpus(name)
        boxes.append({"name": name, "label": label, "reachable": cards is not None, "cards": cards or [],
                      "sys": sysload(name) if cards is not None else None})
    jobs.extend(camco_jobs())
    active, active_alerts = collect_active_jobs(ssh, {b["name"] for b in boxes if b["reachable"]})
    jobs.extend(active)
    alerts.extend(active_alerts)
    jobs.extend(science_a800_jobs())
    for fn in (cvpr3_job, cpu_ctrl_job, cpu_ctrl2_job, cpu_ctrl3_job, pheromones_d1_job, dcas_e1_job, pheromones_v3_job, pheromones_v2_job, pheromones_job):
        j = fn()
        if j: jobs.append(j)
    # 2026-09-27: the jobs below all finished before 09-26 (WWW stage7, ICLR ladders, Science legs, ChronoCheck);
    # they are kept in the code and shown only with FLEET_LEGACY=1 so the board lists current work
    LEGACY = os.environ.get("FLEET_LEGACY") == "1"
    if LEGACY:
        j = stage7(curves);  jobs.append(j) if j else alerts.append("A800 stage7 状态不可读")
        j = ladder();        jobs.append(j) if j else alerts.append("3090 阶梯状态不可读")
        j = b7tp2();         jobs.append(j) if j else None
        for args in STATUS_JOBS:
            j = status_job(*args)
            if j: jobs.append(j)
    if LEGACY:
        R = "/data/xyf/ICLR2027-6/experiments/chronocheck/results/main"
        # fuxin CRITIC originals (rc=137 on 09-17) were resumed and completed on 194; the 194 rows below are the record
        for args in (("194-yyd", "/data/yyd/chronocheck_32b-awq_critic_fuxin_194_status", "/data/yyd/ICLR2027-6/experiments/chronocheck/results/main/32b-awq_critic_fuxin", "CRITIC@32B-AWQ（194 续跑）", [0], 887, ["critic"]),
                     ("194-yyd", "/data/yyd/chronocheck_llama8b_critic_fuxin_194_status", "/data/yyd/ICLR2027-6/experiments/chronocheck/results/main/llama8b_critic_fuxin", "CRITIC@Llama-3.1-8B（194 续跑）", [1], 887, ["critic"]),
                     ("new105", "/home/xyf/logs/chronocheck_chatts14b_status", "/home/xyf/ICLR2027-6/experiments/chronocheck/results/main/chatts14b", "ChatTS-14B 五臂", [1], 887, ["zero_shot", "cot", "chronocheck", "certify_abstain", "repair_gated"])):
            j = chrono(*args)
            if j: jobs.append(j)
    # ownership: a card is "ours" if a running job lists it
    ours = {(j["box"], c) for j in jobs if j["status"] == "running" for c in j["cards"]}
    for b in boxes:
        users = card_users(b["name"]) if b["reachable"] else {}
        mine = OUR_USERS.get(b["name"], set())
        for c in b["cards"]:
            u = users.get(c["idx"], set())
            if (b["name"], c["idx"]) in ours or (u & mine): c["owner"] = "ours"
            elif u or c["mem_used"] > 1500: c["owner"] = "other"
            else: c["owner"] = "free"
    # a box that did not answer this round cannot vouch for its jobs: mark them stale instead of
    # carrying the last known status forward (2026-09-20: unreachable boxes kept showing "running")
    down = {b["name"] for b in boxes if not b["reachable"]}
    for j in jobs:
        if j["box"] in down and j["status"] in ("running", "unknown"):
            j["status"] = "stale"; j["detail"] = f"{j['box']} 本轮未采集到；上次状态：" + j["detail"][:80]
    for name in sorted(down): alerts.append(f"{name} 本轮采集失败：该机任务状态为上次快照，不作数")
    for j in jobs:
        if j["status"] == "failed": alerts.append(f"{j['title']} 失败：{j['detail']}")
        for a in j.get("alerts", []): alerts.append(f"{j['title']}: {a}")
    fleet = {"generated_at": dt.datetime.now(LA).strftime("%Y-%m-%d %H:%M LA"), "boxes": boxes, "jobs": jobs,
             "alerts": alerts, "collect_s": round(time.time() - t0, 1)}
    (DATA / "fleet.json").write_text(json.dumps(fleet, ensure_ascii=False, indent=1))
    (DATA / "curves.json").write_text(json.dumps(curves, ensure_ascii=False))
    hist = DATA / "history.jsonl"
    with hist.open("a") as f:
        f.write(json.dumps({"t": fleet["generated_at"], "cards": {b["name"]: [c["owner"] for c in b["cards"]] for b in boxes},
                            "jobs": {j["id"]: j["progress"] for j in jobs}}, ensure_ascii=False) + "\n")
    if os.environ.get("FLEET_PUSH", "1") == "1":
        subprocess.run(["git", "-C", str(ROOT), "add", "-A"], capture_output=True)
        subprocess.run(["git", "-C", str(ROOT), "commit", "-qm", f"data {fleet['generated_at']}"], capture_output=True)
        [subprocess.run(["git", "-C", str(ROOT), "push", "-q", "origin", "HEAD:main"], capture_output=True, timeout=300) for _ in range(2) if subprocess.run(["git", "-C", str(ROOT), "status", "-sb"], capture_output=True, text=True).stdout.splitlines()[0].find("ahead") >= 0]

if __name__ == "__main__":
    main()
