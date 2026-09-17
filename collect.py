#!/usr/bin/env python3
"""Fleet & training board collector. Probes the four boxes over ssh (read-only),
writes data/fleet.json (latest) and data/curves.json (training series), then commits
and pushes. Runs from cron every 10 minutes; safe to run by hand."""
import json, os, re, subprocess, time, datetime as dt, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
LA = dt.timezone(dt.timedelta(hours=-7))

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
            cards.append({"idx": int(i), "mem_used": int(u), "mem_total": int(tot), "util": int(ut)})
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
    return {"id": "www-stage7", "repo": "WWW2027-1", "title": "stage7 六臂修正版 GRPO", "box": "A800",
            "cards": [0, 1, 2, 3], "kind": "train", "status": "running" if running else "unknown",
            "progress": {"done": max(0, min(done, total_arms)), "total": total_arms, "unit": "臂"},
            "detail": f"当前臂 {running}" + (f"，step {[a for a in arms if a['arm']==running][0]['steps']}/{FULL}" if running and any(a['arm']==running for a in arms) else ""),
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
    for lane, pat in (("lane1", "ladder_[a-z]*_t*.log"), ("lane2", "ladder_lane2_[a-z]*_t*.log")):
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
            "cards": [0, 1, 6, 7], "kind": "gen", "status": "failed" if fails else "running",
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

def main():
    t0 = time.time()
    curves = json.loads((DATA / "curves.json").read_text()) if (DATA / "curves.json").exists() else {}
    boxes, jobs, alerts = [], [], []
    for name, label in (("A800", "A800 ×4 (80 GB)"), ("3090", "3090 ×8 (24 GB, .110)"),
                        ("fuxin", "fuxin 4090 ×8 (48 GB, 公司)"), ("new105", "new105 4090D ×2 (48 GB, 公司)")):
        cards = gpus(name)
        boxes.append({"name": name, "label": label, "reachable": cards is not None, "cards": cards or []})
    j = stage7(curves);  jobs.append(j) if j else alerts.append("A800 stage7 状态不可读")
    j = ladder();        jobs.append(j) if j else alerts.append("3090 阶梯状态不可读")
    R = "/data/xyf/ICLR2027-6/experiments/chronocheck/results/main"
    for args in (("fuxin", "/data/xyf/chronocheck_32b-awq_critic_fuxin_status", f"{R}/32b-awq_critic_fuxin", "CRITIC@32B-AWQ", [3], 887, ["critic"]),
                 ("fuxin", "/data/xyf/chronocheck_llama8b_critic_fuxin_status", f"{R}/llama8b_critic_fuxin", "CRITIC@Llama-3.1-8B", [4], 887, ["critic"]),
                 ("new105", "/home/xyf/logs/chronocheck_chatts14b_status", "/home/xyf/ICLR2027-6/experiments/chronocheck/results/main/chatts14b", "ChatTS-14B 五臂", [1], 887, ["zero_shot", "cot", "chronocheck", "certify_abstain", "repair_gated"])):
        j = chrono(*args)
        if j: jobs.append(j)
    # ownership: a card is "ours" if a running job lists it
    ours = {(j["box"], c) for j in jobs if j["status"] == "running" for c in j["cards"]}
    for b in boxes:
        for c in b["cards"]:
            c["owner"] = "ours" if (b["name"], c["idx"]) in ours else ("other" if c["mem_used"] > 1500 else "free")
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
        subprocess.run(["git", "-C", str(ROOT), "push", "-q", "origin", "HEAD:main"], capture_output=True, timeout=120)

if __name__ == "__main__":
    main()
