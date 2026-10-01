"""Read operational state of the current experiment chains. No result metrics are read."""
import json
import shlex
import inspect


def compute_roots(candidates):
    """A DataLoader child shares its parent's stdout and is not a second run."""
    ids = {pid for pid, ppid, stdout, cards in candidates}
    roots = [row for row in candidates if row[1] not in ids]
    return roots


def science_counts(lines):
    """Count distinct successful discrete cells in the operational ledger."""
    import re
    completed = {}
    for line in lines:
        m = re.search(r'\bFULL (\S+) (\S+) rc=(\d+) ', line)
        if m and not m[2].endswith('_goldll'):
            completed[(m[1], m[2])] = int(m[3])
    return {model: sum(rc == 0 for (m, task), rc in completed.items() if m == model)
            for model, task in completed}


SPECS = {
    "3090": [
        dict(id="gavel-opera", repo="AAAI2027-5", title="GAVEL OPERA 全量对照", root="/data/xyf/scratch/gavel/opera/run3090",
             code="/data/xyf/scratch/gavel/AAAI2027-5", controller="opera_chain_3090.sh", logs="logs",
             queue="queue/full", done="done", failed="failed", phase="phase", total=45),
        dict(id="certhar-w5", repo="IMWUT2027-1", title="CertHAR W5：HHAR 修复依赖重跑", root="/data/xyf/IMWUT2027-1-w5/results/w5/chain",
             code="/data/xyf/IMWUT2027-1-w5", controller="W5_chain_3090.sh", logs="logs",
             markers="markers", total=46, terminal="CHAIN_DONE"),
        dict(id="cvpr2c", repo="CVPR2027-1", title="CVPR-2c：时间组内运动状态", root="/data/xyf/CVPR2027-1/results_2c/_queue",
             code="/data/xyf/CVPR2027-1", controller="chain_2c.sh", logroot="/data/xyf/CVPR2027-1/logs/2c",
             queue="dev main", done="done", failed="failed", phase="phase", total=307),
        dict(id="camco-e13", repo="AAAI2027-4", title="CaMCo E13：Qwen2-VL 第二模型家族", root="/data/xyf/scratch/camco/e13",
             code="/data/xyf/scratch/camco/AAAI2027-4", controller="chain_e13.sh", logs="logs",
             queue="queue/units", done="queue/done", failed="queue/failed", phase="queue/phase", total=33,
             terminal="state/E13_DONE", cost_gate="state/cost_recorded"),
        dict(id="camco-e14", repo="AAAI2027-4", title="CaMCo E14：匹配 No 目标数量的对照", root="/data/xyf/scratch/camco/e13/e14",
             code="/data/xyf/scratch/camco/e13/e14/code", controller="chain_e14.sh", logs="logs",
             queue="queue/units", done="queue/done", failed="queue/failed", phase="queue/phase", total=7,
             terminal="state/E14_DONE", cost_gate="state/cost_recorded"),
    ],
    "new105": [
        dict(id="camco-e12-eval", repo="AAAI2027-4", title="CaMCo E12：13B 留出模型评测", root="/home/xyf/e12/e12",
             code="/home/xyf/e12", controller="run_e12_new105.sh", logs="logs", total=20, terminal="state/E12_DONE",
             targets=["chair_vanilla", "pope_vanilla"] + [f"{metric}_{arm}_seed{s}" for arm in ("random", "cem", "plain_lora")
                        for s in range(3) for metric in ("chair", "pope")]),
        dict(id="camco-e12-recovery", repo="AAAI2027-4", title="CaMCo E12：最后种子迁移重跑", root="/home/xyf/e12/e12",
             code="/home/xyf/e12/e12/recovery_plain_seed2_20261001", controller="recover_plain_seed2_new105.sh",
             logroot="/home/xyf/e12/e12/recovery_plain_seed2_20261001", total=1,
             terminal="state/plain_lora_seed2_train_new105.done", targets=["plain_lora_seed2_train_new105"],
             failure_markers=["recovery_plain_seed2_20261001/STOPPED"]),
    ],
    "4090-jm": [
        dict(id="camco-e12-train", repo="AAAI2027-4", title="CaMCo E12：原 13B 训练链", root="/home/yxy/camco13b/e12",
             code="/home/yxy/camco13b", controller="run_e12_4090jm.sh", logs="logs", total=9, terminal="state/E12_4090JM_DONE",
             targets=[f"{arm}_seed{s}_train" for arm in ("random", "cem", "plain_lora") for s in range(3)]),
    ],
}

# Sent through one read-only SSH command per box. Read only our process identities,
# CUDA_VISIBLE_DEVICES, marker names, file metadata and phase words. Never read
# task outputs, scores, checkpoints, arbitrary environments or private CPU jobs.
REMOTE_PROBE = inspect.getsource(compute_roots) + r'''
import json, os, time
from pathlib import Path
now = time.time()
procs = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        if p.stat().st_uid != os.getuid(): continue
        argv = [a for a in (p/'cmdline').read_bytes().decode(errors='replace').split('\0') if a]
        if not argv: continue
        cwd = os.readlink(p/'cwd')
        stdout = os.readlink(p/'fd/1')
        cards = []
        for entry in (p/'environ').read_bytes().split(b'\0'):
            if entry.startswith(b'CUDA_VISIBLE_DEVICES='):
                cards = [int(c) for c in entry.split(b'=',1)[1].split(b',') if c.isdigit()]
                break
        stat = (p/'stat').read_text().rsplit(')',1)[1].split()
        procs.append((int(p.name), int(stat[1]), argv, cwd, stdout, cards))
    except (OSError, ValueError): pass
def names(p):
    return {x.name for x in p.iterdir() if not x.name.endswith('.tmp')} if p.is_dir() else set()
out = []
for s in specs:
    root = Path(s['root'])
    if not root.exists(): continue
    logroot = Path(s.get('logroot', str(root/s.get('logs','logs'))))
    live, compute, cards, log_bytes, log_mtime = 0, 0, set(), 0, 0
    candidates = []
    for pid, ppid, argv, cwd, stdout, cvd in procs:
        for a in argv[:3]:
            resolved = str(Path(cwd)/a) if not a.startswith('/') else a
            if Path(a).name == s['controller'] and resolved.startswith(s['code']+'/'):
                live += 1
                break
        # Shell workers and memory pollers inherit CUDA settings. Only actual
        # Python compute processes with stdout inside this chain's log dir count.
        if cvd and Path(argv[0]).name.startswith('python') and stdout.startswith(str(logroot)+'/'):
            candidates.append((pid, ppid, stdout, cvd))
    outputs = set()
    for pid, ppid, stdout, cvd in compute_roots(candidates):
        compute += 1
        cards.update(cvd)
        outputs.add(stdout)
    for stdout in outputs:
        try:
            st = Path(stdout).stat()
            log_bytes += st.st_size
            log_mtime = max(log_mtime, st.st_mtime)
        except OSError: pass
    phpath = root/s.get('phase','phase')
    phase = phpath.read_text().strip() if phpath.is_file() else None
    expected = set()
    for sub in s.get('queue','').split(): expected.update(names(root/sub))
    if s.get('markers'):
        m = names(root/s['markers'])
        completed = {x[:-5] for x in m if x.endswith('.done')}
        failures = {x[:-7] for x in m if x.endswith('.failed')}
    elif s.get('targets'):
        completed = {x for x in s['targets'] if (root/'state'/(x+'.done')).is_file()}
        failures = {x for x in names(root/'state') if x.startswith(('BLOCKED_', 'STOPPED_', 'INTERRUPTED_', 'INSTRUMENT_VOID'))}
    else:
        completed = names(root/s['done']) & expected
        failures = names(root/s['failed'])
    failures.update(path for path in s.get('failure_markers', []) if (root/path).exists())
    interrupted = any(name.startswith('INTERRUPTED_') for name in failures)
    terminal = (root/s['terminal']).exists() if s.get('terminal') else phase == 'done'
    timed_out = any(logroot.glob('TIMEOUT*'))
    out.append(dict(id=s['id'], done=len(completed), failed=len(failures), phase=phase,
                    terminal=terminal, timeout=timed_out, controllers=live, compute=compute, interrupted=interrupted,
                    cards=sorted(cards), log_bytes=log_bytes, log_mtime=log_mtime,
                    observed_at=now, cost_recorded=(root/s['cost_gate']).exists() if s.get('cost_gate') else None))
print(json.dumps(out))
'''


def job_from_snapshot(spec, snap, host):
    phase, complete = snap.get("phase"), snap["done"]
    failed = snap["failed"] or snap.get("timeout") or phase == "stop"
    if failed:
        status = "failed"
        detail = f"失败标记 {snap['failed']}；阶段 {phase or '未写入'}；需核对链日志"
        if spec['id'] == 'camco-e12-train' and snap.get('interrupted'):
            detail = "原训练链中断；8 个训练完成，最后种子迁移至 new105"
    elif snap["terminal"]:
        # W5's CHAIN_DONE means workers have drained even if a job failed.
        # All planned successful units are required before claiming completion.
        status = "done" if complete == spec["total"] else "unknown"
        detail = "全部完成" if status == "done" else "终止标记与完成数不一致；需核对"
    elif snap["compute"]:
        status = "running"
        detail = f"{snap['compute']} 个计算进程；阶段 {phase or '作业运行'}"
    elif snap["controllers"]:
        status = "waiting"
        if spec["id"] == "cvpr2c" and phase is None:
            detail = "链在线；等待 GAVEL 队列与 W5 完成"
        elif spec['id'] == 'camco-e12-recovery':
            detail = "恢复链在线；等待前 18 个评测完成及 GPU 1 共享锁"
        elif spec["id"] in ("camco-e13", "camco-e14") and complete == 0:
            detail = "链在线；等待 GAVEL / CVPR-2c 让卡"
        elif complete == spec["total"]:
            detail = "计算单元已完成；等待合并、检查或读出"
        else:
            detail = "链在线；等待依赖、空卡或已登记检查点"
    else:
        status = "unknown"
        detail = "未发现链或计算进程；不能据旧标记确认在跑"
    return dict(id=spec["id"], repo=spec["repo"], title=spec["title"], box=host,
                cards=snap["cards"] if status == "running" else [], kind="gen", status=status,
                progress=dict(done=complete, total=spec["total"], unit="作业"), detail=detail, alerts=[],
                runtime={k: snap[k] for k in ("observed_at", "controllers", "compute", "log_bytes", "log_mtime")})


def collect_active_jobs(ssh, reachable):
    jobs, alerts = [], []
    for host, specs in SPECS.items():
        # An unavailable GPU driver does not imply SSH or the stopped chain's
        # markers are unavailable. Probe these three registered hosts directly.
        code = "specs = " + repr(specs) + "\n" + REMOTE_PROBE
        raw = ssh(host, "python3 -c " + shlex.quote(code), t=35)
        try:
            snapshots = {s["id"]: s for s in json.loads(raw)}
            for spec in specs:
                if spec["id"] in snapshots:
                    jobs.append(job_from_snapshot(spec, snapshots[spec["id"]], host))
                else:
                    alerts.append(f"{spec['title']}：本轮未读到已登记目录")
        except (TypeError, ValueError, KeyError):
            alerts.append(f"{host} 当前实验链状态本轮未读到")
    return jobs, alerts
