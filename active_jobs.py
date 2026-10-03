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


def science_snapshots(completed, consumers):
    """Keep the current large-model matrix visible after its consumers exit."""
    current = {"Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-72B"}
    live_models = {m for c in consumers for m in c.get("EVAL_ONLY", "").split(",") if m}
    out = []
    for model in sorted(current | live_models):
        live = [c for c in consumers if model in c.get("EVAL_ONLY", "").split(",")]
        out.append(dict(model=model, done=completed.get(model, 0), running=bool(live),
                        cards=sorted({int(k) for c in live for k in c.get("EVAL_CARDS", "").split(",") if k.isdigit()})))
    return out


def device_chain_state(chain, runtimes, seeds):
    """Count confirmed seeds from operational JSON, without opening metrics."""
    units = chain.get("units", [])
    valid = (chain.get("seeds") == seeds and len(units) == len(seeds)
             and {u.get("seed") for u in units} == set(seeds))
    completed = set()
    failures = set()
    for unit in units:
        seed = unit.get("seed")
        if seed not in seeds:
            continue
        if unit.get("rc") not in (None, 0):
            failures.add(str(seed))
        runtime = runtimes.get(seed, {})
        if (valid and unit.get("rc") == 0 and runtime.get("seed") == seed
                and runtime.get("status") == "complete"
                and runtime.get("observations") == {"embed": 1, "train_head": 3, "predict": 3}
                and runtime.get("cuda_peak_allocated_bytes", 0) > 0
                and runtime.get("wrapper_sha256") == chain.get("wrapper_sha256")):
            completed.add(str(seed))
    return completed, failures, chain.get("status") == "complete"


SPECS = {
    "A800": [
        dict(id="certhar-w5-cpu", repo="IMWUT2027-1", title="CertHAR W5：12 组 CPU 派生重算",
             root="/data0/xyf/IMWUT2027-1-w5-cpu-20261002",
             code="/data0/xyf/IMWUT2027-1-w5-cpu-20261002", controller="W5_cpu_recompute.py",
             logs="logs", kind="cpu", phase="state/phase", total=12,
             targets=["E10_w1_probe", "frontier", "anatomy", "E12_l3_classifier", "E7_aggregate",
                      "derived", "b_lanes", "W3a", "W3b", "W4bc", "W4a", "E13_llm_fields"],
             terminal="state/CHAIN_DONE", ready="state/DATA_VERIFIED.json",
             failure_markers=["state/CHAIN_FAILED", "state/DATA_COPY_FAILED"],
             ready_detail="CPU 输入已校验；等待计算进程；论文宏仍待在 WSL 生成"),
        dict(id="certhar-w5-w3b-init", repo="IMWUT2027-1", title="CertHAR W5-A3：15 组配对初始化 CPU 复跑",
             root="/data0/xyf/IMWUT2027-1-w5-cpu-20261002/results/w5/w3b_init_confirmed_20261002",
             code="/data0/xyf/IMWUT2027-1-w5-cpu-20261002", controller="W5_w3b_init_cpu.py",
             logs=".", kind="cpu", phase="phase", mode="cpu_pair_chain", total=15,
             datasets=["uci_har", "hhar", "motionsense", "pamap2", "wisdm"], seeds=[42, 43, 44],
             failure_markers=["CHAIN_FAILED"], complete_detail="15 组 CPU 配对初始化复跑均 rc=0；WSL 验证、宏与正文待完成"),
    ],
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
        dict(id="cvpr2d", repo="CVPR2027-1", title="CVPR-2d：复制帧代价的真实视频确认（FAVOR + MotionBench）",
             root="/data/xyf/CVPR2027-1/results_2d/_queue", code="/data/xyf/CVPR2027-1", controller="chain_2d.sh",
             logroot="/data/xyf/CVPR2027-1/logs/2d", queue="main", done="done", failed="failed", phase="phase", total=96),
        dict(id="cvpr2e", repo="CVPR2027-1", title="CVPR-2e：复制帧代价的规模 / 代 / 家族扫描（CLEVRER，8 个模型）",
             root="/data/xyf/CVPR2027-1/results_2e/_queue", code="/data/xyf/CVPR2027-1", controller="chain_2e.sh",
             logroot="/data/xyf/CVPR2027-1/logs/2e", queue="dev main glmdev glm", done="done", failed="failed", phase="phase", total=224,
             ready="/data/xyf/CVPR2027-1/results_2e/inventory.json",
             ready_detail="CPU 盘点已完成；等待 2d A2 重启后起链，再等 2d 让卡"),
        dict(id="camco-e13", repo="AAAI2027-4", title="CaMCo E13：Qwen2-VL 第二模型家族", root="/data/xyf/scratch/camco/e13",
             code="/data/xyf/scratch/camco/AAAI2027-4", controller="chain_e13.sh", logs="logs",
             queue="queue/units", done="queue/done", failed="queue/failed", phase="queue/phase", total=33,
             terminal="state/E13_DONE", cost_gate="state/cost_recorded"),
        dict(id="camco-e14", repo="AAAI2027-4", title="CaMCo E14：匹配 No 目标数量的对照", root="/data/xyf/scratch/camco/e13/e14",
             code="/data/xyf/scratch/camco/e13/e14/code", controller="chain_e14.sh", logs="logs",
             queue="queue/units", done="queue/done", failed="queue/failed", phase="queue/phase", total=7,
             terminal="state/E14_DONE", cost_gate="state/cost_recorded"),
        dict(id="certhar-w5-e7-device", repo="IMWUT2027-1", title="CertHAR W5 E7：CUDA 设备确认复跑",
             root="/data/xyf/IMWUT2027-1-w5/results/w5/e7_device_confirmed",
             code="/data/xyf/IMWUT2027-1-w5", controller="w5_e7_device.py",
             logroot="/data/xyf/IMWUT2027-1-w5/results/w5/e7_device_confirmed",
             mode="device_chain", seeds=[42, 43, 44], total=3),
        *[dict(id=f"gavel-cost-{ds}", repo="AAAI2027-5", title=f"GAVEL 完整成本补测：{ds}",
               root=f"/data/xyf/scratch/gavel/forward-cost-20261002/{ds}/AAAI2027-5/experiments/gavel",
               code=f"/data/xyf/scratch/gavel/forward-cost-20261002/{ds}/AAAI2027-5/experiments/gavel",
               controller="run_forward_cost_3090.sh", logs="logs", total=4, terminal="state/COST_DONE",
               targets=["s1", "s2", "s3", "s4"],
               failure_markers=[f"state/{stage}.failed" for stage in ("s1", "s2", "s3", "s4")])
          for ds in ("pope", "object_halbench", "mmhal_bench")],
        dict(id="science-72b-migration", repo="science", title="Science 72B：剩余算术与同机 goldll",
             root="/data/xyf/science_72b_3090_20261002", code="/data/xyf/science_72b_3090_20261002",
             controller="run_72b_migration_3090.sh", logs="logs", phase="state/phase", total=2,
             terminal="state/CHAIN_DONE", ready="state/INPUTS_VERIFIED.json",
             completion_markers={"arithmetic": "state/ARITHMETIC_DONE", "goldll": "state/GOLDLL_DONE"},
             failure_markers=["state/CHAIN_FAILED", "state/PLACEMENT_FAILED"],
             ready_detail="72B 输入与缓存已校验；等待 8 张 3090 同时空闲；算术与同机 goldll 各 1 项"),
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
# CUDA_VISIBLE_DEVICES, marker names, file metadata, phase words and runtime JSON. Never read
# task outputs, scores, checkpoints, arbitrary environments or unregistered CPU jobs.
REMOTE_PROBE = inspect.getsource(compute_roots) + inspect.getsource(device_chain_state) + r'''
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
    if not root.exists() and not (s.get('ready') and (root/s['ready']).is_file()): continue
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
        if (cvd or s.get('kind') == 'cpu') and Path(argv[0]).name.startswith('python') and stdout.startswith(str(logroot)+'/'):
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
    device_terminal = None
    if s.get('mode') == 'device_chain':
        try:
            chain = json.loads((root/'chain.json').read_text())
            runtimes = {seed: json.loads((root/f'seed{seed}'/'runtime.json').read_text())
                        for seed in s['seeds'] if (root/f'seed{seed}'/'runtime.json').is_file()}
            completed, failures, device_terminal = device_chain_state(chain, runtimes, s['seeds'])
            phase = chain.get('status')
        except (OSError, ValueError, TypeError, AttributeError):
            completed, failures, device_terminal = set(), set(), False
            phase = '运行记录不可读'
    elif s.get('mode') == 'cpu_pair_chain':
        try:
            chain = json.loads((root/'chain.json').read_text())
            units = chain['units']
            keys = [(u['dataset'], u['seed']) for u in units]
            allowed = {(d, seed) for d in s['datasets'] for seed in s['seeds']}
            if len(keys) != len(set(keys)) or not set(keys) <= allowed:
                raise ValueError('duplicate or unknown CPU work unit')
            completed = {f"{u['dataset']}_seed{u['seed']}" for u in units if u['rc'] == 0}
            failures = {f"{u['dataset']}_seed{u['seed']}" for u in units if u['rc'] != 0}
            if chain['status'] == 'failed': failures.add('chain_failed')
            device_terminal = chain['status'] == 'complete' and (root/'CHAIN_DONE').is_file()
        except (OSError, ValueError, TypeError, KeyError):
            completed, failures, device_terminal = set(), {'unreadable_cpu_chain'}, False
    elif s.get('completion_markers'):
        completed = {unit for unit, path in s['completion_markers'].items() if (root/path).is_file()}
        failures = set()
    elif s.get('markers'):
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
    terminal = device_terminal if device_terminal is not None else ((root/s['terminal']).exists() if s.get('terminal') else phase == 'done')
    timed_out = any(logroot.glob('TIMEOUT*'))
    out.append(dict(id=s['id'], done=len(completed), failed=len(failures), phase=phase,
                    terminal=terminal, timeout=timed_out, controllers=live, compute=compute, interrupted=interrupted,
                    cards=sorted(cards), log_bytes=log_bytes, log_mtime=log_mtime,
                    ready=(root/s['ready']).is_file() if s.get('ready') else False,
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
        if spec.get("mode") == "device_chain" and status == "done":
            detail = "3 个固定种子均 rc=0；CUDA 运行记录齐全；W5 CPU 汇总与论文宏仍待完成"
        elif spec.get("kind") == "cpu" and status == "done":
            detail = spec.get("complete_detail", "12 组 CPU 重算均 rc=0；结果已落地，WSL 论文宏仍待完成")
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
    elif snap.get("ready") and spec.get("ready_detail"):
        status = "waiting"
        detail = spec["ready_detail"]
    else:
        status = "unknown"
        detail = "未发现链或计算进程；不能据旧标记确认在跑"
    return dict(id=spec["id"], repo=spec["repo"], title=spec["title"], box=host,
                cards=snap["cards"] if status == "running" else [], kind=spec.get("kind", "gen"), status=status,
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
