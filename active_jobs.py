"""Read operational state of the current experiment chains. No result metrics are read."""
import json
import shlex
import inspect


def held_card_locks(identities, lines):
    """Read active kernel flock records; an existing lock file is insufficient."""
    cards = set()
    for line in lines:
        fields = line.split()
        if len(fields) < 6 or '->' in fields or fields[1] != 'FLOCK':
            continue
        try:
            major, minor, inode = fields[5].split(':')
            identity = (int(major, 16), int(minor, 16), int(inode))
        except ValueError:
            continue
        cards.update(identities.get(identity, []))
    return cards


def apply_allocations(boxes, allocations):
    """Keep borrowed idle cards out of our available capacity; retain live ownership."""
    for box in boxes:
        record = allocations.get(box['name'])
        if not record:
            continue
        reserved = set(record['reserved_cards'])
        box['allocation_detail'] = record['detail']
        for card in box['cards']:
            card['borrowed'] = card['idx'] in reserved
            if card['borrowed'] and card['owner'] == 'free':
                card['owner'] = 'reserved'


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


def cache_range_snapshot(root, spec, now, proc_root='/proc'):
    """Read pinned cache metadata and live process identity, never model bytes."""
    import hashlib
    import json
    from pathlib import Path
    root, proc_root = Path(root), Path(proc_root)
    result = dict(id=spec['id'], observed_at=now, controllers=0, compute=0, cards=[],
                  log_bytes=0, log_mtime=0, cache_status='unknown', cache_detail='本轮缓存状态未核实',
                  cache_bytes=0, verified_blobs=0)
    def read(name):
        path=root/name
        return json.loads(path.read_text()) if path.is_file() and not path.is_symlink() else None
    try:
        data=(root/'manifest.json').read_bytes()
        if hashlib.sha256(data).hexdigest()!=spec['manifest_sha256']:raise ValueError('manifest')
        manifest=json.loads(data)
        if len(manifest['files'])!=17 or sum(v['bytes'] for v in manifest['files'].values())!=spec['total_bytes']:
            raise ValueError('counts')
        launch,started,ready,failed=(read(name) for name in ('LAUNCH.json','STARTED.json','READY.json','FAILED.json'))
        if not launch or launch.get('source_commit')!=spec['source_commit']:raise ValueError('launch')
        for receipt in (started,ready,failed):
            if receipt and (receipt.get('source_commit')!=spec['source_commit'] or
                            receipt.get('manifest_sha256')!=spec['manifest_sha256']):raise ValueError('source')
        pid=launch['pid'];stat_path=proc_root/str(pid)/'stat'
        if stat_path.is_file():
            fields=stat_path.read_text().rsplit(')',1)[1].split()
            live=(fields[0]!='Z' and fields[19]==launch['process_start_ticks'] and
                  (proc_root/'sys/kernel/random/boot_id').read_text().strip()==launch['boot_id'])
            result['controllers']=int(live)
        for name,expected in manifest['files'].items():
            dest=root/'hub'/name;partial=dest.with_name(dest.name+'.partial');sizes=[]
            for path in (dest,partial):
                try:size=path.stat().st_size if path.is_file() and not path.is_symlink() else 0
                except OSError:size=0
                if size>expected['bytes']:raise ValueError('oversized')
                sizes.append(size)
            result['cache_bytes']+=max(sizes)
            result['verified_blobs']+=int(sizes[0]==expected['bytes'])
        log=root/'runtime.log'
        if log.is_file():
            st=log.stat();result.update(log_bytes=st.st_size,log_mtime=st.st_mtime)
        if ready and failed:
            result['cache_detail']='发现相互矛盾的终态回执；需核对'
        elif ready:
            archive=Path(ready.get('archive',''))
            reported=ready.get('files',[])
            file_match=(len(reported)==17 and {r['blob']:(r['bytes'],r['sha256']) for r in reported}==
                        {k:(v['bytes'],v['sha256']) for k,v in manifest['files'].items()})
            valid=(ready.get('status')=='READY' and ready.get('member_count')==35 and ready.get('blob_count')==17 and
                   ready.get('link_count')==17 and ready.get('all_blob_hashes_verified') is True and
                   ready.get('all_link_targets_verified') is True and ready.get('manifest_bytes_preserved') is True and
                   ready.get('input_bytes')==spec['total_bytes'] and file_match and result['verified_blobs']==17 and
                   archive.is_file() and not archive.is_symlink() and archive.resolve().is_relative_to(root.resolve()) and
                   archive.stat().st_size==ready.get('compressed_bytes') and isinstance(ready.get('compressed_sha256'),str) and
                   len(ready['compressed_sha256'])==64 and all(c in '0123456789abcdef' for c in ready['compressed_sha256']))
            if valid:result.update(cache_status='done',cache_detail='缓存READY回执与文件元数据一致；GPU验证是下一独立步骤')
            else:result['cache_detail']='READY回执或文件元数据不完整；不能确认交付'
        elif failed:
            if failed.get('status')=='FAILED':
                result.update(cache_status='failed',cache_detail='本次缓存恢复失败；部分文件保留，需核对终态回执')
            else:result['cache_detail']='失败文件没有有效终态；需核对'
        elif started and result['controllers']:
            phase='封包/完整校验' if result['cache_bytes']==spec['total_bytes'] else '复制/分段接收'
            result.update(cache_status='running',cache_detail=f"{phase}；{result['verified_blobs']}/17整blob已晋级；未启动GPU")
        elif started:result['cache_detail']='已无匹配的原进程，且没有终态回执；不据旧STARTED确认运行'
    except (OSError,ValueError,TypeError,KeyError,IndexError,AttributeError):
        result.update(cache_status='unknown',cache_detail='缓存身份或元数据校验未通过；本轮状态需核对')
    return result


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
             complete_detail="12 组 CPU 重算均 rc=0；实测宏、五图与26页改写稿已同步（50d51f0）",
             ready_detail="CPU 输入已校验；等待计算进程；论文宏仍待在 WSL 生成"),
        dict(id="certhar-w5-w3b-init", repo="IMWUT2027-1", title="CertHAR W5-A3：15 组配对初始化 CPU 复跑",
             root="/data0/xyf/IMWUT2027-1-w5-cpu-20261002/results/w5/w3b_init_confirmed_20261002",
             code="/data0/xyf/IMWUT2027-1-w5-cpu-20261002", controller="W5_w3b_init_cpu.py",
             logs=".", kind="cpu", phase="phase", mode="cpu_pair_chain", total=15,
             datasets=["uci_har", "hhar", "motionsense", "pamap2", "wisdm"], seeds=[42, 43, 44],
             failure_markers=["CHAIN_FAILED"], complete_detail="15 组 CPU 配对初始化复跑均 rc=0；99 对初始化一致；改写稿已同步（50d51f0）"),
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
        dict(id="cvpr2f", repo="CVPR2027-1", title="CVPR-2f：真实视频确认（7 个模型，8 张 3090）",
             root="/data/xyf/CVPR2027-1/results_2f/_queue", code="/data/xyf/CVPR2027-1", controller="chain_2f.sh",
             logroot="/data/xyf/CVPR2027-1/logs/2f", queue="dev main", done="done", failed="failed", phase="phase", total=175),
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
             hold_markers=["state/LOCK_BUSY", "state/CARD_BUSY"],
             ready_detail="72B 输入与缓存已校验；等待 8 张 3090 同时空闲；算术与同机 goldll 各 1 项",
             hold_detail="启动被共享锁或显存检查拦下；评测尚未开始；等待八张 3090 同时可用"),
        dict(id="cvpr2e", repo="CVPR2027-1", title="CVPR-2e：复制帧代价的规模 / 代 / 家族扫描（CLEVRER，8 个模型，3090）",
             root="/data/xyf/CVPR2027-1/results_2e/_queue", code="/data/xyf/CVPR2027-1", controller="chain_2e.sh",
             logroot="/data/xyf/CVPR2027-1/logs/2e", queue="dev main glmdev glm", done="done", failed="failed", phase="phase", total=224),
        dict(id="certhar-w5-a4", repo="IMWUT2027-1", title="CertHAR W5-A4：LLM 标注与学生同设备成本（4 个数据集，3090）",
             root="/data/xyf/IMWUT2027-1-a4/results/w5/a4", code="/data/xyf/IMWUT2027-1-a4", controller="W5_A4_cost4_3090.sh",
             logs="logs", markers="markers", total=4, terminal="CHAIN_DONE"),
        dict(id="cvpr1-1r-a1-3090", repo="CVPR2027-1", title="CVPR-1 1r 修正案 A1：BGR 输入重跑（3090，W5-A4 之后用卡 0–3）",
             root="/data/xyf/CVPR2027-1-1r/results_1r_a1", code="/data/xyf/CVPR2027-1-1r", controller="launch_1r.sh",
             logs="logs", total=2,
             completion_markers={"G1": "markers/g1.done", "shards": "markers/ALL_SHARDS_EXITED"},
             failure_markers=["markers/g1.FAIL", "markers/shard_1.FAIL", "markers/shard_2.FAIL", "markers/shard_3.FAIL", "markers/shard_4.FAIL"],
             terminal="markers/ALL_SHARDS_EXITED", ready="wait.log",
             ready_detail="等待器在线：W5-A4 写出 CHAIN_DONE 后在空闲的卡 0–3 上启动（一次）",
             gate=dict(file="readout.json", stop_key="stopped", fields_file="g0.json",
                       fields=["protocol_top1_pp", "reference_top1_pp", "tolerance_pp"],
                       stop_detail="运行完成，但复现门 G0 不过（{protocol_top1_pp:.2f}% 对登记参考 {reference_top1_pp}%，"
                                   "门槛 ±{tolerance_pp}）：按登记停止，未做比较"),
             pending_detail="运行完成；读出尚未运行，不能视为通过"),
        dict(id="cvpr2-f7-novideo", repo="CVPR2027-1", title="CVPR-2 F7：无视频基线（3 个模型，3090，排在 1r A1 之后）",
             root="/data/xyf/CVPR2027-1-f7/results_2c_novideo", code="/data/xyf/CVPR2027-1-f7", controller="launch_novideo_3090.sh",
             logs="logs", total=3,
             completion_markers={"qwen25": "markers/qwen25.done", "qwen2": "markers/qwen2.done", "internvl": "markers/internvl.done"},
             failure_markers=["markers/qwen25.FAIL", "markers/qwen2.FAIL", "markers/internvl.FAIL"],
             terminal="markers/ALL_UNITS_EXITED", ready="queue.log",
             ready_detail="排队中：1r A1 两份跑完后依次在卡 2、3 上跑三个模型（每个一次）"),
        dict(id="cvpr2e-scale-a3", repo="CVPR2027-1", title="CVPR-2 规模点（2e 修订 A3：3 个模型，3090）",
             root="/data/xyf/CVPR2027-1-scale-a3", code="/data/xyf/CVPR2027-1-scale-a3/src", controller="chain_scale_3090.sh",
             logroot="/data/xyf/CVPR2027-1-scale-a3/logs", total=3,
             completion_markers={"q25_32b": "state/q25_32b.DONE", "q3_32b": "state/q3_32b.DONE", "ivl3_14b": "state/ivl3_14b.DONE"},
             failure_markers=["state/q25_32b.FAILED", "state/q3_32b.FAILED", "state/ivl3_14b.FAILED", "state/PRECONDITION_FAILED"],
             terminal="state/CHAIN_DONE", ready="logs/fetch.log",
             ready_detail="三个模型权重已下载并核验；等 Science 72B 与 CertHAR W5-A4 让卡后部署"),
    ],
    "fuxin": [
        dict(id="dcas-j6", repo="ACL2027-1", title="DCAS J6：72B 本地判官（修订 6A，fuxin 4 卡）",
             root="/data/xyf/dcas-pilot6/results/e1/pilot6", code="/data/xyf/dcas-pilot6", controller="run_e1_pilot6_fuxin.sh",
             logs="logs", total=1, completion_markers={"G-J6": "MANIFEST_fuxin.sha256"}, terminal="MANIFEST_fuxin.sha256",
             failure_markers=["state/FAILED"], ready="HUB_FILES.json",
             ready_detail="判官权重 47/47 核验通过；等 fuxin 4 张卡各空出 ≥42000 MiB 后手动启动"),
        dict(id="cvpr1-1r", repo="CVPR2027-1", title="CVPR-1 1r：TimeSformer-B SSv2 掩码对比（登记 1r，fuxin 卡 3、4）",
             root="/data/xyf/CVPR2027-1-1r/results_1r", code="/data/xyf/CVPR2027-1-1r", controller="launch_1r.sh",
             logs="logs", total=3,
             completion_markers={"G1": "markers/g1.done", "shard 1/2": "markers/shard_1.done", "shard 2/2": "markers/shard_2.done"},
             failure_markers=["markers/g1.FAIL", "markers/shard_1.FAIL", "markers/shard_2.FAIL"],
             terminal="markers/ALL_SHARDS_EXITED",
             gate=dict(file="readout.json", stop_key="stopped", fields_file="g0.json",
                       fields=["protocol_top1_pp", "reference_top1_pp", "tolerance_pp"],
                       stop_detail="运行完成，但复现门 G0 不过（{protocol_top1_pp:.2f}% 对登记参考 {reference_top1_pp}%，"
                                   "门槛 ±{tolerance_pp}）：按登记停止，未做比较；原因已查明（checkpoint 按 BGR 训练），见修正案 A1"),
             pending_detail="运行完成；读出尚未运行，不能视为通过"),
        dict(id="cvpr1-1r-a1", repo="CVPR2027-1", title="CVPR-1 1r 修正案 A1：BGR 输入重跑（fuxin 卡 3、4）",
             root="/data/xyf/CVPR2027-1-1r/results_1r_a1", code="/data/xyf/CVPR2027-1-1r", controller="launch_1r.sh",
             logs="logs", total=3,
             completion_markers={"G1": "markers/g1.done", "shard 1/2": "markers/shard_1.done", "shard 2/2": "markers/shard_2.done"},
             failure_markers=["markers/g1.FAIL", "markers/shard_1.FAIL", "markers/shard_2.FAIL"],
             terminal="markers/ALL_SHARDS_EXITED",
             gate=dict(file="readout.json", stop_key="stopped", fields_file="g0.json",
                       fields=["protocol_top1_pp", "reference_top1_pp", "tolerance_pp"],
                       stop_detail="运行完成，但复现门 G0 不过（{protocol_top1_pp:.2f}% 对登记参考 {reference_top1_pp}%，"
                                   "门槛 ±{tolerance_pp}）：按登记停止，未做比较"),
             pending_detail="运行完成；读出尚未运行，不能视为通过"),
    ],
    "new105": [
        dict(id="camco-cache-range-r2", repo="AAAI2027-4", title="CaMCo 7B：R2固定缓存分段交付",
             root="/home/xyf/camco7b-range-recovery-r2-20261004T180638Z", mode="cache_range", kind="cpu",
             source_commit="c2865e0668bcc9484cbacfb056c9f3f97cfd8119",
             manifest_sha256="7e133f99bf9906a10071daa5d6af0ebec85e2ac9fe0fbc74251e8e784a56f374", total_bytes=14131147625),
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
        dict(id="camco-artifact-smoke", repo="AAAI2027-4", title="CaMCo 发布代码验证：7B / 13B 原链",
             root="/home/yxy/camco13b-artifact-smoke-20261003",
             code="/home/yxy/camco13b-artifact-smoke-20261003", controller="run_artifact_gpu_smoke_4090jm.sh",
             logs="logs", phase="state/phase", total=2, targets=["7b", "13b"],
             terminal="state/CHAIN_DONE", ready="state/DEPLOYED.json", failure_markers=["state/CHAIN_FAILED"],
             ready_detail="发布代码检查已部署；原 7B 权重缓存缺失，失败记录保留"),
        dict(id="camco-artifact-smoke-13b", repo="AAAI2027-4", title="CaMCo 发布代码验证：独立 13B",
             root="/home/yxy/camco13b-artifact-smoke-13b-20261003",
             code="/home/yxy/camco13b-artifact-smoke-13b-20261003", controller="run_artifact_gpu_smoke_13b_4090jm.sh",
             logs="logs", phase="state/phase", total=1, targets=["13b"], terminal="state/CHAIN_DONE",
             ready="state/DEPLOYED.json", failure_markers=["state/CHAIN_FAILED"],
             hold_markers=["state/LOCK_BUSY", "state/CARD_BUSY"],
             ready_detail="独立 13B 发布代码检查已部署；不计入论文效果指标",
             hold_detail="发布代码检查被共享锁或显存检查拦下；尚未开始"),
    ],
}

# Sent through one read-only SSH command per box. Read only our process identities,
# CUDA_VISIBLE_DEVICES, marker names, file metadata, phase words and runtime JSON. Never read
# task outputs, scores, checkpoints, arbitrary environments or unregistered CPU jobs.
REMOTE_PROBE = inspect.getsource(compute_roots) + inspect.getsource(device_chain_state) + inspect.getsource(cache_range_snapshot) + r'''
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
    if s.get('mode') == 'cache_range':
        out.append(cache_range_snapshot(root,s,now))
        continue
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
    holds = [path for path in s.get('hold_markers', []) if (root/path).exists()]
    interrupted = any(name.startswith('INTERRUPTED_') for name in failures)
    terminal = device_terminal if device_terminal is not None else ((root/s['terminal']).exists() if s.get('terminal') else phase == 'done')
    timed_out = any(logroot.glob('TIMEOUT*'))
    gate = s.get('gate'); gate_read = gate_stopped = None; gate_fields = {}
    if gate:
        try:
            g = json.loads((root/gate['file']).read_text())
            gate_read = True
            gate_stopped = g.get(gate['stop_key']) or None
            if gate.get('fields_file'):
                f = json.loads((root/gate['fields_file']).read_text())
                gate_fields = {k: f[k] for k in gate.get('fields', []) if k in f}
        except (OSError, ValueError, TypeError, AttributeError):
            gate_read = False
    out.append(dict(id=s['id'], done=len(completed), failed=len(failures), phase=phase,
                    terminal=terminal, timeout=timed_out, controllers=live, compute=compute, interrupted=interrupted, holds=holds,
                    cards=sorted(cards), log_bytes=log_bytes, log_mtime=log_mtime,
                    ready=(root/s['ready']).is_file() if s.get('ready') else False,
                    observed_at=now, cost_recorded=(root/s['cost_gate']).exists() if s.get('cost_gate') else None,
                    gate_read=gate_read, gate_stopped=gate_stopped, gate_fields=gate_fields))
print(json.dumps(out))
'''


def job_from_snapshot(spec, snap, host):
    if spec.get('mode') == 'cache_range':
        return dict(id=spec['id'],repo=spec['repo'],title=spec['title'],box=host,cards=[],kind='cpu',
                    status=snap['cache_status'],progress=dict(done=round(snap['cache_bytes']/1048576,1),
                    total=round(spec['total_bytes']/1048576,1),unit='MiB'),detail=snap['cache_detail'],alerts=[],
                    runtime={k:snap[k] for k in ('observed_at','controllers','compute','log_bytes','log_mtime')})
    phase, complete = snap.get("phase"), snap["done"]
    failed = snap["failed"] or snap.get("timeout") or phase == "stop"
    if failed:
        status = "failed"
        detail = f"失败标记 {snap['failed']}；阶段 {phase or '未写入'}；需核对链日志"
        if spec['id'] == 'camco-e12-train' and snap.get('interrupted'):
            detail = "原训练链中断；8 个训练完成，最后种子迁移至 new105 并按修正案重跑完成，见 camco-e12-recovery"
        elif spec['id'] == 'camco-artifact-smoke':
            detail = "原 7B 权重缓存缺失，该链未跑完；13B 见独立验证 camco-artifact-smoke-13b"
    elif snap["terminal"]:
        # W5's CHAIN_DONE means workers have drained even if a job failed.
        # All planned successful units are required before claiming completion.
        status = "done" if complete == spec["total"] else "unknown"
        detail = "全部完成" if status == "done" else "终止标记与完成数不一致；需核对"
        if spec.get("mode") == "device_chain" and status == "done":
            detail = "3 个固定种子均 rc=0；CUDA 运行记录齐全；W5 汇总与改写稿已同步（50d51f0）"
        elif spec.get("kind") == "cpu" and status == "done":
            detail = spec.get("complete_detail", "12 组 CPU 重算均 rc=0；结果已落地，WSL 论文宏仍待完成")
        if spec.get("gate") and status == "done":
            # A finished run is not a passed experiment: the registered read-out decides.
            if snap.get("gate_stopped"):
                status = "failed"
                try:
                    detail = spec["gate"]["stop_detail"].format(**snap.get("gate_fields", {}))
                except (KeyError, ValueError, IndexError):
                    detail = "运行完成，但登记的门未通过：" + str(snap["gate_stopped"])
            elif not snap.get("gate_read"):
                status = "waiting"
                detail = spec.get("pending_detail", "运行完成；读出尚未运行")
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
        detail = spec.get("hold_detail", spec["ready_detail"]) if snap.get("holds") else spec["ready_detail"]
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
