import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from active_jobs import (REMOTE_PROBE, SPECS, compute_roots, science_counts,
                         science_snapshots, device_chain_state, job_from_snapshot)


class ActiveJobsTests(unittest.TestCase):
    def snap(self, **kw):
        return dict(done=0, failed=0, phase="full", terminal=False, timeout=False,
                    controllers=1, compute=0, cards=[], observed_at=1,
                    log_bytes=0, log_mtime=0, **kw)

    def test_inventory_ready_before_queue_is_created(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            inventory = base / "inventory.json"
            inventory.write_text("not read: metadata only")
            spec = dict(next(s for s in SPECS["A800"] if s["id"] == "cvpr2e"))
            spec.update(root=str(base / "missing_queue"), ready=str(inventory),
                        code=str(base), logroot=str(base / "logs"))
            raw = subprocess.check_output([sys.executable, "-c", "specs = " + repr([spec]) + "\n" + REMOTE_PROBE], text=True)
            snap = json.loads(raw)
            self.assertEqual(len(snap), 1)
            job = job_from_snapshot(spec, snap[0], "A800")
            self.assertEqual(job["status"], "waiting")
            self.assertEqual(job["progress"], dict(done=0, total=224, unit="作业"))
            self.assertEqual(job["cards"], [])

    def test_live_controller_does_not_claim_gpu_work(self):
        s = self.snap()
        j = job_from_snapshot(SPECS["3090"][0], s, "3090")
        self.assertEqual(j["status"], "waiting")
        self.assertEqual(j["cards"], [])

    def test_missing_process_does_not_claim_running_from_phase(self):
        s = self.snap()
        s["controllers"] = 0
        self.assertEqual(job_from_snapshot(SPECS["3090"][0], s, "3090")["status"], "unknown")

    def test_failed_w5_is_not_done_even_with_terminal_marker(self):
        s = self.snap()
        s.update(done=45, failed=1, terminal=True)
        self.assertEqual(job_from_snapshot(SPECS["3090"][1], s, "3090")["status"], "failed")

    def test_incomplete_terminal_marker_is_not_completion(self):
        s = self.snap()
        s.update(done=44, terminal=True)
        self.assertEqual(job_from_snapshot(SPECS["3090"][0], s, "3090")["status"], "unknown")

    def test_all_gavel_units_with_terminal_marker_is_complete(self):
        s = self.snap()
        s.update(done=45, terminal=True)
        self.assertEqual(job_from_snapshot(SPECS["3090"][0], s, "3090")["status"], "done")

    def test_remote_probe_counts_only_full_queue_markers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for d in ("queue/full", "done", "failed", "logs"):
                (root/d).mkdir(parents=True)
            for f in ("full1", "full2"):
                (root/"queue/full"/f).touch()
            for f in ("full1", "smoke1", "smoke2"):
                (root/"done"/f).touch()
            (root/"phase").write_text("full\n")
            spec = dict(SPECS["3090"][0], root=td)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(REMOTE_PROBE, {"specs": [spec]})
            snap = json.loads(buf.getvalue())[0]
            self.assertEqual(snap["done"], 1)
            self.assertFalse(snap["terminal"])

    def test_dataloader_forks_are_not_counted_as_runs(self):
        candidates = [(10, 1, '/log/a', [0]), (11, 10, '/log/a', [0]),
                      (12, 11, '/log/a', [0]), (20, 1, '/log/b', [6])]
        self.assertEqual([r[0] for r in compute_roots(candidates)], [10, 20])

    def test_interrupted_training_preserves_completed_count(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'state').mkdir()
            spec = dict(SPECS['4090-jm'][0], root=td)
            for name in spec['targets'][:-1]:
                (root/'state'/(name+'.done')).touch()
            (root/'state'/'INTERRUPTED_plain_lora_seed2_train').touch()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(REMOTE_PROBE, {'specs': [spec]})
            snap = json.loads(buf.getvalue())[0]
            job = job_from_snapshot(spec, snap, '4090-jm')
            self.assertEqual(job['status'], 'failed')
            self.assertEqual(job['progress']['done'], 8)
            self.assertIn('迁移至 new105', job['detail'])

    def test_science_excludes_goldll_and_duplicate_attempts(self):
        rows = ['t FULL Qwen/Qwen2.5-32B gsm8k rc=0 wall=1s ',
                't FULL Qwen/Qwen2.5-32B gsm8k rc=0 wall=2s ',
                't FULL Qwen/Qwen2.5-32B gsm8k_goldll rc=0 wall=3s ',
                't FULL Qwen/Qwen2.5-32B mbpp_all964 rc=1 wall=4s ']
        self.assertEqual(science_counts(rows), {'Qwen/Qwen2.5-32B': 1})

    def test_completed_science_models_remain_visible_without_processes(self):
        rows = science_snapshots({'Qwen/Qwen2.5-14B': 17, 'Qwen/Qwen2.5-32B': 17,
                                  'Qwen/Qwen2.5-72B': 16, 'Qwen/Qwen2.5-7B': 17}, [])
        by_model = {row['model']: row for row in rows}
        self.assertEqual(len(rows), 3)
        self.assertEqual(by_model['Qwen/Qwen2.5-14B']['done'], 17)
        self.assertEqual(by_model['Qwen/Qwen2.5-72B']['done'], 16)
        self.assertTrue(all(not row['running'] and not row['cards'] for row in rows))

    def test_live_science_model_has_only_its_observed_cards(self):
        rows = science_snapshots({}, [{'EVAL_ONLY': 'other/live-model', 'EVAL_CARDS': '3'}])
        live = next(row for row in rows if row['model'] == 'other/live-model')
        self.assertTrue(live['running'])
        self.assertEqual(live['cards'], [3])

    def test_device_chain_needs_all_runtime_records_and_distinct_seeds(self):
        seeds = [42, 43, 44]
        chain = dict(seeds=seeds, status='complete', wrapper_sha256='pin',
                     units=[dict(seed=s, rc=0) for s in seeds])
        runtimes = {s: dict(seed=s, status='complete', wrapper_sha256='pin',
                           cuda_peak_allocated_bytes=100,
                           observations=dict(embed=1, train_head=3, predict=3)) for s in seeds}
        self.assertEqual(device_chain_state(chain, runtimes, seeds), ({'42', '43', '44'}, set(), True))
        del runtimes[44]
        completed, failures, terminal = device_chain_state(chain, runtimes, seeds)
        spec = next(s for s in SPECS['3090'] if s.get('mode') == 'device_chain')
        snap = self.snap()
        snap.update(done=len(completed), failed=len(failures), terminal=terminal, controllers=0)
        self.assertEqual(job_from_snapshot(spec, snap, '3090')['status'], 'unknown')
        chain['units'][-1] = dict(seed=43, rc=0)
        self.assertEqual(device_chain_state(chain, runtimes, seeds)[0], set())

    def test_verified_migration_waits_without_claiming_gpu_use(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'state').mkdir()
            (root/'state/INPUTS_VERIFIED.json').write_text('{"status":"verified"}')
            spec = dict(next(s for s in SPECS['3090'] if s['id'] == 'science-72b-migration'), root=td)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(REMOTE_PROBE, {'specs': [spec]})
            snap = json.loads(buf.getvalue())[0]
            job = job_from_snapshot(spec, snap, '3090')
            self.assertEqual(job['status'], 'waiting')
            self.assertEqual(job['cards'], [])
            self.assertEqual(job['progress']['done'], 0)
            (root/'state/ARITHMETIC_DONE').write_text('done')
            (root/'state/CHAIN_DONE').write_text('done')
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(REMOTE_PROBE, {'specs': [spec]})
            snap = json.loads(buf.getvalue())[0]
            self.assertEqual(job_from_snapshot(spec, snap, '3090')['status'], 'unknown')

    def test_registered_cpu_chain_runs_without_claiming_cards(self):
        spec = SPECS['A800'][0]
        snap = self.snap()
        snap.update(compute=1, phase='W3b')
        job = job_from_snapshot(spec, snap, 'A800')
        self.assertEqual(job['status'], 'running')
        self.assertEqual(job['kind'], 'cpu')
        self.assertEqual(job['cards'], [])

    def test_cpu_terminal_needs_all_twelve_successful_groups(self):
        spec = SPECS['A800'][0]
        snap = self.snap()
        snap.update(done=11, terminal=True)
        self.assertEqual(job_from_snapshot(spec, snap, 'A800')['status'], 'unknown')
        snap['done'] = 12
        job = job_from_snapshot(spec, snap, 'A800')
        self.assertEqual(job['status'], 'done')
        self.assertIn('论文宏仍待完成', job['detail'])

    def test_remote_probe_observes_cpu_only_in_registered_log_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'logs').mkdir()
            spec = dict(SPECS['A800'][0], root=td, code=td)
            with (root/'logs/compute.log').open('w') as output:
                child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'],
                                         stdout=output, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
                try:
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        exec(REMOTE_PROBE, {'specs': [spec, dict(spec, id='unregistered', kind='gen')]})
                    cpu, other = json.loads(buf.getvalue())
                    self.assertEqual(cpu['compute'], 1)
                    self.assertEqual(cpu['cards'], [])
                    self.assertEqual(other['compute'], 0)
                finally:
                    child.terminate()
                    child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
