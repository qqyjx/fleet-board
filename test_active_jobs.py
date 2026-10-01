import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from active_jobs import REMOTE_PROBE, SPECS, compute_roots, science_counts, job_from_snapshot


class ActiveJobsTests(unittest.TestCase):
    def snap(self, **kw):
        return dict(done=0, failed=0, phase="full", terminal=False, timeout=False,
                    controllers=1, compute=0, cards=[], observed_at=1,
                    log_bytes=0, log_mtime=0, **kw)

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

    def test_science_excludes_goldll_and_duplicate_attempts(self):
        rows = ['t FULL Qwen/Qwen2.5-32B gsm8k rc=0 wall=1s ',
                't FULL Qwen/Qwen2.5-32B gsm8k rc=0 wall=2s ',
                't FULL Qwen/Qwen2.5-32B gsm8k_goldll rc=0 wall=3s ',
                't FULL Qwen/Qwen2.5-32B mbpp_all964 rc=1 wall=4s ']
        self.assertEqual(science_counts(rows), {'Qwen/Qwen2.5-32B': 1})


if __name__ == "__main__":
    unittest.main()
