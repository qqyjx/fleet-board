import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from active_jobs import SPECS,cache_range_snapshot,job_from_snapshot


class CacheRangeStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'job';self.root.mkdir();self.proc=Path(self.temp.name)/'proc'
        self.spec=dict(next(s for s in SPECS['new105'] if s['id']=='camco-cache-range-r2'),root=str(self.root),source_commit='fixture')
        files={f'model/blobs/{i}':{'bytes':100,'sha256':hashlib.sha256(bytes([i])*100).hexdigest()} for i in range(17)}
        self.manifest={'files':files};data=json.dumps(self.manifest).encode();(self.root/'manifest.json').write_bytes(data)
        self.spec.update(manifest_sha256=hashlib.sha256(data).hexdigest(),total_bytes=1700)
        self.identity={'source_commit':'fixture','manifest_sha256':self.spec['manifest_sha256']}
        self.write('LAUNCH.json',{'source_commit':'fixture','pid':4242,'process_start_ticks':'123','boot_id':'boot'})
        self.write('STARTED.json',dict(self.identity,status='RUNNING'))
        stat=self.proc/'4242/stat';stat.parent.mkdir(parents=True);fields=['0']*20;fields[0]='S';fields[19]='123'
        stat.write_text('4242 (timeout) '+' '.join(fields))
        boot=self.proc/'sys/kernel/random/boot_id';boot.parent.mkdir(parents=True);boot.write_text('boot\n')

    def write(self,name,value):(self.root/name).write_text(json.dumps(value))
    def snap(self):return cache_range_snapshot(self.root,self.spec,1234,self.proc)
    def make_ready(self):
        for i,(name,value) in enumerate(self.manifest['files'].items()):
            path=self.root/'hub'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(bytes([i])*100)
        archive=self.root/'cache.tar.gz';archive.write_bytes(b'metadata fixture only')
        ready=dict(self.identity,status='READY',member_count=35,blob_count=17,link_count=17,
                   all_blob_hashes_verified=True,all_link_targets_verified=True,manifest_bytes_preserved=True,
                   input_bytes=1700,archive=str(archive),compressed_bytes=archive.stat().st_size,compressed_sha256='a'*64,
                   files=[dict(blob=name,**value) for name,value in self.manifest['files'].items()])
        self.write('READY.json',ready);return ready

    def test_live_identity_reports_transfer_without_gpu_claim(self):
        path=self.root/'hub/model/blobs/0.partial';path.parent.mkdir(parents=True);path.write_bytes(b'x'*30)
        snap=self.snap();job=job_from_snapshot(self.spec,snap,'new105')
        self.assertEqual(snap['cache_status'],'running');self.assertEqual(snap['cache_bytes'],30)
        self.assertEqual(job['cards'],[]);self.assertEqual(job['kind'],'cpu');self.assertEqual(job['progress']['unit'],'MiB')

    def test_stale_pid_or_reboot_never_confirms_running(self):
        (self.proc/'4242/stat').write_text('4242 (timeout) '+' '.join(['S']+['0']*18+['999']))
        self.assertEqual(self.snap()['cache_status'],'unknown')
        (self.proc/'4242/stat').unlink();self.assertEqual(self.snap()['cache_status'],'unknown')

    def test_ready_is_confirmed_from_bound_receipt_and_metadata_without_reading_model(self):
        self.make_ready();(self.proc/'4242/stat').unlink()
        original=Path.read_bytes
        def guarded(path):
            if 'hub' in path.parts:raise AssertionError('Do not read model bytes')
            return original(path)
        with patch.object(Path,'read_bytes',guarded):snap=self.snap()
        self.assertEqual(snap['cache_status'],'done');self.assertEqual(snap['verified_blobs'],17)
        self.assertEqual(snap['cache_bytes'],1700)

    def test_incomplete_or_mismatched_ready_is_not_done(self):
        ready=self.make_ready();ready['files'][0]['sha256']='b'*64;self.write('READY.json',ready)
        self.assertEqual(self.snap()['cache_status'],'unknown')
        ready=self.make_ready();ready['member_count']=34;self.write('READY.json',ready)
        self.assertEqual(self.snap()['cache_status'],'unknown')

    def test_failure_and_contradictory_terminal_are_distinct(self):
        self.write('FAILED.json',dict(self.identity,status='FAILED'))
        self.assertEqual(self.snap()['cache_status'],'failed')
        self.make_ready();self.assertEqual(self.snap()['cache_status'],'unknown')

    def test_full_bytes_with_live_process_remain_verifying_until_ready(self):
        self.make_ready();(self.root/'READY.json').unlink()
        snap=self.snap();self.assertEqual(snap['cache_status'],'running');self.assertIn('完整校验',snap['cache_detail'])

    def test_manifest_identity_change_is_unknown(self):
        (self.root/'manifest.json').write_text('{}')
        self.assertEqual(self.snap()['cache_status'],'unknown')

    def test_complete_and_partial_links_are_not_double_counted(self):
        path=self.root/'hub/model/blobs/0';path.parent.mkdir(parents=True);path.write_bytes(b'x'*100)
        path.with_name(path.name+'.partial').write_bytes(b'x'*100)
        self.assertEqual(self.snap()['cache_bytes'],100)


if __name__=='__main__':unittest.main()
