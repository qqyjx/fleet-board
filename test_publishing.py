import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import publishing as pub


def fixture(stamp='new'):
    return {'generated_at':stamp,'boxes':[{'name':'box','reachable':True,'cards':[{'idx':0,'mem_used':100,'mem_total':1000,'util':10}]}],
            'jobs':[{'id':'job','status':'running','progress':{'done':1,'total':2}}]}


class PublishingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.cache=self.base/'cache';self.cache.mkdir()
        self.write_cache(fixture())

    def write_cache(self,fleet):
        (self.cache/'fleet.json').write_text(json.dumps(fleet))
        (self.cache/'curves.json').write_text('{}')
        (self.cache/'history.jsonl').write_text(json.dumps({'t':fleet['generated_at']})+'\n')

    def repository(self):
        origin=self.base/'origin.git';root=self.base/'repo'
        subprocess.run(['git','init','--bare',str(origin)],check=True,capture_output=True)
        subprocess.run(['git','clone',str(origin),str(root)],check=True,capture_output=True)
        pub.git(root,'config','user.name','Fixture')
        pub.git(root,'config','user.email','fixture@example.invalid')
        pub.git(root,'config','core.fileMode','false')
        pub.git(root,'checkout','-b','main')
        (root/'data').mkdir();old=fixture('old')
        for name,content in [('fleet.json',json.dumps(old)),('curves.json','{}'),('history.jsonl',json.dumps({'t':'old'})+'\n')]:
            (root/'data'/name).write_text(content)
        (root/'source.py').write_text('original source\n')
        pub.git(root,'add','data','source.py');pub.git(root,'-c','core.hooksPath=/dev/null','commit','-m','base')
        pub.git(root,'push','-u','origin','main')
        base=pub.git(root,'rev-parse','HEAD')
        (self.cache/'history.jsonl').write_bytes((root/'data/history.jsonl').read_bytes()+json.dumps({'t':'new'}).encode()+b'\n')
        return root,base

    def test_valid_snapshot(self):
        payload,summary=pub.validate_snapshot(self.cache)
        self.assertEqual(set(payload),set(pub.FILES));self.assertEqual(summary['jobs'],1)

    def test_duplicate_jobs_rejected(self):
        data=fixture();data['jobs'].append(copy.deepcopy(data['jobs'][0]));self.write_cache(data)
        with self.assertRaises(pub.PublicationError):pub.validate_snapshot(self.cache)

    def test_duplicate_machines_rejected(self):
        data=fixture();data['boxes'].append(copy.deepcopy(data['boxes'][0]));self.write_cache(data)
        with self.assertRaises(pub.PublicationError):pub.validate_snapshot(self.cache)

    def test_cloud_history_advance_refreshes_published_curves(self):
        root=self.base/'published';(root/'data').mkdir(parents=True)
        for name in pub.FILES:shutil.copyfile(self.cache/name,root/'data'/name)
        with (root/'data/history.jsonl').open('a') as stream:stream.write('{"t":"cloud-newer"}\n')
        (root/'data/curves.json').write_text('{"new_curve":[]}')
        pub.initialize_cache(root,self.cache)
        self.assertEqual((self.cache/'curves.json').read_text(),'{"new_curve":[]}')
        self.assertEqual((self.cache/'history.jsonl').read_bytes(),(root/'data/history.jsonl').read_bytes())

    def test_out_of_range_gpu_rejected(self):
        data=fixture();data['boxes'][0]['cards'][0]['util']=101;self.write_cache(data)
        with self.assertRaises(pub.PublicationError):pub.validate_snapshot(self.cache)

    def test_nonfinite_json_rejected(self):
        (self.cache/'curves.json').write_text('{"loss":NaN}')
        with self.assertRaises(pub.PublicationError):pub.validate_snapshot(self.cache)

    def test_history_timestamp_mismatch_rejected(self):
        (self.cache/'history.jsonl').write_text('{"t":"old"}\n')
        with self.assertRaises(pub.PublicationError):pub.validate_snapshot(self.cache)

    def test_symlink_snapshot_rejected(self):
        target=self.cache/'fleet.json';target.unlink();target.symlink_to(self.cache/'curves.json')
        with self.assertRaises(pub.PublicationError):pub.validate_snapshot(self.cache)

    def test_git_transaction_preserves_checkout_and_index(self):
        root,base=self.repository();payload,_=pub.validate_snapshot(self.cache)
        before=pub.git(root,'write-tree');commit,changed=pub.build_commit(root,self.cache,base,payload)
        self.assertEqual(set(changed),{'data/fleet.json','data/history.jsonl'})
        self.assertEqual(pub.git(root,'rev-parse','HEAD'),base)
        self.assertEqual(pub.git(root,'write-tree'),before)
        self.assertEqual(pub.git(root,'status','--porcelain'),'')
        self.assertEqual(pub.git(root,'show',commit+':source.py'),'original source')

    def test_history_overwrite_and_extra_file_rejected(self):
        root,base=self.repository();payload,_=pub.validate_snapshot(self.cache)
        payload['history.jsonl']=b'{"t":"new"}\n'
        with self.assertRaises(pub.PublicationError):pub.build_commit(root,self.cache,base,payload)
        payload['secret.txt']=b'unrelated'
        with self.assertRaises(pub.PublicationError):pub.build_commit(root,self.cache,base,payload)

    def test_dirty_checkout_and_existing_pending_block_publication(self):
        root,base=self.repository();(root/'unrelated.txt').write_text('preserve')
        with self.assertRaises(pub.PublicationError):pub.publish_snapshot(root,self.cache,'example/board',base)
        self.assertEqual((root/'unrelated.txt').read_text(),'preserve')
        (root/'unrelated.txt').unlink();(self.cache/'publication-pending.json').write_text('{}')
        with self.assertRaises(pub.PublicationError):pub.publish_snapshot(root,self.cache,'example/board',base)

    def test_mocked_pr_path_merges_exact_snapshot_and_syncs_main(self):
        root,base=self.repository();original=pub.command;merged=False
        def api(repo,path,payload=None):
            op=json.loads((self.cache/'publication-pending.json').read_text())
            if path=='pulls':return {'number':1,'html_url':'https://github.com/example/board/pull/1'}
            if path.endswith('/files'):return [{'filename':name} for name in op['changed_files']]
            return {'head':{'sha':op['head']},'base':{'sha':base},'merged':merged,'merge_commit_sha':op['head'] if merged else None}
        def run(args,**kwargs):
            nonlocal merged
            if args[:3]==['gh','pr','merge']:
                op=json.loads((self.cache/'publication-pending.json').read_text())
                self.assertEqual(args[-2:],['--match-head-commit',op['head']])
                # Local bare fixture simulates GitHub's accepted merge; no network.
                pub.git(root,'push','origin',op['head']+':refs/heads/main');merged=True;return b''
            return original(args,**kwargs)
        with patch.object(pub,'gh',side_effect=api),patch.object(pub,'command',side_effect=run):
            result=pub.publish_snapshot(root,self.cache,'example/board',base)
        self.assertEqual(result['status'],'MERGED')
        self.assertEqual(result['ahead_behind'].split(),['0','0'])
        self.assertFalse((self.cache/'publication-pending.json').exists())
        self.assertTrue((self.cache/'publications/pr-1.json').exists())
        self.assertEqual(pub.git(root,'status','--porcelain'),'')

    def test_changed_remote_head_retains_pending_without_merge(self):
        root,base=self.repository()
        def api(repo,path,payload=None):
            if path=='pulls':return {'number':1,'html_url':'https://github.com/example/board/pull/1'}
            if path.endswith('/files'):return []
            return {'head':{'sha':'0'*40},'base':{'sha':base}}
        with patch.object(pub,'gh',side_effect=api):
            with self.assertRaises(pub.PublicationError):pub.publish_snapshot(root,self.cache,'example/board',base)
        self.assertEqual(json.loads((self.cache/'publication-pending.json').read_text())['status'],'PR_CREATED')
        self.assertEqual(pub.git(root,'rev-parse','HEAD'),base)

    def test_confirmed_prior_merge_is_finalized_without_another_merge_request(self):
        root,base=self.repository();payload,validation=pub.validate_snapshot(self.cache)
        commit,changed=pub.build_commit(root,self.cache,base,payload)
        pub.git(root,'push','origin',commit+':refs/heads/main')
        pub.write_json(self.cache/'publication-pending.json',{'repo':'example/board','pr_number':1,'head':commit,
                       'status':'PR_CREATED','validation':validation,'changed_files':changed})
        with patch.object(pub,'gh',return_value={'merged':True,'head':{'sha':commit},'merge_commit_sha':commit}) as api:
            result=pub.publish_snapshot(root,self.cache,'example/board',base)
        self.assertEqual(result['status'],'MERGED')
        self.assertEqual(api.call_count,1)
        self.assertFalse((self.cache/'publication-pending.json').exists())


if __name__=='__main__':unittest.main()
