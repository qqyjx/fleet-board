import copy
import datetime as dt
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
        old['jobs'][0]['progress']['done']=0
        for name,content in [('fleet.json',json.dumps(old)),('curves.json','{}'),('history.jsonl',json.dumps({'t':'old'})+'\n')]:
            (root/'data'/name).write_text(content)
        (root/'source.py').write_text('original source\n')
        pub.git(root,'add','data','source.py')
        pub.git(root,'-c','core.hooksPath=/dev/null','commit','-m','base',
                env=dict(os.environ,GIT_AUTHOR_DATE='2020-01-01T00:00:00Z',GIT_COMMITTER_DATE='2020-01-01T00:00:00Z'))
        pub.git(root,'push','-u','origin','main')
        base=pub.git(root,'rev-parse','HEAD')
        (self.cache/'history.jsonl').write_bytes((root/'data/history.jsonl').read_bytes()+json.dumps({'t':'new'}).encode()+b'\n')
        return root,base

    def test_valid_snapshot(self):
        payload,summary=pub.validate_snapshot(self.cache)
        self.assertEqual(set(payload),set(pub.FILES));self.assertEqual(summary['jobs'],1)

    def test_timestamp_utilization_and_runtime_noise_do_not_publish(self):
        before=fixture('old');after=copy.deepcopy(before)
        after.update(generated_at='new',collect_s=123)
        after['boxes'][0]['cards'][0].update(mem_used=300,util=80)
        after['jobs'][0].update(runtime={'observed_at':123},detail='new log timestamp')
        self.assertEqual(pub.material_state(before),pub.material_state(after))

    def test_completion_and_ownership_are_material(self):
        before=fixture()
        for change in ('done','status','owner','lock'):
            after=copy.deepcopy(before)
            if change=='done':after['jobs'][0]['progress']['done']=2
            elif change=='status':after['jobs'][0]['status']='failed'
            elif change=='owner':after['boxes'][0]['cards'][0]['owner']='other'
            else:after['boxes'][0]['cards'][0]['lock_state']='held'
            self.assertNotEqual(pub.material_state(before),pub.material_state(after))

    def test_transfer_byte_sampling_waits_for_outcome(self):
        before=fixture();before['jobs'][0]['progress']['unit']='MiB'
        after=copy.deepcopy(before);after['jobs'][0]['progress']['done']=2
        self.assertEqual(pub.material_state(before),pub.material_state(after))
        after['jobs'][0]['status']='done'
        self.assertNotEqual(pub.material_state(before),pub.material_state(after))

    def test_remote_main_timestamp_enforces_daily_boundary(self):
        root,base=self.repository();payload,_=pub.validate_snapshot(self.cache)
        last=int(pub.git(root,'log','-1','--format=%ct',base,'--','data/fleet.json'))
        self.assertEqual(pub.publication_decision(root,base,payload,
            now=dt.datetime.fromtimestamp(last+86399,dt.timezone.utc))['status'],'BATCH_NOT_DUE')
        self.assertEqual(pub.publication_decision(root,base,payload,
            now=dt.datetime.fromtimestamp(last+86400,dt.timezone.utc))['status'],'ELIGIBLE')

    def test_unchanged_state_skips_git_and_pr_creation(self):
        root,base=self.repository();self.write_cache(fixture('old'))
        fleet=fixture('new');fleet['jobs'][0]['progress']['done']=0
        self.write_cache(fleet)
        with patch.object(pub,'gh') as api,patch.object(pub,'build_commit') as commit:
            self.assertEqual(pub.publish_snapshot(root,self.cache,'example/board',base)['status'],'NO_MATERIAL_CHANGE')
        api.assert_not_called();commit.assert_not_called()

    def test_changed_state_inside_window_preserves_cache_without_pr(self):
        root,base=self.repository()
        pub.git(root,'-c','core.hooksPath=/dev/null','commit','--amend','--no-edit')
        base=pub.git(root,'rev-parse','HEAD');pub.git(root,'push','--force','origin','main')
        before={name:(self.cache/name).read_bytes() for name in pub.FILES}
        with patch.object(pub,'gh') as api,patch.object(pub,'build_commit') as commit:
            self.assertEqual(pub.publish_snapshot(root,self.cache,'example/board',base)['status'],'BATCH_NOT_DUE')
        api.assert_not_called();commit.assert_not_called()
        self.assertEqual(before,{name:(self.cache/name).read_bytes() for name in pub.FILES})

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

    def test_unavailable_utilization_keeps_error_observation(self):
        data=fixture();data['boxes'][0]['cards'][0].update(util=-1,error='[N/A]',owner='other',lock_state='held')
        self.write_cache(data)
        payload,_=pub.validate_snapshot(self.cache)
        card=json.loads(payload['fleet.json'])['boxes'][0]['cards'][0]
        self.assertEqual((card['util'],card['error'],card['owner'],card['lock_state']),
                         (-1,'[N/A]','other','held'))

    def test_invalid_negative_utilization_is_not_masked(self):
        for util,error in [(-1,None),(-1,''),(-1,'  '),(-2,'[N/A]')]:
            with self.subTest(util=util,error=error):
                data=fixture();data['boxes'][0]['cards'][0].update(util=util,error=error)
                self.write_cache(data)
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
        root,base=self.repository();merged=False
        def api(repo,path,payload=None,*,method='POST'):
            nonlocal merged
            op=json.loads((self.cache/'publication-pending.json').read_text())
            if path=='pulls':return {'number':1,'html_url':'https://github.com/example/board/pull/1'}
            if path.endswith('/files'):return [{'filename':name} for name in op['changed_files']]
            if path.endswith('/merge'):
                self.assertEqual(method,'PUT')
                self.assertEqual(payload,{'sha':op['head'],'merge_method':'merge'})
                # Local bare fixture simulates GitHub's accepted merge; no network.
                pub.git(root,'push','origin',op['head']+':refs/heads/main');merged=True
                return {'merged':True,'sha':op['head']}
            return {'head':{'sha':op['head']},'base':{'sha':base},'merged':merged,'merge_commit_sha':op['head'] if merged else None}
        with patch.object(pub,'gh',side_effect=api):
            result=pub.publish_snapshot(root,self.cache,'example/board',base)
        self.assertEqual(result['status'],'MERGED')
        self.assertEqual(result['ahead_behind'].split(),['0','0'])
        self.assertFalse((self.cache/'publication-pending.json').exists())
        self.assertTrue((self.cache/'publications/pr-1.json').exists())
        self.assertEqual(pub.git(root,'status','--porcelain'),'')

    def test_merge_api_uses_put_and_expected_head_on_old_gh(self):
        payload={'sha':'a'*40,'merge_method':'merge'}
        with patch.object(pub,'command',return_value=b'{"merged":true,"sha":"merge"}') as run:
            result=pub.gh('example/board','pulls/1/merge',payload,method='PUT')
        self.assertTrue(result['merged'])
        self.assertEqual(run.call_args.args[0],['gh','api','repos/example/board/pulls/1/merge','--method','PUT','--input','-'])
        self.assertEqual(json.loads(run.call_args.kwargs['data']),payload)

    def test_api_rejection_keeps_publication_pending(self):
        root,base=self.repository()
        def api(repo,path,payload=None,*,method='POST'):
            op=json.loads((self.cache/'publication-pending.json').read_text())
            if path=='pulls':return {'number':1,'html_url':'https://github.com/example/board/pull/1'}
            if path.endswith('/files'):return [{'filename':name} for name in op['changed_files']]
            if path.endswith('/merge'):return {'merged':False,'message':'head changed'}
            return {'head':{'sha':op['head']},'base':{'sha':base},'merged':False}
        with patch.object(pub,'gh',side_effect=api):
            with self.assertRaises(pub.PublicationError):pub.publish_snapshot(root,self.cache,'example/board',base)
        self.assertEqual(json.loads((self.cache/'publication-pending.json').read_text())['status'],'PR_CREATED')
        self.assertEqual(pub.git(root,'rev-parse','HEAD'),base)
        self.assertFalse((self.cache/'publications').exists())

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
