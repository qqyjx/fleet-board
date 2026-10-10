"""Publish generated telemetry through a data-only PR; never stage the checkout."""
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

FILES = ('fleet.json', 'curves.json', 'history.jsonl')
MIN_PUBLICATION_SECONDS = 24 * 60 * 60


class PublicationError(RuntimeError):
    pass


def command(args, *, cwd=None, data=None, env=None):
    result = subprocess.run(args, cwd=cwd, input=data, capture_output=True,
                            env=env, timeout=120)
    if result.returncode:
        raise PublicationError('COMMAND_FAILED: ' + args[0])
    return result.stdout


def git(root, *args, data=None, env=None):
    return command(['git', '-C', str(root), *args], data=data, env=env).decode().strip()


def gh(repo, path, payload=None, *, method='POST'):
    args = ['gh', 'api', 'repos/' + repo + '/' + path]
    if payload is not None:
        args += ['--method', method, '--input', '-']
    raw = command(args, data=json.dumps(payload).encode() if payload is not None else None)
    return json.loads(raw)


def write_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def settings(root):
    config = root / '.fleet-local.json'
    value = json.loads(config.read_text()) if config.exists() else {}
    cache_name = os.environ.get('FLEET_CACHE_DIR', value.get('cache_dir'))
    if not cache_name:
        raise PublicationError('Configure FLEET_CACHE_DIR or .fleet-local.json outside the checkout')
    cache = Path(cache_name).expanduser().resolve()
    if cache.is_relative_to(root.resolve()):
        raise PublicationError('Telemetry cache must be outside the Git checkout')
    origin = git(root, 'remote', 'get-url', 'origin')
    match = re.fullmatch(r'(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+?)(?:\.git)?', origin)
    if not match or any(x in ('.', '..') for x in match[1].split('/')):
        raise PublicationError('Expected the configured github.com repository')
    repo = match[1]
    if value.get('repository', repo) != repo:
        raise PublicationError('Repository configuration mismatch')
    cache.mkdir(parents=True, exist_ok=True)
    return cache, repo


@contextlib.contextmanager
def collector_lock(cache):
    with (cache / 'collector.lock').open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def synchronize(root):
    if git(root, 'branch', '--show-current') != 'main' or git(root, 'status', '--porcelain'):
        raise PublicationError('Source checkout must be clean main; preserve unrelated work')
    before = git(root, 'rev-parse', 'HEAD')
    git(root, 'fetch', '-q', 'origin', 'main')
    git(root, '-c', 'core.hooksPath=/dev/null', 'merge', '--ff-only', 'origin/main')
    after = git(root, 'rev-parse', 'HEAD')
    changed = git(root, 'diff', '--name-only', before, after).splitlines()
    return after, any(name.endswith('.py') for name in changed)


def initialize_cache(root, cache):
    for name in FILES:
        source, target = root / 'data' / name, cache / name
        if not target.exists() and source.is_file():
            shutil.copyfile(source, target)
    published = root / 'data/history.jsonl'
    local = cache / 'history.jsonl'
    if published.exists() and local.exists():
        old, current = published.read_bytes(), local.read_bytes()
        if old.startswith(current):
            for name in FILES:
                shutil.copyfile(root / 'data' / name, cache / name)
        elif not current.startswith(old):
            raise PublicationError('History diverged; preserve both sides for coordinator review')


def strict_json(data):
    def reject(value):
        raise PublicationError('Nonfinite JSON value: ' + value)
    return json.loads(data, parse_constant=reject)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_snapshot(cache):
    payloads = {}
    for name in FILES:
        path = cache / name
        if path.is_symlink() or not path.is_file():
            raise PublicationError('Missing or redirected snapshot file: ' + name)
        payloads[name] = path.read_bytes()
    fleet = strict_json(payloads['fleet.json'])
    if not isinstance(fleet, dict) or not isinstance(fleet.get('generated_at'), str):
        raise PublicationError('Missing snapshot timestamp')
    if not isinstance(fleet.get('boxes'), list) or not isinstance(fleet.get('jobs'), list):
        raise PublicationError('Invalid snapshot collections')
    ids = [job.get('id') for job in fleet['jobs']]
    if not all(isinstance(x, str) and x for x in ids) or len(ids) != len(set(ids)):
        raise PublicationError('Missing or duplicate job ids')
    box_names = [box.get('name') for box in fleet['boxes']]
    if not all(isinstance(name, str) and name for name in box_names) or len(box_names) != len(set(box_names)):
        raise PublicationError('Missing or duplicate machine names')
    for box in fleet['boxes']:
        if not isinstance(box.get('name'), str) or type(box.get('reachable')) is not bool or not isinstance(box.get('cards'), list):
            raise PublicationError('Invalid machine observation')
        card_ids = [card.get('idx') for card in box['cards']]
        if len(card_ids) != len(set(card_ids)):
            raise PublicationError('Duplicate GPU indices')
        for card in box['cards']:
            if not all(number(card.get(key)) for key in ('idx', 'mem_used', 'mem_total')):
                raise PublicationError('Invalid GPU observation')
            # The collector preserves nvidia-smi N/A as -1 with its error text;
            # the board renders this as ERR, never as an idle 0% observation.
            unavailable = (type(card.get('util')) in (int, float)
                           and card['util'] == -1
                           and isinstance(card.get('error'), str)
                           and bool(card['error'].strip()))
            if not (number(card.get('util')) or unavailable):
                raise PublicationError('Invalid GPU observation')
            if card['mem_used'] > card['mem_total'] or card['util'] > 100:
                raise PublicationError('GPU observation out of range')
    for job in fleet['jobs']:
        progress = job.get('progress', {})
        if not isinstance(job.get('status'), str) or not all(number(progress.get(key)) for key in ('done', 'total')):
            raise PublicationError('Invalid job progress')
    if not isinstance(strict_json(payloads['curves.json']), dict):
        raise PublicationError('Invalid curve collection')
    history = [strict_json(line) for line in payloads['history.jsonl'].splitlines() if line.strip()]
    if not history or history[-1].get('t') != fleet['generated_at']:
        raise PublicationError('Snapshot/history timestamp mismatch')
    return payloads, {'generated_at': fleet['generated_at'], 'boxes': len(fleet['boxes']), 'jobs': len(fleet['jobs']),
                      'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}


def build_commit(root, cache, base, payloads):
    if set(payloads) != set(FILES):
        raise PublicationError('Only the three generated snapshot files may be published')
    previous = command(['git', '-C', str(root), 'show', base + ':data/history.jsonl'])
    if not payloads['history.jsonl'].startswith(previous):
        raise PublicationError('Published history would be overwritten')
    with tempfile.TemporaryDirectory(prefix='git-index-', dir=cache) as directory:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / 'index'), GIT_TERMINAL_PROMPT='0')
        git(root, 'read-tree', base, env=env)
        for name, data in payloads.items():
            oid = git(root, 'hash-object', '-w', '--stdin', data=data, env=env)
            git(root, 'update-index', '--add', '--cacheinfo', '100644,' + oid + ',data/' + name, env=env)
        tree = git(root, 'write-tree', env=env)
        changed = git(root, 'diff', '--name-only', base, tree, env=env).splitlines()
        if not changed:
            return None, []
        if not set(changed).issubset({'data/' + name for name in FILES}):
            raise PublicationError('Snapshot transaction includes non-data changes')
        commit = git(root, 'commit-tree', tree, '-p', base, '-m', 'Update verified fleet telemetry snapshot', env=env)
    return commit, changed


def material_state(fleet):
    """Compare job outcomes/progress and ownership, not sampling noise."""
    boxes = []
    for box in fleet['boxes']:
        cards = [{key: card.get(key) for key in ('idx', 'owner', 'lock_state', 'borrowed')}
                 for card in sorted(box['cards'], key=lambda card: card['idx'])]
        boxes.append({'name': box['name'], 'reachable': box['reachable'], 'cards': cards,
                      'allocation_detail': box.get('allocation_detail')})
    jobs = []
    for job in fleet['jobs']:
        item = {key: job.get(key) for key in ('id', 'status', 'box', 'cards')}
        progress = job.get('progress', {})
        if progress.get('unit') not in ('MiB', 'GiB', 'bytes'):
            item['progress'] = progress
        if job['status'] not in ('running', 'done'):
            item['blocker'] = job.get('detail')
        jobs.append(item)
    return {'boxes': sorted(boxes, key=lambda box: box['name']),
            'jobs': sorted(jobs, key=lambda job: job['id'])}


def publication_decision(root, base, payloads, *, now=None, force=False):
    """Use remote-main history so another publishing host shares the same cap.

    force=True (FLEET_FORCE_PUBLISH=1, only when the user asks for an extra publication) skips the 24-hour interval and
    nothing else: an unchanged material state still means no PR, and the publication record says it was forced."""
    before = strict_json(command(['git', '-C', str(root), 'show', base + ':data/fleet.json']))
    previous_curves = strict_json(command(['git', '-C', str(root), 'show', base + ':data/curves.json']))
    if (material_state(before) == material_state(strict_json(payloads['fleet.json']))
            and previous_curves == strict_json(payloads['curves.json'])):
        return {'status': 'NO_MATERIAL_CHANGE'}
    last = int(git(root, 'log', '-1', '--format=%ct', base, '--', 'data/fleet.json', 'data/curves.json'))
    current = (now or dt.datetime.now(dt.timezone.utc)).timestamp()
    eligible = last + MIN_PUBLICATION_SECONDS
    if current < eligible and not force:
        return {'status': 'BATCH_NOT_DUE', 'next_eligible_utc': dt.datetime.fromtimestamp(eligible, dt.timezone.utc).isoformat()}
    return {'status': 'ELIGIBLE', 'forced': bool(force and current < eligible)}


def finish_publication(root, cache, operation, merge_sha):
    pending = cache / 'publication-pending.json'
    operation.update(status='MERGED', merge=merge_sha)
    write_json(pending, operation)
    synchronize(root)
    operation.update(local_main=git(root, 'rev-parse', 'HEAD'), ahead_behind=git(root, 'rev-list', '--left-right', '--count', 'HEAD...origin/main'))
    receipts = cache / 'publications'; receipts.mkdir(exist_ok=True)
    write_json(receipts / ('pr-' + str(operation['pr_number']) + '.json'), operation)
    pending.unlink()
    return operation


def publish_snapshot(root, cache, repo, base):
    pending = cache / 'publication-pending.json'
    if pending.exists():
        operation = json.loads(pending.read_text())
        if operation.get('repo') == repo and operation.get('pr_number'):
            existing = gh(repo, 'pulls/' + str(operation['pr_number']))
            if existing.get('merged') and existing['head']['sha'] == operation['head']:
                return finish_publication(root, cache, operation, existing['merge_commit_sha'])
        raise PublicationError('A previous publication needs reconciliation; see publication-pending.json')
    if git(root, 'status', '--porcelain') or git(root, 'rev-parse', 'HEAD') != base:
        raise PublicationError('Source checkout changed during collection')
    git(root, 'fetch', '-q', 'origin', 'main')
    if git(root, 'rev-parse', 'origin/main') != base:
        raise PublicationError('Remote main advanced; keep cache and synchronize on next collection')
    payloads, validation = validate_snapshot(cache)
    decision = publication_decision(root, base, payloads, force=os.environ.get('FLEET_FORCE_PUBLISH') == '1')
    if decision['status'] != 'ELIGIBLE':
        return decision
    commit, changed = build_commit(root, cache, base, payloads)
    if commit is None:
        return {'status': 'NO_DATA_CHANGE'}
    branch = 'codex/fleet-data-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    operation = {'status': 'PREPARED', 'repo': repo, 'branch': branch, 'base': base, 'head': commit,
                 'changed_files': changed, 'validation': validation, 'recorded_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                 'forced_inside_24h': decision.get('forced', False)}
    write_json(pending, operation)
    git(root, 'push', '-q', 'origin', commit + ':refs/heads/' + branch)
    operation['status'] = 'PUSHED'; write_json(pending, operation)
    body = ('Publish the current read-only fleet observations. Only generated fleet.json, curves.json and history.jsonl may change. '
            'Local snapshot schema/range/id checks and append-only history checks passed; no collector code or scientific results changed. '
            'Base SHA: ' + base + '. Snapshot: ' + validation['generated_at'] + '.'
            + (' Published inside the 24-hour interval at the user\'s request (FLEET_FORCE_PUBLISH=1).' if decision.get('forced') else ''))
    pr = gh(repo, 'pulls', {'head': branch, 'base': 'main', 'title': 'Update fleet snapshot ' + validation['generated_at'], 'body': body})
    operation.update(status='PR_CREATED', pr_number=pr['number'], url=pr['html_url']); write_json(pending, operation)
    # Recheck the actual PR payload after creation, before asking GitHub to merge.
    remote = gh(repo, 'pulls/' + str(pr['number']))
    files = gh(repo, 'pulls/' + str(pr['number']) + '/files')
    if remote['head']['sha'] != commit or remote['base']['sha'] != base:
        raise PublicationError('PR head/base changed before merge; preserve it for review')
    if {item['filename'] for item in files} != set(changed):
        raise PublicationError('PR file scope differs from validated snapshot')
    outcome = gh(repo, 'pulls/' + str(pr['number']) + '/merge',
                 {'sha': commit, 'merge_method': 'merge'}, method='PUT')
    if not outcome.get('merged') or not outcome.get('sha'):
        raise PublicationError('GitHub did not confirm the requested merge')
    merged = gh(repo, 'pulls/' + str(pr['number']))
    if not merged.get('merged') or merged['head']['sha'] != commit or merged.get('merge_commit_sha') != outcome['sha']:
        raise PublicationError('Merge outcome is not confirmed')
    # Keep the branch until the App coordinator has attached and recorded this PR.
    return finish_publication(root, cache, operation, merged['merge_commit_sha'])
