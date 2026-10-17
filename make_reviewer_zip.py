#!/usr/bin/env python3
"""Create a reviewed, integrity-checked GraphPAE source + evidence ZIP (no datasets)."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from statistics import mean, pstdev
from zipfile import ZipFile, ZIP_DEFLATED
import json, os, re, sys

root = Path(os.environ.get('GRAPHPAE_REVISION_ROOT',
    '/home/insuper/GraphPAE_Project/PRL_Revision_20261014_004139_949694')).resolve()
out = root / 'Balanced_IA_GraphPAE_PRL_Revision_20261017.zip'
tmp = root / (out.name + '.building')
results = root / 'revision_results'
assert (root / 'train_node.py').is_file() and (root / 'ia_utils.py').is_file(), 'Missing final source'
assert (root / 'config').is_dir() and (root / 'scripts').is_dir(), 'Missing configs or scripts'
assert results.is_dir(), 'Missing revision_results'
assert not out.exists() and not tmp.exists(), 'ZIP already exists: do not overwrite without review'

paired_batches = sorted(results.glob('PAIRED_CONTROL_*'))
assert paired_batches, 'Missing paired-edge reviewer control'
paired = paired_batches[-1]
expected_logs = [paired / f'{ds}_{sampler}_{budget:.2f}.log'
                 for ds in ('cora','chameleon')
                 for sampler in ('random','ia')
                 for budget in (0.10,0.25)]
for p in expected_logs:
    assert p.is_file(), f'Missing paired run: {p}'
    value = p.read_text(errors='replace')
    assert 'final result:' in value and 'Traceback (most recent call last)' not in value, f'Incomplete/failed paired run: {p}'

allowed = {'.py','.sh','.yaml','.yml','.toml','.json','.jsonl','.log','.txt','.csv','.tsv','.md','.ini'}
blocked_dirs = {'dataset','datasets','checkpoints','__pycache__','.git','.ssh','wandb','cache','.cache','secrets','node_modules'}
blocked_names = {'.env','id_rsa','id_ed25519','credentials.json','secrets.json'}
secret = re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',re.I)
files = {}

def add(dst, src):
    if src.is_symlink() or not src.is_file():
        return
    relative = src.relative_to(root) if src.is_relative_to(root) else src.relative_to(root.parent)
    if any(part.lower() in blocked_dirs for part in relative.parts):
        return
    if src.name.lower() in blocked_names:
        return
    if src.suffix.lower() not in allowed:
        return
    if src.stat().st_size > 15_000_000:
        raise RuntimeError(f'Unexpected large evidence/source file: {src}')
    data = src.read_bytes()
    if secret.search(data):
        raise RuntimeError(f'Potential secret detected; manually review: {src}')
    assert dst not in files, f'Duplicate ZIP entry {dst}'
    if dst.startswith('source/') and src.suffix == '.py':
        compile(data, str(src), 'exec')
    files[dst] = data

for p in sorted(root.iterdir()):
    if p.is_file():
        if p.name == 'README.md':
            add('docs/ORIGINAL_SERVER_README.md', p)
        else:
            add('source/' + p.name, p)
for section in ('config','scripts'):
    for p in sorted((root / section).rglob('*')):
        if p.is_file():
            add('source/' + p.relative_to(root).as_posix(), p)
for p in sorted(results.rglob('*')):
    if p.is_file():
        add('evidence/' + p.relative_to(root).as_posix(), p)
# Older experiments remain clearly segregated from the 2026 PRL revision.
historical = root.parent / 'results_server'
if historical.is_dir():
    for p in sorted(historical.rglob('*')):
        if p.is_file():
            add('evidence/historical_results_server/' + p.relative_to(historical).as_posix(), p)

num_checked=0
for p in sorted(results.rglob('seed_results.json')):
    summary = p.with_name('summary.json')
    if not summary.is_file():
        continue
    try:
        rows=json.loads(p.read_text())
        report=json.loads(summary.read_text())
        if not (isinstance(rows,list) and isinstance(report,dict)
                and {'mean','std_population','accuracies'}<=report.keys()):
            continue
        vals=[float(row['accuracy']) for row in rows]
        assert vals and len(vals)==len(report['accuracies']), str(p)
        assert all(abs(a-b)<1e-8 for a,b in zip(vals,report['accuracies'])), str(p)
        assert abs(mean(vals)-float(report['mean']))<1e-8, str(p)
        assert abs(pstdev(vals)-float(report['std_population']))<1e-8, str(p)
        num_checked+=1
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        raise RuntimeError(f'Seed results/summary mismatch at {p}: {exc}') from exc
assert num_checked>0, 'No matching seed-level accuracy summaries verified'

readme = '''# Balanced IA-GraphPAE — PRL revision

This repository contains the Balanced Importance-Aware GraphPAE source and
original experiment records for the Pattern Recognition Letters revision.

- **Implementation:** root Python files, `config/`, `scripts/`.
- **Review evidence:** `evidence/revision_results/`, including raw logs,
  seed-level results, effective configurations, and reviewer controls.
- **Integrity:** `SHA256_MANIFEST.json` records SHA-256 for every delivered file.
- **Provenance:** `REPRODUCIBILITY.md` explains the experiment archive.
- **Datasets:** not redistributed. Reproduce using the processed datasets
  and splits described in the article and experiment configuration records.

**Important:** historical `results/` entries from the earlier GitHub release
may use a different code snapshot, evaluation protocol or random seeds.
Do not treat them as the revised paper's results. The preserved revision
records are under `evidence/revision_results/`.

The positional-edge budget refers to selected directed **non-self** edges.
Self-loops are separately retained under the default protocol and are
removed in the dedicated self-loop ablation. Reported gains are
**dataset-dependent**, not universal.

See the article for full configuration details and dataset attribution.
'''
repro = f'''# PRL Revision — evidence provenance

Pack generated (UTC): {datetime.now(timezone.utc).isoformat()}
Original server project: `{root}`
Paired-edge control batch: `{paired.name}`
Paired-edge completed log files checked: {len(expected_logs)}
Seed-level result/summary pairs validated: {num_checked}

## Interpretation

`evidence/revision_results/` is the preserved experiment audit trail.
`evidence/historical_results_server/` contains older GraphPAE experiments when available;
those records must NOT be conflated with the controlled PRL-revision runs.
It contains main runs, ablations, budget sensitivity, large graphs,
self-loop tests, and paired-edge controls where recorded. This archive
also contains exploratory runs: **not every result is reported in the paper**.
Compare the paper to the corresponding raw logs and effective configurations.
The selector's non-self directed edge budget is distinct from active encoder
entries including self-loops; their loss contribution is separately defined.

## Verification and limitations

Code was syntax-checked, paired-control final logs were checked, selected
seed-result means and population SDs were recomputed, and archived bytes
were checked with SHA-256. This packaging process does **not** rerun
experiments or establish that every manuscript number matches a log.
Dataset tensors, checkpoints, server machine files, and secrets are excluded.
The original working source tree was not changed by the packager.
'''
files['source/README.md']=readme.encode('utf-8')
files['source/.gitignore']=b'dataset/\ndatasets/\ncheckpoints/\n__pycache__/\n*.pt\n*.pth\n*.ckpt\n*.zip\n.env\n'
files['source/REPRODUCIBILITY.md']=repro.encode('utf-8')
manifest={name:sha256(data).hexdigest() for name,data in sorted(files.items())}
files['SHA256_MANIFEST.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode('utf-8')
assert len(files)>100, f'Suspiciously small package: {len(files)} files'
try:
    with ZipFile(tmp,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(files.items()):
            z.writestr(name,data)
    with ZipFile(tmp,'r') as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)
        for name,digest in manifest.items():
            assert sha256(z.read(name)).hexdigest()==digest, f'Bad digest: {name}'
    tmp.rename(out)
finally:
    if tmp.exists():
        tmp.unlink()

h=sha256(out.read_bytes()).hexdigest()
(out.with_name(out.name+'.sha256')).write_text(f'{h}  {out.name}\n')
print('ZIP CHECK: PASS')
print('ARCHIVE:',out)
print('FILES:',len(files),'| original source+evidence:',len(manifest))
print('SEED SUMMARIES CHECKED:',num_checked)
print('SIZE MiB:',round(out.stat().st_size/1048576,2))
print('SHA256:',h)
print('ORIGINAL FILES UNCHANGED')
