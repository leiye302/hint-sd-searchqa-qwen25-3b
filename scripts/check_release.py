"""Privacy, history and standalone allowlist audit; never inspect secrets."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = set('''
.env.example .gitattributes .github/workflows/check.yml .gitignore
LICENSE README.md THIRD_PARTY.md
configs/comparison.json configs/data.json configs/evaluation.json configs/experiment.json
configs/models.json configs/release.json configs/requirements-cpu.txt configs/requirements-retriever.txt
configs/retrieval.json configs/upstream.json configs/validation_subset.json
docs/METHOD.md docs/EVALUATION.md
hint_searchqa/__init__.py hint_searchqa/checkpoint_io.py hint_searchqa/common.py
hint_searchqa/engine.py hint_searchqa/environment.py hint_searchqa/feedback.py
hint_searchqa/report.py hint_searchqa/rollout.py hint_searchqa/run.py
hint_searchqa/setup.py hint_searchqa/train.py hint_searchqa/trainer.py
scripts/check_release.py scripts/setup.sh scripts/train.sh scripts/eval.sh
src/common.py src/data.py src/downloads.py src/env_smoke.py src/evaluation_contract.py
src/report.py src/retrieval.py src/setup.py src/validation_subset.py
tests/test_contract.py tests/test_native_trainer.py tests/test_searchqa.py
'''.split())
PRIVATE = [
    re.compile(r'[A-Za-z]:[/\\](?:Users[/\\]|app[/\\]|mnt[/\\])', re.I),
    re.compile(r'/mnt/[a-z0-9_.-]+/(?:test/)?VLA_test(?:/|\b)', re.I),
    re.compile(r'connect\.[a-z0-9]+\.seetacloud\.com', re.I),
    re.compile(r'(?:oc_sk_|github_pat_|ghp_)[A-Za-z0-9_-]{18,}'),
    re.compile(r'\bsk-[A-Za-z0-9_-]{24,}'),
    re.compile(r'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----'),
]


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True).stdout


def audit():
    names = set(git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode('utf-8').strip('\0').split('\0'))
    if names != ALLOWED:
        raise RuntimeError('Public allowlist differs: ' + str(sorted(names ^ ALLOWED)))
    for name in names:
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError('External link refused: ' + name)
        if any(rule.search(path.read_text(encoding='utf-8')) for rule in PRIVATE):
            raise RuntimeError('Private material in public file: ' + name)
    # An unborn repository has no history; once committed scan ALL historic blobs too.
    heads = git('for-each-ref', '--format=%(objectname)').strip()
    blobs = 0
    for line in (git('rev-list', '--objects', '--all').decode().splitlines() if heads else []):
        oid = line.split(' ', 1)[0]
        if git('cat-file', '-t', oid).strip() != b'blob':
            continue
        raw = git('cat-file', 'blob', oid)
        if len(raw) > 2 * 1024**2 or any(rule.search(raw.decode('utf-8', errors='replace')) for rule in PRIVATE):
            raise RuntimeError('Unexpected large/private historical blob; publication refused')
        blobs += 1
    sys.path.insert(0, str(ROOT))
    from hint_searchqa.common import verify_sources
    verify_sources()
    print(f'PASS: {len(names)} public allowlisted files; {blobs} historical blobs; both native source locks.')


if __name__ == '__main__':
    audit()
