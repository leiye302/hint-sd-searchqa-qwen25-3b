"""Fetch unmodified public HINT-SD and SDAR sources or pinned base model weights."""
import argparse
from pathlib import Path
import time
import uuid
from .common import ROOT, HINT, SDAR, atomic_json, git, load, sha, verify_sources


def framework(mirrors=None):
    lock = load('upstream.json')
    for key, target in [('hint_sd', HINT), ('sdar_environment', SDAR)]:
        source = lock[key]
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + '-' + uuid.uuid4().hex[:8])
        mirror = (mirrors or {}).get(key)
        if mirror:
            git(ROOT, 'clone', '--no-checkout', '--no-hardlinks', str(Path(mirror).resolve()), str(staging))
            git(staging, 'checkout', '--detach', source['commit'])
            git(staging, 'remote', 'set-url', 'origin', source['repository'])
        else:
            staging.mkdir()
            git(staging, 'init')
            git(staging, 'remote', 'add', 'origin', source['repository'])
            for attempt in range(3):
                try:
                    git(staging, 'fetch', '--depth', '1', 'origin', source['commit'])
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            git(staging, 'checkout', '--detach', 'FETCH_HEAD')
        staging.rename(target)
    verify_sources()
    print('Pristine official HINT-SD and SDAR Search-QA sources verified; no GPU or training used.')


def download(size):
    from huggingface_hub import snapshot_download
    spec = load('models.json')[size]
    target = ROOT / 'models' / spec['model_id'].split('/')[-1]
    snapshot_download(spec['model_id'], revision=spec['revision'], local_dir=target,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE', 'README.md'])
    atomic_json(target / 'download_receipt.json', dict(spec, files={
        path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
        for path in sorted(target.iterdir()) if path.is_file() and path.name != 'download_receipt.json'}))
    print('Pinned public base model downloaded and hashed:', spec['model_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--download-model', choices=['1.5b', '3b'])
    parser.add_argument('--hint-source', type=Path, help='Optional read-only public source mirror')
    parser.add_argument('--sdar-source', type=Path, help='Optional read-only public source mirror')
    args = parser.parse_args()
    if not any([args.framework, args.verify, args.download_model]):
        parser.error('Choose --framework, --verify or --download-model')
    if args.framework:
        framework({'hint_sd': args.hint_source, 'sdar_environment': args.sdar_source})
    if args.verify:
        verify_sources()
        print('Public source and common Search-QA evaluation locks passed.')
    if args.download_model:
        download(args.download_model)


if __name__ == '__main__':
    main()
