"""CPU-only dual-source locks and exact shared Search-QA task identities."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HINT = ROOT / '.deps/hint-sd'
SDAR = ROOT / '.deps/sdar'
sys.path.insert(0, str(ROOT / 'src'))


def load(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def git(path, *args):
    return subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(path), *args],
        check=True, capture_output=True, text=True, encoding='utf-8', errors='replace').stdout.strip()


def devices(spec):
    result = [x.strip() for x in spec.split(',')]
    if len(result) != 4 or len(set(result)) != 4 or any(not x.isdigit() for x in result):
        raise ValueError('Choose four distinct allocated physical GPU indices')
    return result


def data_directory():
    return Path(os.environ.get('HINTSD_QA_DATA_DIR', ROOT / 'data/searchqa')).resolve()


def verify_tasks(root=None):
    del root
    for name, expected in load('upstream.json')['shared_locks'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Shared Search-QA source/data/protocol changed: ' + name)
    recipe = load('experiment.json')
    expected = {'max_env_steps': 4, 'history_length': 4, 'max_prompt_tokens': 4096,
                'max_completion_length': 512, 'eval_every': 20, 'eval_temperature': 0.,
                'eval_rollout_n': 1, 'eval_batch_size': 512, 'updates': 150}
    if any(recipe[key] != value for key, value in expected.items()):
        raise ValueError('Shared evaluation generation/budget parameter changed')
    from validation_subset import manifest
    value = manifest()
    if value['questions'] != 448 or set(value['source_counts'].values()) != {64}:
        raise ValueError('Expected the shared seven-source448 monitoring manifest')
    return value


def verify_sources():
    lock = load('upstream.json')
    for key, directory in [('hint_sd', HINT), ('sdar_environment', SDAR)]:
        source = lock[key]
        if git(directory, 'rev-parse', 'HEAD') != source['commit']:
            raise ValueError('Upstream revision differs: ' + key)
        if git(directory, 'diff', '--name-only', 'HEAD'):
            raise ValueError('Upstream tracked source must remain pristine: ' + key)
        for name, expected in source['files'].items():
            content = (directory / name).read_bytes().replace(b'\r\n', b'\n')
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Upstream source hash differs: ' + name)
    verify_tasks()
    return lock


def task_row(value, index, split):
    from common import plain, question_id
    kwargs = plain(value['env_kwargs'])
    return {'task_id': question_id(kwargs), 'qa_index': int(index), 'seed': 0,
            'split': split, 'env_kwargs': kwargs, 'data_source': kwargs['data_source']}


def evaluation_spec(scope):
    directory = data_directory()
    if scope == 'subset':
        receipt = json.loads((directory / 'validation_subset_receipt.json').read_text(encoding='utf-8'))
        return receipt['rows'], receipt['source_test_indices']
    if scope == 'full':
        receipt = json.loads((directory / 'data_receipt.json').read_text(encoding='utf-8'))
        rows = receipt['files']['test']['rows']
        return rows, list(range(rows['count']))
    raise ValueError('Evaluation scope must be subset or full')


@lru_cache(maxsize=4)
def _task_rows(directory, scope):
    import pandas as pd
    name = 'validation_subset.parquet' if scope == 'subset' else 'test.parquet'
    frame = pd.read_parquet(Path(directory) / name, columns=['env_kwargs'])
    rows = [task_row(value, index, 'validation') for index, value in enumerate(frame.to_dict('records'))]
    expected, _ = evaluation_spec(scope)
    if [row['task_id'] for row in rows] != expected['question_ids']:
        raise ValueError('Actual evaluation questions/order differ from the shared receipt')
    return rows


def task_rows(scope='subset'):
    return _task_rows(str(data_directory()), scope)


def checkpoint_complete(path):
    path = Path(path)
    required = ['adapter_config.json', 'adapter_model.safetensors',
        'teacher_adapter/adapter_config.json', 'teacher_adapter/adapter_model.safetensors',
        'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'training_args.bin']
    missing = [name for name in required if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        raise ValueError('Incomplete resumable checkpoint: ' + ', '.join(missing))
    return path
