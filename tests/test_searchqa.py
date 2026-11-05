"""Native HTTP transitions plus fixed-question evaluation/storage boundary tests."""
import ast
from collections import Counter
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hint_searchqa.common import atomic_json, load, sha, task_row, verify_sources
from hint_searchqa.environment import Game, Service, policy_messages
from hint_searchqa.train import needs_periodic_evaluation, train_or_finalize
from hint_searchqa.checkpoint_io import recover, seal, verify, VerifiedCheckpointMixin


def test_native_environment_search_answer_and_hidden_label():
    pytest.importorskip('omegaconf')
    from env_smoke import fixture_retriever
    value = {'question': 'What is the capital of France?',
             'ground_truth': {'target': ['Paris', 'HIDDEN_ALIAS_NOT_IN_OBSERVATION']}, 'data_source': 'fixture'}
    row = task_row({'env_kwargs': value}, 0, 'train')
    with fixture_retriever() as (url, requests):
        game = Game(row, 4, url)
        try:
            initial = game.state()['observation']
            assert value['question'] in initial
            assert 'HIDDEN_ALIAS' not in str(policy_messages(initial))
            result = game.step('<think>Search evidence.</think><search>capital France</search>')
            assert result['reward'] == 0 and not result['terminated'] and result['action_valid']
            assert 'Paris' in result['observation'] and requests[0]['topk'] == 3
            assert 'HIDDEN_ALIAS' not in result['observation']
            final = game.step('<think>Use retrieved evidence.</think><answer>PARIS!</answer>')
            assert final['reward'] == 1 and final['won'] and final['terminated']
        finally:
            game.close()


def test_native_four_turn_limit_and_instance_isolation():
    pytest.importorskip('omegaconf')
    from env_smoke import fixture_retriever
    row = task_row({'env_kwargs': {'question': 'Capital of France?',
                    'ground_truth': {'target': ['Paris']}, 'data_source': 'fixture'}}, 0, 'validation')
    with fixture_retriever() as (url, requests):
        service = Service(url)
        try:
            first = service.call('create', {'task': row})
            second = service.call('create', {'task': row})
            assert first['instance_id'] != second['instance_id']
            for i in range(4):
                result = service.call('step', {'instance_id': first['instance_id'],
                    'response': '<think>Need evidence.</think><search>capital France</search>'})
                assert result['terminated'] == (i == 3)
            assert result['reward'] == 0 and not result['won']
            correct = service.call('step', {'instance_id': second['instance_id'], 'response': '<answer>Paris</answer>'})
            assert correct['won'] and correct['reward'] == 1
            service.call('release', {'instance_id': first['instance_id']})
            assert first['instance_id'] not in service.games
            with pytest.raises(ValueError, match='released'):
                service.call('step', {'instance_id': first['instance_id'], 'response': '<answer>Paris</answer>'})
        finally:
            service.close()
        # Native SearchEnv terminates at the fourth turn BEFORE another search.
        assert service.games == {} and len(requests) == 3


def evaluation_harness(tmp_path, count=7, mutate=None):
    """Execute the actual adapter evaluate method without importing GPU dependencies."""
    from evaluation_contract import summarize_questions
    tree = ast.parse((ROOT / 'hint_searchqa/trainer.py').read_text(encoding='utf-8'))
    source = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate']
    cls = ast.ClassDef(name='Harness', bases=[], keywords=[], body=methods, decorator_list=[])
    calls = []
    rows = [{'task_id': f'q{i}', 'qa_index': i, 'data_source': f'source{i % 2}'} for i in range(count)]
    expected = {'question_ids': [row['task_id'] for row in rows],
                'source_counts': dict(Counter(row['data_source'] for row in rows))}
    class Episode:
        def __init__(self, row):
            self.task_id = row['task_id']
            self.run_metadata = {'qa_index': row['qa_index'], 'data_source': row['data_source']}
            self.final_reward = float(row['qa_index'] % 2 == 0)
        def to_dict(self):
            return {'task_id': self.task_id}
    def collect(selected, trainer, **kwargs):
        calls.append((len(selected), kwargs))
        episodes = [Episode(row) for row in selected]
        if mutate:
            mutate(episodes)
        return SimpleNamespace(episodes=episodes)
    space = dict(Path=Path, Counter=Counter, time=time, uuid=uuid, atomic_json=atomic_json,
                 evaluation_spec=lambda scope: (expected, list(range(count))),
                 task_rows=lambda scope: rows, run_rollouts=collect)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'actual_evaluate', 'exec'), space)
    instance = space['Harness']()
    instance.args = SimpleNamespace(output_dir=str(tmp_path))
    instance.state = SimpleNamespace(global_step=0)
    instance.recipe = {**load('experiment.json'), 'eval_batch_size': 3}
    instance.log = lambda value: None
    instance.ensure_final_checkpoint = lambda: calls.append('saved-final')
    return instance, calls


@pytest.mark.parametrize('step,scope', [(0, 'subset'), (20, 'subset'), (140, 'subset'), (150, 'full')])
def test_actual_evaluation_complete_ids_scope_and_greedy(tmp_path, step, scope):
    trainer, calls = evaluation_harness(tmp_path)
    trainer.state.global_step = step
    metrics = trainer.evaluate()
    assert metrics[f'eval/{scope}/questions'] == 7
    if step == 150:
        assert calls.pop(0) == 'saved-final'
    assert [length for length, _ in calls] == [3, 3, 1]
    assert all(kwargs == {'rollout_n': 1, 'temperature': 0.} for _, kwargs in calls)
    latest = json.loads((tmp_path / 'evaluation/latest.json').read_text())
    report = json.loads((tmp_path / latest['path'] / 'summary.json').read_text())
    assert report['questions'] == 7 and report['correct'] == 4 and report['student_only']
    assert report['question_micro_em'] == 4 / 7 and report['dataset_macro_em'] == 0.5
    trainer.evaluate()
    assert len(list((tmp_path / f'evaluation/update_{step:03d}').glob('attempt-*'))) == 2


@pytest.mark.parametrize('kind', ['missing', 'identity', 'order', 'source', 'nonbinary'])
def test_incomplete_or_altered_evaluation_never_publishes_success(tmp_path, kind):
    def mutate(episodes):
        if kind == 'missing':
            episodes.pop()
        elif kind == 'identity':
            episodes[0].task_id = 'wrong-question'
        elif kind == 'order':
            episodes.reverse()
        elif kind == 'source':
            episodes[0].run_metadata['data_source'] = 'wrong-source'
        elif kind == 'nonbinary':
            episodes[0].final_reward = 0.5
    trainer, calls = evaluation_harness(tmp_path, mutate=mutate)
    trainer.state.global_step = 150
    with pytest.raises(RuntimeError):
        trainer.evaluate()
    assert calls[0] == 'saved-final'
    assert not (tmp_path / 'evaluation/latest.json').exists()
    assert not list(tmp_path.rglob('summary.json'))


def test_smoke_results_are_explicitly_not_the_full_benchmark(tmp_path):
    trainer, _ = evaluation_harness(tmp_path)
    trainer.evaluation_limit = 2
    metrics = trainer.evaluate()
    assert metrics['eval/smoke/questions'] == 2
    assert 'eval/subset/questions' not in metrics


def test_schedule_and_final_resume_cannot_update151(tmp_path):
    recipe = load('experiment.json')
    assert [i for i in range(1, 151) if needs_periodic_evaluation(i, recipe)] == list(range(20, 150, 20)) + [150]
    calls = []
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=0),
        _load_from_checkpoint=lambda value: calls.append('load'),
        evaluate=lambda: calls.append('evaluate'), train=lambda **kwargs: calls.append('train'))
    for step in (150, 20, 151):
        atomic_json(tmp_path / 'trainer_state.json', {'global_step': step})
        calls.clear()
        if step == 151:
            with pytest.raises(ValueError):
                train_or_finalize(trainer, tmp_path, 150)
        else:
            train_or_finalize(trainer, tmp_path, 150)
            assert calls == (['load', 'evaluate'] if step == 150 else ['train'])


def make_checkpoint(root, step):
    import zipfile
    path = root / f'checkpoint-{step}'
    path.mkdir()
    for name in ['adapter_config.json', 'adapter_model.safetensors', 'teacher_adapter/adapter_config.json',
                 'teacher_adapter/adapter_model.safetensors', 'trainer_state.json']:
        target = path / name
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps({'global_step': step}), encoding='utf-8')
    for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin']:
        with zipfile.ZipFile(path / name, 'w') as archive:
            archive.writestr('record', 'fixture storage only')
    seal(path, root)
    return path


def test_resume_rotation_keeps_newest_two_verified_only(tmp_path):
    atomic_json(tmp_path / 'binding.json', {'model': 'fixture'})
    for step in (20, 40, 60):
        make_checkpoint(tmp_path, step)
    corrupt = make_checkpoint(tmp_path, 80)
    (corrupt / 'optimizer.pt').write_bytes(b'corruption')
    partial = tmp_path / 'checkpoint-100'
    partial.mkdir()
    assert recover(tmp_path).name == 'checkpoint-60'
    assert not (tmp_path / 'checkpoint-20').exists()
    assert all((tmp_path / f'checkpoint-{i}').exists() for i in (40, 60, 80, 100))
    verify(tmp_path / 'checkpoint-60')
    with pytest.raises(ValueError):
        verify(corrupt)


def test_native_save_failure_does_not_remove_older_verified_saves(tmp_path):
    atomic_json(tmp_path / 'binding.json', {'model': 'fixture'})
    for step in (20, 40):
        make_checkpoint(tmp_path, step)
    class BrokenSerializer:
        def _save_checkpoint(self, *args, **kwargs):
            raise OSError('simulated disk-full')
    class Harness(VerifiedCheckpointMixin, BrokenSerializer):
        pass
    harness = Harness()
    harness.args = SimpleNamespace(output_dir=str(tmp_path), save_total_limit=2)
    harness.state = SimpleNamespace(global_step=60)
    with pytest.raises(OSError, match='disk-full'):
        harness._save_checkpoint(None)
    assert harness.args.output_dir == str(tmp_path) and harness.args.save_total_limit == 2
    assert all(verify(tmp_path / f'checkpoint-{i}') for i in (20, 40))


def test_fixed_sources_manifest_and_comparison_locks():
    verify_sources()
    lock = load('upstream.json')['shared_locks']
    assert lock == load('comparison.json')['shared_file_sha256']
    assert len(lock) == 15
    assert all(sha(ROOT / name) == digest for name, digest in lock.items())
    value = load('validation_subset.json')
    assert value['ordered_question_ids_sha256'] == '29d67a3da4b11211f8cfd1cd95c286375936c3e075a80109207c273c332b7bda'
    assert load('evaluation.json')['full_validation_updates'] == [150]


def test_occupied_port_refused_without_touching_owner():
    import socket
    from hint_searchqa.run import require_unused_port
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', 0))
        sock.listen()
        with pytest.raises(OSError):
            require_unused_port(sock.getsockname()[1])
