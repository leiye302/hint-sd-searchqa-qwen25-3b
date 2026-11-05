"""Launch only owned HINT-SD, native QA and E5 processes; never another job's services."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from .common import ROOT, HINT, atomic_json, data_directory, devices, load, sha, verify_sources


def resolved(size, gpu_spec='0,1,2,3', smoke=False):
    ids = devices(gpu_spec)
    recipe = load('experiment.json')
    if smoke:
        recipe['updates'] = 1
    return {'method': 'HINT-SD-Multi', 'model': load('models.json')[size], 'recipe': recipe,
        'devices': {'training': ids[0], 'inference': ids[1:]},
        'training_processes': 1, 'inference_replicas': 3, 'vllm_tensor_parallel_size': 1,
        'algorithm': 'official selected-turn reverse KL; EMA Teacher; no GRPO/RETHINK/beta gate',
        'evaluation_manifest_sha256': sha(ROOT / 'configs/validation_subset.json'),
        'evaluation_protocol_sha256': sha(ROOT / 'configs/evaluation.json'),
        'retriever': 'unchanged E5+FP16 GPU-sharded Flat index, sharing the same four allocated GPUs'}


def preflight(gpus):
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    inventory = {p[0]: (p[1], int(p[2]), int(p[3])) for line in text.splitlines()
                 if (p := [x.strip() for x in line.split(',')])}
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader,nounits'], text=True)
    for gpu in gpus:
        if (gpu not in inventory or inventory[gpu][1] < 70 * 1024
                or inventory[gpu][2] < 60000 or inventory[gpu][0] in occupied):
            raise ValueError('Allocated A10080GB GPU is unavailable/occupied: ' + gpu)


def check_retriever(url, timeout=5):
    payload = json.dumps({'query': 'What is the capital of France?', 'topk': 3, 'return_scores': True}).encode()
    request = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        result = json.load(response)
    if len(result.get('result', [])) != 1 or len(result['result'][0]) != 3:
        raise ValueError('Expected native one-query top3 retrieval response')
    if any(not isinstance(row.get('document', {}).get('contents'), str) for row in result['result'][0]):
        raise ValueError('Native retrieval document schema differs')


def require_unused_port(port):
    with socket.socket() as stream:
        stream.bind(('0.0.0.0', port))


def owned_listener(process, port):
    """Native uvicorn has one process; verify its own listening FD, not another server."""
    sockets = set()
    for fd in (Path('/proc') / str(process.pid) / 'fd').iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith('socket:['):
            sockets.add(target[8:-1])
    table = Path('/proc') / str(process.pid) / 'net/tcp'
    for line in table.read_text(encoding='ascii').splitlines()[1:]:
        fields = line.split()
        if int(fields[1].split(':')[1], 16) == port and fields[3] == '0A' and fields[9] in sockets:
            return
    raise ValueError('Owned native retriever has not bound its listening socket')


def wait_ready(process, probe, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Owned service exited before readiness; inspect its retained log')
        try:
            probe()
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError('Owned service did not become ready; no training update started')


def stop_owned(process):
    if process is None:
        return
    # Every process below starts a NEW session; signal only that owned group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def launch(args):
    config = resolved(args.size, args.gpus, args.smoke)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if sys.platform != 'linux':
        raise RuntimeError('Use the documented Linux/CUDA runtimes')
    verify_sources()
    from data import verify as verify_data
    from retrieval import verify as verify_retrieval
    directory, retrieval = args.data_dir.resolve(), args.retrieval_dir.resolve()
    verify_data(directory)
    retrieval_receipt = verify_retrieval(retrieval)
    output = (args.output or ROOT / 'runs' / ('hint_sd_' + args.size
        + ('_smoke' if args.smoke else '') + ('_eval' if args.command == 'eval' else ''))).resolve()
    if output == ROOT or not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('Output must be a child of this checkout\'s runs directory')
    binding = {'model': config['model'], 'recipe': config['recipe'],
        'data_directory': str(directory), 'retrieval_directory': str(retrieval),
        'data_receipt_sha256': sha(directory / 'data_receipt.json'),
        'subset_receipt_sha256': sha(directory / 'validation_subset_receipt.json'),
        'retrieval_receipt': retrieval_receipt,
        'evaluation_manifest_sha256': config['evaluation_manifest_sha256'],
        'evaluation_protocol_sha256': config['evaluation_protocol_sha256'],
        'upstream_lock_sha256': sha(ROOT / 'configs/upstream.json'),
        'adapter_code': {p.name: sha(p) for p in sorted((ROOT / 'hint_searchqa').glob('*.py'))}}
    resume = None
    if args.resume:
        if json.loads((output / 'binding.json').read_text(encoding='utf-8')) != binding:
            raise ValueError('Resume model/data/config/source identity differs')
        from .checkpoint_io import recover
        resume = recover(output)
    elif output.exists():
        raise FileExistsError('Output already exists; use --resume or a new --output')
    model = ROOT / 'models' / config['model']['model_id'].split('/')[-1]
    receipt = json.loads((model / 'download_receipt.json').read_text(encoding='utf-8'))
    if any(receipt[key] != config['model'][key] for key in ('model_id', 'revision')):
        raise ValueError('Expected the pinned original base model')
    if {p.name for p in model.iterdir() if p.is_file() and p.name != 'download_receipt.json'} != set(receipt['files']):
        raise ValueError('Model file set differs from pinned receipt')
    if any(sha(model / name) != item['sha256'] for name, item in receipt['files'].items()):
        raise ValueError('Model file hash differs')
    if args.eval_checkpoint:
        from .checkpoint_io import verify
        verify(args.eval_checkpoint)
        source_binding = json.loads((args.eval_checkpoint.parent / 'binding.json').read_text(encoding='utf-8'))
        if source_binding['model'] != config['model']:
            raise ValueError('Evaluation checkpoint belongs to a different backbone/revision')
    training_python = HINT / '.venv/bin/python'
    env_python = Path(os.environ.get('HINTSD_ENV_PYTHON', ROOT / '.env-searchqa/bin/python'))
    retriever_python = Path(os.environ.get('HINTSD_RETRIEVER_PYTHON', ROOT / '.env-retriever/bin/python'))
    if not all(path.is_file() for path in (training_python, env_python, retriever_python)):
        raise ValueError('Prepare the documented HINT-SD, QA CPU and native FAISS/E5 runtimes')
    preflight(devices(args.gpus))
    if args.port == args.retriever_port:
        raise ValueError('Environment and retrieval services require distinct ports')
    require_unused_port(args.port)
    require_unused_port(args.retriever_port)
    if not args.resume:
        output.mkdir(parents=True)
        atomic_json(output / 'binding.json', binding)
    import fcntl
    with (output / 'controller.lock').open('a+') as lock_file, ExitStack() as streams:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This run already has an active controller') from None
        base_env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / 'src')]),
            TOKENIZERS_PARALLELISM='false', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
            HINTSD_QA_DATA_DIR=str(directory))
        for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            base_env.pop(name, None)
        search_url = f'http://127.0.0.1:{args.retriever_port}/retrieve'
        retriever = service = trainer = None
        try:
            retriever_log = streams.enter_context((output / 'retriever.log').open('a'))
            retriever = subprocess.Popen([str(retriever_python), str(ROOT / 'src/retrieval.py'), '--serve',
                '--directory', str(retrieval), '--gpus', args.gpus, '--port', str(args.retriever_port)],
                cwd=ROOT, env=base_env, stdout=retriever_log, stderr=subprocess.STDOUT, start_new_session=True)
            def retrieval_ready():
                owned_listener(retriever, args.retriever_port)
                check_retriever(search_url)
            wait_ready(retriever, retrieval_ready)
            env_log = streams.enter_context((output / 'environment.log').open('a'))
            service = subprocess.Popen([str(env_python), '-m', 'hint_searchqa.environment', '--port', str(args.port),
                '--search-url', search_url], cwd=ROOT, env=dict(base_env, CUDA_VISIBLE_DEVICES=''),
                stdout=env_log, stderr=subprocess.STDOUT, start_new_session=True)
            from .environment import Client
            client = Client(f'http://127.0.0.1:{args.port}')
            def environment_ready():
                if client.call('health', timeout=2).get('pid') != service.pid:
                    raise ValueError('Port belongs to a different environment process')
            wait_ready(service, environment_ready, timeout=120)
            command = [str(training_python), '-m', 'hint_searchqa.train', '--size', args.size,
                       '--environment-url', client.url, '--output', str(output)]
            for flag, value in [('--resume', resume), ('--eval-checkpoint', args.eval_checkpoint),
                                ('--eval-scope', args.eval_scope)]:
                if value is not None:
                    command += [flag, str(value)]
            if args.eval_base:
                command.append('--eval-base')
            if args.smoke:
                command.append('--smoke')
            train_log = streams.enter_context((output / 'training.log').open('a'))
            trainer = subprocess.Popen(command, cwd=ROOT,
                env=dict(base_env, CUDA_VISIBLE_DEVICES=config['devices']['training'],
                         HINTSD_INFERENCE_GPUS=','.join(config['devices']['inference'])),
                stdout=train_log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(output / 'processes.json', {'launcher': os.getpid(), 'retriever': retriever.pid,
                'environment': service.pid, 'trainer': trainer.pid, 'gpus': config['devices']})
            print('Started HINT-SD', args.size, '; retained log:', output / 'training.log', flush=True)
            if trainer.wait():
                raise RuntimeError('Training/evaluation failed; logs and partial artifacts retained')
        finally:
            stop_owned(trainer)
            stop_owned(service)
            stop_owned(retriever)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'eval'])
    parser.add_argument('--size', choices=['1.5b', '3b'], required=True)
    parser.add_argument('--gpus', default=os.environ.get('HINTSD_GPUS', '0,1,2,3'))
    parser.add_argument('--port', type=int, default=18760)
    parser.add_argument('--retriever-port', type=int, default=8000)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/searchqa')
    parser.add_argument('--retrieval-dir', type=Path, default=ROOT / 'data/retrieval')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--eval-checkpoint', type=Path)
    parser.add_argument('--eval-base', action='store_true')
    parser.add_argument('--eval-scope', choices=['subset', 'full'])
    args = parser.parse_args()
    if args.command == 'eval' and not (args.eval_base or args.eval_checkpoint):
        parser.error('eval requires --eval-base or --eval-checkpoint')
    if args.command != 'eval' and (args.eval_base or args.eval_checkpoint or args.eval_scope):
        parser.error('Evaluation options require the eval command')
    if args.eval_base and args.eval_checkpoint or args.resume and args.command == 'eval':
        parser.error('Conflicting evaluation/resume options')
    launch(args)


if __name__ == '__main__':
    main()
