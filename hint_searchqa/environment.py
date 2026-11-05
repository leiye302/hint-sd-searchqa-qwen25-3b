"""Owned loopback service over SDAR's unchanged native SearchEnv and formatter."""
import argparse
import ast
import copy
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
import signal
import threading
import uuid
from types import SimpleNamespace
from .common import SDAR, load, verify_sources


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_class(path, name, namespace, methods=None):
    node = copy.deepcopy(next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                              if isinstance(n, ast.ClassDef) and n.name == name))
    node.bases = [ast.Name(id='object', ctx=ast.Load())]
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace[name]


@lru_cache(maxsize=1)
def native_parts():
    import sys
    sys.path.insert(0, str(SDAR))
    from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import SearchEnv
    from agent_system.environments.env_package.search.projection import search_projection
    sync = native_class(SDAR / 'agent_system/environments/env_package/search/envs.py',
        'SearchMultiProcessEnv', {}, {'_sync_reset', '_sync_step'})
    prompts = module_at('hint_searchqa_native_prompts', SDAR / 'agent_system/environments/prompts/search.py')
    memory = native_class(SDAR / 'agent_system/memory/memory.py', 'SearchMemory', {})
    formatter = native_class(SDAR / 'agent_system/environments/env_manager.py', 'SearchEnvironmentManager', {
        'SEARCH_TEMPLATE': prompts.SEARCH_TEMPLATE,
        'SEARCH_TEMPLATE_NO_HIS': prompts.SEARCH_TEMPLATE_NO_HIS}, {'build_text_obs'})
    class SingleSearchWorker:
        """One native SearchEnv, exact native sync methods; HTTP provides concurrency.

        Avoid allocating an otherwise-unused asyncio loop/executor for each of
        512 validation questions. No native prompt, transition or scorer changes.
        """
        def __init__(self, seed, env_num, group_n, is_train, env_config):
            del seed, is_train
            if env_num != 1 or group_n != 1:
                raise ValueError('Expected a single independent SearchEnv per instance')
            self.env = SearchEnv(env_config.search)
            self.sync = sync()
            self.sync.max_steps = env_config.max_steps
        def reset(self, kwargs):
            obs, info = self.sync._sync_reset(self.env, kwargs[0])
            return [obs], [info]
        def step(self, actions):
            obs, reward, done, info = self.sync._sync_step(self.env, actions[0])
            return [obs], [reward], [done], [info]
        def close(self):
            self.env.close()
    return SingleSearchWorker, formatter, memory, search_projection


def policy_messages(observation, admissible=None, task=None, history=None, history_length=4):
    del admissible, task, history, history_length
    return [{'role': 'user', 'content': observation}]


class Game:
    def __init__(self, row, max_steps, search_url):
        from omegaconf import OmegaConf
        from common import question_id
        if row['split'] not in ('train', 'validation') or question_id(row['env_kwargs']) != row['task_id']:
            raise ValueError('Search-QA task identity/split differs')
        Worker, Formatter, Memory, self.projection = native_parts()
        config = OmegaConf.create({'max_steps': max_steps, 'search': {
            'search_url': search_url, 'topk': 3, 'timeout': 60, 'log_requests': False}})
        # This is the unchanged native threaded Search worker, not a rewritten scorer.
        self.worker = Worker(seed=row['seed'], env_num=1, group_n=1,
                             is_train=row['split'] == 'train', env_config=config)
        try:
            raw, infos = self.worker.reset([row['env_kwargs']])
            self.manager = Formatter()
            self.manager.tasks = raw
            self.manager.config = SimpleNamespace(env=SimpleNamespace(history_length=4))
            self.manager.memory = Memory()
            self.manager.memory.reset(batch_size=1)
            self.info = infos[0]
            self.raw_observation = raw[0]
            self.observation = self.manager.build_text_obs(raw, init=True)[0]
            self.task_id, self.task = row['task_id'], row['env_kwargs']['question']
            self.lock = threading.Lock()
        except BaseException:
            self.worker.close()
            raise

    def state(self):
        return {'observation': self.observation, 'raw_observation': self.raw_observation,
                'admissible_commands': [], 'task': self.task, 'history': [], 'task_id': self.task_id}

    def step(self, response):
        commands, valid = self.projection([response])
        raw, rewards, dones, infos = self.worker.step(commands)
        self.manager.memory.store({'search': commands, 'information': raw})
        self.raw_observation = raw[0]
        self.observation = self.manager.build_text_obs(raw)[0]
        self.info = infos[0]
        return dict(self.state(), command=commands[0], action_valid=bool(valid[0]),
                    executable=bool(valid[0]), won=bool(self.info['won']),
                    terminated=bool(dones[0]), reward=float(rewards[0]))

    def close(self):
        self.worker.close()


class Service:
    def __init__(self, search_url='http://127.0.0.1:8000/retrieve', max_steps=4):
        self.games, self.lock = {}, threading.Lock()
        self.search_url, self.max_steps = search_url, max_steps

    def call(self, operation, value):
        if operation == 'health':
            return {'ready': True, 'pid': os.getpid()}
        if operation == 'create':
            game = Game(value['task'], self.max_steps, self.search_url)
            identity = uuid.uuid4().hex
            with self.lock:
                self.games[identity] = game
            return dict(game.state(), instance_id=identity)
        identity = value['instance_id']
        with self.lock:
            game = self.games.get(identity)
        if operation == 'release':
            if game is not None:
                with game.lock:
                    game.close()
                with self.lock:
                    self.games.pop(identity, None)
            return {'released': True}
        if game is None:
            raise ValueError('Unknown/released environment instance')
        if operation == 'step':
            with game.lock:
                return game.step(str(value['response']))
        raise ValueError('Unknown operation')

    def close(self):
        for identity in list(self.games):
            self.call('release', {'instance_id': identity})


class Client:
    def __init__(self, url):
        if not url.startswith('http://127.0.0.1:'):
            raise ValueError('Environment service must use loopback')
        self.url = url.rstrip('/')

    def call(self, operation, *, timeout=300, **payload):
        import urllib.request
        request = urllib.request.Request(self.url + '/' + operation, data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18760)
    parser.add_argument('--search-url', default='http://127.0.0.1:8000/retrieve')
    args = parser.parse_args()
    verify_sources()
    service = Service(args.search_url)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 <= length <= 4 * 1024**2:
                    raise ValueError('Invalid request size')
                result = service.call(self.path.lstrip('/'), json.loads(self.rfile.read(length) or b'{}'))
                encoded = json.dumps(result).encode()
                self.send_response(200)
            except Exception as error:
                encoded = json.dumps({'error': type(error).__name__, 'message': str(error)}).encode()
                self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    def terminate(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    try:
        print('Native Search-QA CPU service ready on loopback.', flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
