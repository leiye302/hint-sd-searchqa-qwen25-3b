"""Subclass the pinned trainer ONLY at environment, inference transport and evaluation boundaries."""
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from src.train.trainer import Trainer as OfficialTrainer
from src.train.appworld_feedback import parse_multi_step_failure_json
from src.train.bfcl_feedback import render_episode_for_detection
from .common import atomic_json, task_rows, evaluation_spec, load
from .engine import AdapterPublisher, RemotePool
from .checkpoint_io import VerifiedCheckpointMixin
from .feedback import feedback_messages
from .rollout import run_rollouts


class SearchQATrainer(VerifiedCheckpointMixin, OfficialTrainer):
    def __init__(self, *args, recipe, environment_url, **kwargs):
        self.recipe = recipe
        self.environment_url = environment_url
        super().__init__(*args, **kwargs)

    def _init_vllm_engine(self):
        self.publisher = AdapterPublisher(Path(self.args.output_dir) / 'inference_adapters')
        return RemotePool(self.model.config._name_or_path, os.environ['HINTSD_INFERENCE_GPUS'].split(','),
                            self.recipe['max_model_len'], self.recipe['vllm_gpu_memory_utilization'],
                            self.args.seed)

    def _sync_adapter_weights_to_vllm(self, adapter_name):
        # Equivalent active-adapter weights, without the upstream private V0 driver_worker access.
        self.llm.adapter = self.publisher.publish(self, adapter_name)

    def close(self):
        self.llm.close()

    def _prepare_inputs(self, generation_batch):
        if 'prompt_ids' in generation_batch:
            return generation_batch
        batch = run_rollouts(generation_batch['rows'], self, rollout_n=self.recipe['rollout_n'],
                             temperature=self.recipe['temperature'])
        output = Path(self.args.output_dir) / 'rollouts'
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            for episode in batch.episodes:
                stream.write(json.dumps({'update_before': self.state.global_step, **episode.to_dict()}) + '\n')
        metrics = {**batch.metrics, 'rollout/mean_score': sum(batch.scores) / max(1, len(batch.scores))}
        # Selection, success skipping, selected completion masks and tensor construction are NATIVE.
        return self._prepare_bfcl_multi_step_inputs(batch, rollout_metrics=metrics)

    def _detect_bfcl_failure_steps_multi(self, episodes, env_feedback, *, max_steps=3):
        results = [None] * len(episodes)
        prompts, indices = [], []
        for index, episode in enumerate(episodes):
            if episode.success or not episode.steps:
                continue
            trajectory = render_episode_for_detection(episode, env_feedback[index],
                max_chars=self.recipe['feedback_max_trajectory_chars'])
            prompts.append(self._apply_chat_template(feedback_messages(trajectory, max_steps), True))
            indices.append(index)
        generated = self._sample_teacher_model_texts(prompts, max_new_tokens=self.recipe['feedback_max_tokens'])
        if len(generated) != len(indices):
            raise RuntimeError('Feedback output count differs from failed rollout count')
        audit = []
        for index, raw in zip(indices, generated, strict=True):
            parsed = parse_multi_step_failure_json(raw, len(episodes[index].steps), max_steps=max_steps)
            results[index] = parsed
            audit.append({'task_id': episodes[index].task_id, 'raw': raw, 'selected': parsed})
        with (Path(self.args.output_dir) / 'feedback.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'update_before': self.state.global_step, 'feedback': audit}) + '\n')
        return results

    def _build_bfcl_per_step_teacher_prompt(self, episode, failure_step_index, hint):
        # Use exactly the actual (possibly left-truncated) Student prefix, not a reconstructed
        # full trajectory. This special message is rendered by the adapter below, not by Qwen.
        return [{'role': '__exact_hindsight__', 'content': hint,
                 'prefix_ids': episode.steps[failure_step_index].agent_metadata['prompt_token_ids']}]

    def _apply_chat_template(self, messages, add_generation_prompt):
        if messages and messages[0]['role'] == '__exact_hindsight__':
            item = messages[0]
            # Qwen2.5 has no thinking toggle. Infer its native assistant header, never hand-tokenize tags.
            probe = [{'role': 'user', 'content': 'prefix-probe'}]
            closed = super()._apply_chat_template(probe, False)
            opened = super()._apply_chat_template(probe, True)
            if not opened.startswith(closed):
                raise ValueError('Unsupported non-prefix-stable chat template')
            header = opened[len(closed):]
            raw = self.processing_class.decode(item['prefix_ids'], skip_special_tokens=False)
            if not header or not raw.endswith(header):
                raise ValueError('Recorded Student prefix does not end at the native assistant boundary')
            feedback = [{'role': 'user', 'content': 'Consider the following feedback to solve this step:\n\n' + item['content']}]
            # Rendering feedback alone can insert a SECOND implicit default system message.
            # Derive only the appended user-feedback suffix from one shared native probe.
            combined = super()._apply_chat_template(probe + feedback, add_generation_prompt)
            if not combined.startswith(closed):
                raise ValueError('Unsupported non-prefix-stable feedback chat template')
            return raw[:-len(header)] + combined[len(closed):]
        return super()._apply_chat_template(messages, add_generation_prompt)

    def ensure_final_checkpoint(self):
        """Publish complete update150 before the potentially long full benchmark."""
        from .checkpoint_io import verify
        checkpoint = Path(self.args.output_dir) / f'checkpoint-{self.state.global_step}'
        if checkpoint.exists():
            verify(checkpoint)
        else:
            self._save_checkpoint(self.model, trial=None)
            verify(checkpoint)
        return checkpoint

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        del eval_dataset, ignore_keys, metric_key_prefix
        from evaluation_contract import validation_scope, summarize_questions
        scope = validation_scope(int(self.state.global_step), getattr(self, 'evaluation_mode', 'auto'))
        if (scope == 'full' and not getattr(self, 'evaluation_only', False)
                and self.state.global_step == self.recipe['updates']):
            self.ensure_final_checkpoint()
        started = time.monotonic()
        expected, original_indices = evaluation_spec(scope)
        rows = task_rows(scope)
        limit = getattr(self, 'evaluation_limit', None)
        smoke = limit is not None and limit < len(rows)
        if smoke:
            rows = rows[:limit]
            expected = {'question_ids': expected['question_ids'][:limit],
                'source_counts': dict(Counter(row['data_source'] for row in rows))}
        expected_ids = expected['question_ids']
        if [row['task_id'] for row in rows] != expected_ids:
            raise RuntimeError('Evaluation task identity/order changed')
        label = 'smoke' if smoke else scope
        output = (Path(self.args.output_dir) / 'evaluation' / f'update_{self.state.global_step:03d}'
                  / ('attempt-' + uuid.uuid4().hex[:8]))
        results = []
        batch_size = self.recipe['eval_batch_size']
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            batch = run_rollouts(selected, self, rollout_n=1, temperature=0.)
            if len(batch.episodes) != len(selected):
                raise RuntimeError('Evaluation missed input questions')
            records = []
            for index, (episode, row) in enumerate(zip(batch.episodes, selected, strict=True), start):
                metadata = episode.run_metadata
                if (episode.task_id != row['task_id'] or metadata['qa_index'] != index
                        or metadata['data_source'] != row['data_source']):
                    raise RuntimeError('Actual rollout question identity/order changed')
                result = {'index': index, 'question_id': row['task_id'], 'data_source': row['data_source'],
                          'episode_reward': float(episode.final_reward)}
                results.append(result)
                records.append({**result, 'original_test_index': original_indices[index],
                                'episode': episode.to_dict()})
            atomic_json(output / f'batch_{start // batch_size:03d}.json', records)
        summary = summarize_questions(results, expected_ids, expected['source_counts'])
        elapsed = time.monotonic() - started
        summary.update({'status': 'completed', 'scope': label, 'update': int(self.state.global_step),
            'wall_seconds': elapsed, 'questions_per_second': len(results) / elapsed if elapsed else None,
            'temperature': 0., 'rollouts_per_question': 1, 'student_only': True,
            'max_steps': 4, 'history_length': 4, 'max_prompt_tokens': 4096, 'max_response_tokens': 512,
            'monitor_target_seconds': 600 if scope == 'subset' else None,
            'monitor_target_is_not_a_timeout': True,
            'native_metrics_unchanged': 'Not applicable: HINT-SD uses its original Trainer, not SDAR Trainer metrics.'})
        atomic_json(output / 'summary.json', summary)
        atomic_json(Path(self.args.output_dir) / 'evaluation/latest.json',
                    {'path': str(output.relative_to(self.args.output_dir)), 'scope': label,
                     'update': int(self.state.global_step)})
        metrics = {f'eval/{label}/question_micro_em': summary['question_micro_em'],
                   f'eval/{label}/dataset_macro_em': summary['dataset_macro_em'],
                   f'eval/{label}/questions': len(results), f'eval/{label}/wall_seconds': elapsed}
        metrics.update({f'eval/{label}/{source}_question_em': record['em']
                        for source, record in summary['by_dataset'].items()})
        self.log(metrics)
        return metrics
