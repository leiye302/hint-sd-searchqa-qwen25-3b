"""Run unmodified HINT-SD optimizer/loss/EMA with native Search-QA interaction."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from .common import ROOT, HINT, atomic_json, checkpoint_complete, load, data_directory, task_row, verify_sources


def native_arguments(recipe, model_path, output):
    """Resolve the author's own argument parser and TrainingArguments builder."""
    from src.train.train import parse_args
    options = {
        '--model-name-or-path': str(model_path), '--backend': 'bfcl', '--variant': 'multi',
        '--train-file': str(data_directory() / 'train.parquet'),
        '--eval-file': str(data_directory() / 'validation_subset.parquet'),
        '--output-dir': str(output), '--seed': recipe['seed'], '--report-to': 'none',
        '--max-steps': recipe['updates'], '--learning-rate': recipe['learning_rate'],
        '--batch-size': recipe['batch_size'], '--gradient-accumulation-steps': recipe['gradient_accumulation_steps'],
        '--rollout-n': recipe['rollout_n'], '--temperature': recipe['temperature'],
        '--feedback-temperature': recipe['feedback_temperature'], '--max-model-len': recipe['max_model_len'],
        '--max-completion-length': recipe['max_completion_length'], '--feedback-max-tokens': recipe['feedback_max_tokens'],
        '--ema-decay': recipe['ema_decay'], '--distillation-topk': recipe['distillation_topk'],
        '--distillation-microbatch-size': recipe['distillation_microbatch_size'],
        '--max-distillation-tokens-per-batch': recipe['max_distillation_tokens_per_batch'],
        '--lora-rank': recipe['lora_rank'], '--lora-alpha': recipe['lora_alpha'],
        '--lora-dropout': recipe['lora_dropout'], '--lora-target-modules': ','.join(recipe['lora_target_modules']),
        '--warmup-steps': recipe['warmup_ratio'], '--save-steps': recipe['save_every'],
        '--save-total-limit': recipe['keep_checkpoints'], '--logging-steps': 1,
        '--feedback-multi-step-max': recipe['feedback_max_steps'], '--eval-strategy': 'no',
    }
    previous = sys.argv
    try:
        sys.argv = ['hint-sd-searchqa'] + [str(item) for pair in options.items() for item in pair]
        return parse_args()
    finally:
        sys.argv = previous


def needs_periodic_evaluation(step, recipe):
    return 0 < step <= recipe['updates'] and (step % recipe['eval_every'] == 0 or step == recipe['updates'])


def train_or_finalize(trainer, resume, updates):
    """Restore final weights and retry evaluation only; never run update151."""
    if resume:
        step = json.loads((resume / 'trainer_state.json').read_text(encoding='utf-8'))['global_step']
        if step > updates:
            raise ValueError('Resume checkpoint exceeds this recipe\'s update budget')
        if step == updates:
            trainer._load_from_checkpoint(str(resume))
            trainer.state.global_step = updates
            trainer.evaluate()
            return
    else:
        trainer.evaluate()
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=['1.5b', '3b'], required=True)
    parser.add_argument('--environment-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--eval-checkpoint', type=Path)
    parser.add_argument('--eval-base', action='store_true')
    parser.add_argument('--eval-scope', choices=['subset', 'full'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    verify_sources()
    sys.path.insert(0, str(HINT))
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from src.train.train import apply_dual_adapter_lora, build_training_args
    from .trainer import SearchQATrainer
    if torch.cuda.device_count() != 1:
        raise ValueError('Each trainer must see exactly its one allocated training GPU')
    recipe = load('experiment.json')
    if args.smoke:
        recipe['updates'] = 1
    model_spec = load('models.json')[args.size]
    model_path = ROOT / 'models' / model_spec['model_id'].split('/')[-1]
    native = native_arguments(recipe, model_path, args.output)
    set_seed(recipe['seed'])
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = apply_dual_adapter_lora(model, native)
    training_args = build_training_args(native, has_eval_dataset=False)
    rows = Dataset.from_parquet(str(data_directory() / 'train.parquet')).select_columns(['env_kwargs', 'extra_info'])
    if len(rows) != recipe['training_pool_size']:
        raise ValueError('Expected unchanged169615-question training pool')
    trainer = SearchQATrainer(model=model, args=training_args,
        train_dataset=rows, processing_class=tokenizer,
        data_collator=lambda examples: {'rows': [task_row(item, item['extra_info']['index'], 'train') for item in examples]},
        recipe=recipe, environment_url=args.environment_url,
        distillation_topk=recipe['distillation_topk'], distillation_add_tail=recipe['distillation_add_tail'],
        max_distillation_tokens_per_batch=recipe['max_distillation_tokens_per_batch'],
        distillation_microbatch_size=recipe['distillation_microbatch_size'], ema_decay=recipe['ema_decay'])
    trainer.evaluation_limit = 2 if args.smoke else None
    trainer.evaluation_only = bool(args.eval_checkpoint or args.eval_base)
    trainer.evaluation_mode = (args.eval_scope or 'full') if trainer.evaluation_only else 'auto'

    class Records(TrainerCallback):
        def on_log(self, cb_args, state, control, logs=None, **kwargs):
            with (args.output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'update': state.global_step, **(logs or {})}, allow_nan=False) + '\n')

        def on_step_end(self, cb_args, state, control, **kwargs):
            if needs_periodic_evaluation(state.global_step, recipe):
                trainer.evaluate()
            return control

        def on_save(self, cb_args, state, control, **kwargs):
            checkpoint = args.output / f'checkpoint-{state.global_step}'
            from .checkpoint_io import verify
            verify(checkpoint)
            return control

    trainer.add_callback(Records())
    try:
        if args.eval_checkpoint:
            trainer._load_from_checkpoint(str(args.eval_checkpoint))
            state_path = args.eval_checkpoint / 'trainer_state.json'
            trainer.state.global_step = json.loads(state_path.read_text(encoding='utf-8'))['global_step'] if state_path.exists() else 0
        if args.eval_checkpoint or args.eval_base:
            trainer.evaluate()
            return
        train_or_finalize(trainer, args.resume, recipe['updates'])
        final_checkpoint = args.output / f'checkpoint-{trainer.state.global_step}'
        if not (final_checkpoint / 'complete.json').exists():
            trainer._save_checkpoint(trainer.model, trial=None)
            Records().on_save(training_args, trainer.state, trainer.control)
        checkpoint_complete(final_checkpoint)
        trainer.save_model(str(args.output / 'final_adapter'))
        latest_path = args.output / 'evaluation/latest.json'
        latest = json.loads(latest_path.read_text(encoding='utf-8')) if latest_path.is_file() else {}
        completed_summary = args.output / latest.get('path', '') / 'summary.json'
        if (latest.get('update') != trainer.state.global_step or not completed_summary.is_file()):
            trainer.evaluate()
        atomic_json(args.output / 'completed.json', {'updates': trainer.state.global_step,
                    'method': 'HINT-SD-Multi', 'checkpoint': final_checkpoint.name,
                    'base_model': model_spec, 'no_grpo_no_rethink_no_external_analyzer': True})
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
