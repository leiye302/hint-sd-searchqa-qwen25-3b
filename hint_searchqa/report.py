"""Protocol telemetry only; native SearchEnv reward remains the score source."""
import re


def protocol(text):
    reasoning = re.search(r'<think>(.*?)</think>', text, re.S)
    clean = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    match = re.fullmatch(r'\s*<think>(.*?)</think>\s*<(search|answer)>(.*?)</\2>\s*', clean, re.S)
    return {'reasoning': bool(reasoning and reasoning[1].strip()),
            'strict': bool(match and match[1].strip() and match[3].strip()
                           and clean.count('<think>') == clean.count('</think>') == 1)}


def summarize(episodes):
    steps = [step for episode in episodes for step in episode.steps]
    flags = [protocol(step.action['content']) for step in steps]
    return {'tasks': len(episodes), 'successes': sum(episode.success for episode in episodes),
            'success_rate': sum(episode.success for episode in episodes) / max(1, len(episodes)),
            'reasoning_rate': sum(flag['reasoning'] for flag in flags) / max(1, len(flags))}
