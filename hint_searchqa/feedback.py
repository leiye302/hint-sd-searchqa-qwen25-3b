"""Necessary domain terms only; original HINT-SD Multi JSON selection contract."""


def feedback_messages(trajectory, max_steps=3):
    return [
        {'role': 'system', 'content':
         'You analyze failed search-based question-answering agent trajectories. '
         f'Identify up to {max_steps} problematic steps where the agent made mistakes '
         f'(between 1 and {max_steps} steps total, ordered earliest first). '
         'Examples include misreading retrieved evidence, confusing entities, an unsupported inference, '
         'a search query that contradicts the needed fact, a premature answer, or invalid search/answer syntax. '
         'Do not label reasonable exploration as a mistake merely because the final answer is wrong. '
         'For EACH problematic step, write a distinct correction in less than three sentences '
         'that targets only that step\'s mistake. Use the observed trajectory, not a hidden reference answer. '
         'Output valid JSON only, no other text.'},
        {'role': 'user', 'content':
         'Review this failed Search-QA attempt and identify the problematic steps. '
         'Return one (step, feedback) pair per problematic step.\n\n' + trajectory +
         '\n\nOutput format (JSON only, no other text):\n'
         '{"failures": [{"step": <1-indexed step number>, "feedback": "<correction for this step>"}, ...]}'}]
