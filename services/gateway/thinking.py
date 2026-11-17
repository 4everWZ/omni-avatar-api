"""Default Chat/Responses thinking on, preserving explicit caller overrides."""


def apply_thinking_default(payload):
    kwargs = payload.get('chat_template_kwargs')
    if kwargs is None:
        kwargs = payload['chat_template_kwargs'] = {}
    if not isinstance(kwargs, dict):
        return  # The upstream request validator owns malformed template options.
    reasoning = payload.get('reasoning')
    effort = reasoning.get('effort') if isinstance(reasoning, dict) else payload.get('reasoning_effort')
    kwargs.setdefault('enable_thinking', effort != 'none')
