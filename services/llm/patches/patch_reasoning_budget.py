"""Forward explicit Responses budgets to the text sampler."""
from importlib.metadata import distribution


def patch_source(source):
    replacements = (
        ('from vllm.sampling_params import (\n',
         'from vllm.sampling_params import (\n    ThinkingTokenBudget,\n'),
        ('    reasoning: Reasoning | None = None\n    include_reasoning: bool = Field(',
         '    reasoning: Reasoning | None = None\n    thinking_token_budget: ThinkingTokenBudget = None\n'
         '    include_reasoning: bool = Field('),
        ('            max_tokens=max_tokens,\n',
         '            max_tokens=max_tokens,\n            thinking_token_budget=self.thinking_token_budget,\n'),
    )
    for old, new in replacements:
        if source.count(old) != 1 or new in source:
            raise RuntimeError('Recheck the pinned Responses reasoning-budget conversion')
        source = source.replace(old, new, 1)
    return source


def main():
    package = distribution('vllm')
    if package.version.split('+', 1)[0] != '0.32.0':
        raise RuntimeError('Recheck reasoning budgets against this vLLM version')
    path = package.locate_file('vllm/entrypoints/openai/responses/protocol.py')
    result = patch_source(path.read_text())
    compile(result, str(path), 'exec')
    path.write_text(result)


if __name__ == '__main__':
    main()
