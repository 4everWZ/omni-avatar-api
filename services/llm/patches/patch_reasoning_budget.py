"""Forward explicit Responses budgets to the text sampler."""
from importlib.metadata import distribution
import re


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


def verify_native(source):
    """Verify vLLM natively defines thinking_token_budget on ResponsesRequest."""
    if not re.search(r'^[ \t]*thinking_token_budget\s*:', source, re.MULTILINE):
        raise RuntimeError('Native thinking_token_budget missing from Responses protocol')


def main():
    package = distribution('vllm')
    version = package.version.split('+', 1)[0]
    if version not in ('0.32.0', '0.34.0'):
        raise RuntimeError('Recheck reasoning budgets against this vLLM version')
    path = package.locate_file('vllm/entrypoints/openai/responses/protocol.py')
    source = path.read_text()
    if version == '0.34.0':
        verify_native(source)
        return
    result = patch_source(source)
    compile(result, str(path), 'exec')
    path.write_text(result)


if __name__ == '__main__':
    main()
