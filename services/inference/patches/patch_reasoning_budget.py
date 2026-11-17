"""Forward explicit reasoning budgets through the pinned Responses protocol."""
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


def patch_minicpmo_config(source):
    anchor = '        if self.reasoning_config is not None and self.model_config is not None:\n'
    if source.count(anchor) != 1 or '# MiniCPM-o Chat budget boundaries.' in source:
        raise RuntimeError('Recheck the pinned MiniCPM-o reasoning-config initialization')
    return source.replace(anchor, (
        '        # MiniCPM-o Chat budget boundaries. Native duplex sets no thinking budget.\n'
        '        if (self.reasoning_config is None and self.model_config is not None\n'
        '                and getattr(self.model_config, "stage_id", 0) == 0\n'
        '                and getattr(self.model_config.hf_config, "model_type", None) == "minicpmo"):\n'
        '            from vllm.config.reasoning import ReasoningConfig\n'
        '            self.reasoning_config = ReasoningConfig(\n'
        '                reasoning_start_str="<think>", reasoning_end_str="</think>")\n'
        '            logger.info("MiniCPM-o thinker reasoning-budget boundaries initialized")\n\n') + anchor, 1)


def main():
    package = distribution('vllm')
    if package.version.split('+', 1)[0] != '0.32.0':
        raise RuntimeError('Recheck reasoning budgets against this vLLM version')
    path = package.locate_file('vllm/entrypoints/openai/responses/protocol.py')
    result = patch_source(path.read_text())
    compile(result, str(path), 'exec')
    path.write_text(result)
    config = package.locate_file('vllm/config/vllm.py')
    result = patch_minicpmo_config(config.read_text())
    compile(result, str(config), 'exec')
    config.write_text(result)


if __name__ == '__main__':
    main()
