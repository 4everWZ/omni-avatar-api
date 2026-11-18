"""Initialize reasoning boundaries for the MiniCPM-o thinker."""
from importlib.metadata import distribution


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
    config = package.locate_file('vllm/config/vllm.py')
    result = patch_minicpmo_config(config.read_text())
    compile(result, str(config), 'exec')
    config.write_text(result)


if __name__ == '__main__':
    main()
