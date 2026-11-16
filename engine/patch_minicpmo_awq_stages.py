"""Keep MiniCPM-o 4.5 AWQ on the Thinker in vLLM-Omni 0.32.0."""

from importlib.metadata import distribution


def patch_stage_source(source):
    anchor = """        # Build the vLLM config first, then use it to create the Omni config.
        try:
            model_config = super().create_model_config()
"""
    replacement = """        # OpenBMB's AWQ metadata is at checkpoint level, but only the Thinker
        # has packed weights. Clear it before base vLLM can auto-detect AWQ
        # and pass it to the Talker or Code2Wav configuration.
        if self.model_stage in {"tts", "code2wav"} and self.model_arch in {
            "MiniCPMO45OmniForConditionalGeneration",
            "MiniCPMO45OmniTTSForConditionalGeneration",
            "MiniCPMO45Code2Wav",
        }:
            self.quantization = None
            current_overrides = self.hf_overrides
            if callable(current_overrides):
                def unquantized_hf_config(config):
                    config = current_overrides(config)
                    config.quantization_config = None
                    return config

                self.hf_overrides = unquantized_hf_config
            else:
                self.hf_overrides = {**(current_overrides or {}), "quantization_config": None}

        # Build the vLLM config first, then use it to create the Omni config.
        try:
            model_config = super().create_model_config()
"""
    if source.count(anchor) != 1 or replacement in source:
        raise RuntimeError("Omni no longer matches the pinned stage model configuration path")
    return source.replace(anchor, replacement, 1)


def main():
    omni = distribution("vllm-omni")
    vllm = distribution("vllm")
    if omni.version != "0.32.0" or vllm.version.split("+", 1)[0] != "0.32.0":
        raise RuntimeError("Recheck MiniCPM-o AWQ stage scoping for these versions")

    path = omni.locate_file("vllm_omni/engine/arg_utils.py")
    patched = patch_stage_source(path.read_text(encoding="utf-8"))
    compile(patched, str(path), "exec")
    path.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
