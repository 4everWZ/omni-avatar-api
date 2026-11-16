"""Forward Chat Completions constraints to the vLLM-Omni 0.32.0 sampler."""

from importlib.metadata import distribution


package = distribution("vllm-omni")
if package.version != "0.32.0":
    raise RuntimeError("Recheck the Omni Chat Completions constraint forwarding for this version")

path = package.locate_file("vllm_omni/entrypoints/openai/serving_chat.py")
source = path.read_text()
anchor = """                setattr(params, field_name, value)

        return params
"""
replacement = """                setattr(params, field_name, value)

        # Use vLLM's protocol conversion for response_format and structured_outputs.
        if (constraints := request.extract_structured_outputs()) is not None:
            params.structured_outputs = constraints
        return params
"""
if source.count(anchor) != 1:
    raise RuntimeError("Omni no longer matches the pinned Chat Completions sampling path")
patched = source.replace(anchor, replacement, 1)
compile(patched, str(path), "exec")
path.write_text(patched)
