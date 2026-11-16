"""Adapt the pinned Omni Chat sampler and MiniCPM-o reasoning boundaries."""

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
        elif (
            self.model_config.hf_config.model_type == "minicpmo"
            and self._effective_chat_template_kwargs(request).get("enable_thinking") is True
            and (request.modalities or ["text"]) == ["text"]
            and not (request.tools or request.stop or request.stop_token_ids)
        ):
            from vllm.sampling_params import StructuredOutputsParams

            tokenizer = self.renderer.get_tokenizer()
            vocab = tokenizer.get_vocab() if tokenizer is not None else {}
            markers = ("<think>", "</think>", "<|im_end|>", "<|endoftext|>")
            if all(marker in vocab for marker in markers):
                start, end, *stops = [vocab[marker] for marker in markers]
                # Allow a direct reply or a closed thought followed by a reply.
                # Token-ID boundaries match the parser; stop is legal only
                # after a reply. The caller's generation budget still applies.
                excluded = [start, end, *stops]
                thought = {
                    "type": "tag",
                    "begin": {"type": "token", "token": start},
                    "content": {"type": "any_tokens", "exclude_tokens": [start, *stops]},
                    "end": {"type": "token", "token": end},
                }
                structure = {
                    "type": "structural_tag",
                    "format": {
                        "type": "sequence",
                        "elements": [
                            {"type": "optional", "content": thought},
                            {"type": "exclude_token", "exclude_tokens": excluded},
                            {"type": "any_tokens", "exclude_tokens": excluded},
                        ],
                    },
                }
                params.structured_outputs = StructuredOutputsParams(structural_tag=json.dumps(structure))
        return params
"""
if source.count(anchor) != 1:
    raise RuntimeError("Omni no longer matches the pinned Chat Completions sampling path")
patched = source.replace(anchor, replacement, 1)

anchor = """            tool_parser = self.parser_cls.tool_parser_cls if self.parser_cls is not None else None
"""
replacement = """            # MiniCPM-o may either open a think block explicitly or answer
            # directly. Keep the tags out of content while leaving the sampler
            # and structured-output grammar unchanged.
            if reasoning_parser is None and entry_stage_id == 0 and self.model_config.hf_config.model_type == "minicpmo":
                from dataclasses import replace as replace_parser_config
                from vllm.parser.qwen3 import ParserState, Transition, qwen3_config
                from vllm.reasoning.qwen3_engine_reasoning_parser import Qwen3ParserReasoningAdapter

                config = qwen3_config(thinking=False)
                config = replace_parser_config(
                    config,
                    transitions={
                        **config.transitions,
                        (ParserState.CONTENT, "THINK_START"): Transition(ParserState.REASONING, ()),
                    },
                )
                reasoning_parser = Qwen3ParserReasoningAdapter(tokenizer, parser_engine_config=config)

            tool_parser = self.parser_cls.tool_parser_cls if self.parser_cls is not None else None
"""
if patched.count(anchor) != 1:
    raise RuntimeError("Omni no longer matches the pinned MiniCPM-o Chat parser path")
patched = patched.replace(anchor, replacement, 1)
compile(patched, str(path), "exec")
path.write_text(patched)
