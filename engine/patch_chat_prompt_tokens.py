"""Expose exact text prompt counts from the pinned vLLM-Omni Chat renderer."""

from importlib.metadata import distribution


def _replace_once(source, old, new):
    if source.count(old) != 1 or new in source:
        raise RuntimeError("Omni no longer matches the pinned Chat prompt count hook")
    return source.replace(old, new, 1)


def patch_chat_source(source):
    source = _replace_once(
        source,
        "from dataclasses import fields, is_dataclass\n",
        "from dataclasses import fields, is_dataclass, replace\n",
    )
    source = _replace_once(
        source,
        """    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
""",
        """    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
        *,
        count_only: bool = False,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse | dict[str, int]:
""",
    )
    source = _replace_once(
        source,
        "        normalized_extra_args: dict[str, object] = {}\n        diffusion_request_args: dict[str, Any] = {}\n",
        """        if count_only and (serves_diffusion or self.use_harmony):
            return self.create_error_response("Text prompt counts require the standard Omni Chat renderer.")
        normalized_extra_args: dict[str, object] = {}
        diffusion_request_args: dict[str, Any] = {}
""",
    )
    source = _replace_once(
        source,
        "                    entry_stage_id=entry_stage_id,\n                )\n            else:\n",
        """                    entry_stage_id=entry_stage_id,
                    count_only=count_only,
                )
            else:
""",
    )
    source = _replace_once(
        source,
        "        request_id = f\"chatcmpl-{self._base_request_id(raw_request, request.request_id)}\"\n\n"
        "        request_metadata = RequestResponseMetadata(request_id=request_id)\n",
        """        if count_only:
            (engine_prompt,) = engine_prompts
            token_ids = engine_prompt.get("prompt_token_ids")
            if entry_stage_id or engine_prompt.get("type") != "token" or not token_ids:
                return self.create_error_response("The Chat renderer did not produce a single text token sequence.")
            return {"count": len(token_ids)}

        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
""",
    )
    source = _replace_once(
        source,
        "        entry_stage_id: int = 0,\n    ) -> tuple[list[ConversationMessage], list[TokPrompt]]:\n",
        "        entry_stage_id: int = 0,\n        count_only: bool = False,\n"
        "    ) -> tuple[list[ConversationMessage], list[TokPrompt]]:\n",
    )
    return _replace_once(
        source,
        "        tok_params = request.build_tok_params(self.model_config)\n        mm_config = self.model_config.multimodal_config\n",
        """        tok_params = request.build_tok_params(self.model_config)
        if count_only:
            # Compaction needs the full input count even when generation would reject the request.
            tok_params = replace(tok_params, max_total_tokens=None, max_output_tokens=0)
        mm_config = self.model_config.multimodal_config
""",
    )


def patch_api_source(source):
    source = _replace_once(
        source,
        'async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):\n',
        'async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):\n'
        '    from vllm.exceptions import VLLMValidationError\n',
    )
    source = _replace_once(
        source,
        '''        generator = await handler.create_chat_completion(request, raw_request)
    except (EngineGenerateError, EngineDeadError) as exc:
''',
        '''        generator = await handler.create_chat_completion(request, raw_request)
    except VLLMValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EngineGenerateError, EngineDeadError) as exc:
''',
    )
    anchor = '@router.post(\n    "/v1/chat/completions",\n'
    return _replace_once(
        source,
        anchor,
        """@router.post("/v1/chat/prompt_tokens", dependencies=[Depends(validate_json_request)])
@with_cancellation
async def count_chat_prompt_tokens(request: ChatCompletionRequest, raw_request: Request):
    from vllm.exceptions import VLLMValidationError
    body = await raw_request.json()
    _validate_chat_completion_raw_body(body)
    extra_body = body.get("extra_body")
    if (
        body.get("modalities") not in (None, ["text"])
        or any(body.get(key) is not None for key in ("audio", "ref_audio", "prompt_embeds"))
        or (isinstance(extra_body, dict) and extra_body.get("ref_audio") is not None)
    ):
        raise HTTPException(status_code=400, detail="Prompt counts accept text Chat requests only.")
    if request.truncate_prompt_tokens is not None:
        raise HTTPException(status_code=400, detail="Prompt counts require an untruncated Chat request.")
    for message in body["messages"]:
        content = message.get("content")
        if message.get("audio") is not None or (
            isinstance(content, list)
            and any(not isinstance(part, dict) or part.get("type") != "text" for part in content)
        ):
            raise HTTPException(status_code=400, detail="Prompt counts accept text Chat requests only.")

    handler = Omnichat(raw_request)
    if handler is None:
        raise HTTPException(status_code=404, detail="The model does not support Chat Completions API")
    try:
        result = await handler._create_chat_completion(request, raw_request, count_only=True)
    except VLLMValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EngineGenerateError, EngineDeadError) as exc:
        return _create_engine_error_json_response(raw_request, exc)
    if isinstance(result, ErrorResponse):
        return _error_response_to_json_response(result)
    return result


""" + anchor,
    )


def main():
    omni = distribution("vllm-omni")
    vllm = distribution("vllm")
    if omni.version != "0.32.0" or vllm.version.split("+", 1)[0] != "0.32.0":
        raise RuntimeError("Recheck the Chat prompt count hook for these versions")

    patched_files = []
    for filename, patch in (("serving_chat.py", patch_chat_source), ("api_server.py", patch_api_source)):
        path = omni.locate_file(f"vllm_omni/entrypoints/openai/{filename}")
        patched = patch(path.read_text(encoding="utf-8"))
        compile(patched, str(path), "exec")
        patched_files.append((path, patched))
    for path, patched in patched_files:
        path.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
