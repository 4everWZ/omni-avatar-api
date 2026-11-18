"""Private vLLM 0.32 renderer-only capacity estimates; no GPU generation."""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.exceptions import VLLMValidationError

router = APIRouter()


def failed(value):
    if isinstance(value, ErrorResponse):
        return JSONResponse(content=value.model_dump(mode='json', by_alias=True), status_code=value.error.code)


def capacity(handler, engine_input, output_limit, truncate=None, sequences=1):
    count = handler._extract_prompt_len(engine_input)
    output = get_max_tokens(handler.model_config.max_model_len, output_limit, count,
                            handler.default_sampling_params, handler.override_max_tokens,
                            truncate_prompt_tokens=truncate)
    return {'input_tokens': count, 'max_output_tokens': output, 'sequences': sequences,
            'reserved_tokens': (count + output) * sequences}


@router.post('/internal/admission/responses')
async def responses_capacity(request: ResponsesRequest, raw_request: Request):
    handler = raw_request.app.state.openai_serving_responses
    try:
        for check in (await handler._check_model(request), handler._validate_create_responses_input(request)):
            if (error := failed(check)) is not None:
                return error
        template_error = handler.online_renderer.validate_chat_template(
            request_chat_template=None, chat_template_kwargs=request.chat_template_kwargs,
            trust_request_chat_template=handler.online_renderer.trust_request_chat_template)
        if (error := failed(template_error)) is not None:
            return error
        previous = None
        if request.previous_response_id:
            async with handler.response_store_lock:
                previous = handler.response_store.get(request.previous_response_id)
            if previous is None:
                return failed(handler._make_not_found_error(request.previous_response_id))
        rendered = await handler._render_resolved_response_inputs(request, previous)
        if (error := failed(rendered)) is not None:
            return error
        if (error := failed(handler._validate_generator_input(rendered.engine_input))) is not None:
            return error
        return capacity(handler, rendered.engine_input, request.max_output_tokens,
                        -1 if request.truncation != 'disabled' else None)
    except (VLLMValidationError, ValueError) as error:
        raise HTTPException(400, str(error)) from error


@router.post('/internal/admission/chat')
async def chat_capacity(request: ChatCompletionRequest, raw_request: Request):
    handler = raw_request.app.state.openai_serving_chat
    try:
        rendered = await handler.render_chat_request(request)
        if (error := failed(rendered)) is not None:
            return error
        _, inputs = rendered
        if len(inputs) != 1:
            raise ValueError('Admission requires one Chat prompt per request.')
        result = capacity(handler, inputs[0], request.max_completion_tokens
                          if request.max_completion_tokens is not None else request.max_tokens,
                          request.truncate_prompt_tokens)
        if request.use_beam_search:
            params = request.to_beam_search_params(result['max_output_tokens'], handler.default_sampling_params)
            sequences = params.beam_width
        else:
            params = request.to_sampling_params(result['max_output_tokens'], handler.default_sampling_params)
            sequences = max(params.n, getattr(params, 'best_of', None) or 1)
        result['sequences'] = sequences
        result['reserved_tokens'] *= sequences
        return result
    except (VLLMValidationError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
