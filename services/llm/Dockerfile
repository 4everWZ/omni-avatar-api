ARG LLM_BASE_IMAGE=vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
FROM ${LLM_BASE_IMAGE}
COPY llm.py /opt/llm/
COPY patches/ /opt/llm/patches/
RUN python3 /opt/llm/patches/patch_llm_admission.py && python3 /opt/llm/patches/patch_reasoning_budget.py
ENTRYPOINT []
CMD ["python3", "/opt/llm/llm.py", "serve"]
