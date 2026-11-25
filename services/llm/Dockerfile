ARG LLM_BASE_IMAGE=vllm/vllm-openai:v0.34.0@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1
FROM ${LLM_BASE_IMAGE}
COPY llm.py /opt/llm/
COPY patches/ /opt/llm/patches/
RUN python3 /opt/llm/patches/patch_llm_admission.py && python3 /opt/llm/patches/patch_reasoning_budget.py
ENTRYPOINT []
CMD ["python3", "/opt/llm/llm.py", "serve"]
