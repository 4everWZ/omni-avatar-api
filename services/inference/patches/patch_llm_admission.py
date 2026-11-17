"""Install private renderer-based budget routes in the pinned text engine."""
from importlib.metadata import distribution
from pathlib import Path
import shutil


def main():
    package = distribution('vllm')
    if package.version.split('+', 1)[0] != '0.32.0':
        raise RuntimeError('Recheck the admission renderer hooks for this vLLM version')
    target = package.locate_file('vllm/entrypoints/openai/responses/api_router.py')
    source = target.read_text()
    anchor = 'def attach_router(app: FastAPI):\n    app.include_router(router)\n'
    if source.count(anchor) != 1 or 'admission_router' in source:
        raise RuntimeError('vLLM Responses router no longer matches the pinned hook')
    source = source.replace(anchor, anchor + (
        '    from vllm.entrypoints.openai.llm_admission import router as admission_router\n'
        '    app.include_router(admission_router)\n'))
    compile(source, str(target), 'exec')
    helper = Path(__file__).with_name('llm_admission.py')
    compile(helper.read_text(), str(helper), 'exec')
    shutil.copyfile(helper, target.parent.parent / 'llm_admission.py')
    target.write_text(source)
    core = package.locate_file('vllm/v1/engine/core.py')
    source = core.read_text()
    anchor = '        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0\n'
    if source.count(anchor) != 1 or 'LLM_GPU_STEP_PAUSE_MS' in source:
        raise RuntimeError('Recheck the synchronous GPU batch yield hook')
    source = source.replace(anchor, (
        '        pause_ms = float(os.environ.get("LLM_GPU_STEP_PAUSE_MS", "0"))\n'
        '        if pause_ms > 0 and scheduler_output.total_num_scheduled_tokens > 0:\n'
        '            time.sleep(pause_ms / 1000)\n'
    ) + anchor)
    compile(source, str(core), 'exec')
    core.write_text(source)


if __name__ == '__main__':
    main()
