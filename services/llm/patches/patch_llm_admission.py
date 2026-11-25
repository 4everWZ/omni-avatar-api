"""Install private renderer-based budget routes in the pinned text engine."""
from importlib.metadata import distribution
from pathlib import Path
import shutil


def patch_api_source(source):
    anchor = 'def attach_router(app: FastAPI):\n    app.include_router(router)\n'
    if source.count(anchor) != 1 or 'admission_router' in source:
        raise RuntimeError('vLLM Responses router no longer matches the pinned hook')
    return source.replace(anchor, anchor + (
        '    from vllm.entrypoints.openai.llm_admission import router as admission_router\n'
        '    app.include_router(admission_router)\n'
        '    from vllm.entrypoints.openai.llm_gpu_control import attach_gpu_control\n'
        '    attach_gpu_control(app)\n'))


def main():
    package = distribution('vllm')
    if package.version.split('+', 1)[0] not in ('0.32.0', '0.34.0'):
        raise RuntimeError('Recheck the admission renderer hooks for this vLLM version')
    target = package.locate_file('vllm/entrypoints/openai/responses/api_router.py')
    source = patch_api_source(target.read_text())
    compile(source, str(target), 'exec')
    helper = Path(__file__).with_name('llm_admission.py')
    compile(helper.read_text(), str(helper), 'exec')
    shutil.copyfile(helper, target.parent.parent / 'llm_admission.py')
    helper = helper.with_name('llm_gpu_control.py')
    compile(helper.read_text(), str(helper), 'exec')
    shutil.copyfile(helper, target.parent.parent / helper.name)
    target.write_text(source)


if __name__ == '__main__':
    main()
