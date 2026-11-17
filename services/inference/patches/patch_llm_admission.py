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


if __name__ == '__main__':
    main()
