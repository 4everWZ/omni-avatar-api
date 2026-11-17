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
        '    app.include_router(admission_router)\n'))


def patch_core_source(source):
    anchor = ('        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())\n'
              '        future = self.model_executor.execute_model(scheduler_output, non_block=True)\n')
    if source.count(anchor) != 1 or 'yield_to_renderer' in source:
        raise RuntimeError('Recheck the pinned synchronous GPU dispatch hook')
    return source.replace(anchor, anchor.splitlines(keepends=True)[0] + (
        '        from vllm.v1.engine.gpu_deadline import yield_to_renderer\n'
        '        yield_to_renderer()\n') + anchor.splitlines(keepends=True)[1])


def main():
    package = distribution('vllm')
    if package.version.split('+', 1)[0] != '0.32.0':
        raise RuntimeError('Recheck the admission renderer hooks for this vLLM version')
    target = package.locate_file('vllm/entrypoints/openai/responses/api_router.py')
    source = patch_api_source(target.read_text())
    compile(source, str(target), 'exec')
    helper = Path(__file__).with_name('llm_admission.py')
    compile(helper.read_text(), str(helper), 'exec')
    shutil.copyfile(helper, target.parent.parent / 'llm_admission.py')
    target.write_text(source)
    core = package.locate_file('vllm/v1/engine/core.py')
    source = patch_core_source(core.read_text())
    compile(source, str(core), 'exec')
    shutil.copyfile(Path(__file__).with_name('gpu_deadline.py'), core.with_name('gpu_deadline.py'))
    core.write_text(source)


if __name__ == '__main__':
    main()
