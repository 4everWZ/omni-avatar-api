"""Target-built FP16 warp/decoder engine, with persistent CUDA tensor bindings."""
import copy
import json
import logging
import os
from pathlib import Path


class TensorRTFrame:
    """One fixed-shape engine; ownership follows the portrait worker GPU lease."""

    def __init__(self, warp, decoder, torch, cache):
        import tensorrt as trt
        from .backend import CODE_REVISION, MODEL_REVISION

        self.torch, self.trt = torch, trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        device = torch.cuda.current_device()
        identity = {'code': CODE_REVISION, 'model': MODEL_REVISION,
                    'tensorrt': trt.__version__, 'torch': torch.__version__,
                    'gpu': torch.cuda.get_device_name(device),
                    'capability': list(torch.cuda.get_device_capability(device)),
                    'precision': 'fp16-fp32-coordinates', 'format': 3}
        folder = Path(cache)
        folder.mkdir(parents=True, exist_ok=True)
        plan, manifest = folder / 'warp-decoder.plan', folder / 'warp-decoder.json'
        try:
            cached_identity = json.loads(manifest.read_text())
        except (OSError, ValueError):
            cached_identity = None
        if not (plan.exists() and cached_identity == identity):
            self.build(warp, decoder, folder, plan)
            temporary = manifest.with_suffix('.json.new')
            temporary.write_text(json.dumps(identity, indent=2))
            os.replace(temporary, manifest)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT warp/decoder engine deserialization failed')
        expected = {'feature': (1, 32, 16, 64, 64), 'source': (1, 21, 3),
                    'driving': (1, 21, 3), 'output': (1, 3, 512, 512)}
        if self.engine.num_io_tensors != len(expected):
            raise RuntimeError('Unexpected TensorRT frame tensor count')
        for name, shape in expected.items():
            if (tuple(self.engine.get_tensor_shape(name)) != shape
                    or self.engine.get_tensor_dtype(name) != trt.float32
                    or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise RuntimeError('Unexpected TensorRT frame tensor contract: ' + name)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT execution context creation failed')
        logging.warning('Portrait TensorRT FP16 ready: version=%s gpu=%s', trt.__version__, identity['gpu'])
        self.output = torch.empty(tuple(self.engine.get_tensor_shape('output')),
                                  device='cuda', dtype=torch.float32)

    def build(self, warp, decoder, folder, plan):
        torch, trt = self.torch, self.trt

        class Frame(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.warp = copy.deepcopy(warp).half().eval()
                self.decoder = copy.deepcopy(decoder).half().eval()

            def forward(self, feature, source, driving):
                with torch.autocast('cuda', dtype=torch.float16):
                    feature = self.warp(feature.half(), source, driving).float()
                    return self.decoder(feature).float()

        inputs = (torch.zeros((1, 32, 16, 64, 64), device='cuda'),
                  torch.zeros((1, 21, 3), device='cuda'),
                  torch.zeros((1, 21, 3), device='cuda'))
        model = Frame().eval()
        onnx = folder / 'warp-decoder.onnx'
        with torch.no_grad():
            torch.onnx.export(model, inputs, str(onnx), input_names=['feature', 'source', 'driving'],
                              output_names=['output'], opset_version=20, dynamo=True)
        del model, inputs
        torch.cuda.empty_cache()
        builder = trt.Builder(self.logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        # TensorRT 11 uses explicit ONNX FP16 types. Build on the serving GPU.
        config.max_aux_streams = 0
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT FP16 warp/decoder build failed')
        temporary = plan.with_suffix('.new')
        temporary.write_bytes(bytes(serialized))
        os.replace(temporary, plan)

    def __call__(self, feature, source, driving):
        for name, value in [('feature', feature), ('source', source), ('driving', driving),
                            ('output', self.output)]:
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError('TensorRT tensor binding failed: ' + name)
        if not self.context.execute_async_v3(self.torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT warp/decoder enqueue failed')
        return self.output


if __name__ == '__main__':
    import argparse
    import importlib
    import sys
    import torch
    from .backend import CODE_REVISION, MODEL_REVISION

    parser = argparse.ArgumentParser(description='Build/load the target GPU FP16 frame engine')
    parser.add_argument('--root', default='/portrait')
    parser.add_argument('--cache', default='/cache/tensorrt')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.root) / 'code' / CODE_REVISION))
    load = importlib.import_module('core.utils.load_model').load_model
    models = Path(args.root) / 'models' / MODEL_REVISION / 'ditto_pytorch/models'
    warp, _ = load(str(models / 'warp_network.pth'), module_name='WarpingNetwork')
    decoder, _ = load(str(models / 'decoder.pth'), module_name='SPADEDecoder')
    TensorRTFrame(warp, decoder, torch, args.cache)
