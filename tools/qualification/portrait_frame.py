"""Compare fixed-shape Ditto frame backends on real prepared source features."""
import argparse
import importlib
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace


def main(args):
    import numpy as np
    import torch
    from services.portrait.backend import CODE_REVISION, MODEL_REVISION, CudaFrame

    sys.path.insert(0, str(Path(args.root) / 'code' / CODE_REVISION))
    load = importlib.import_module('core.utils.load_model').load_model
    models = Path(args.root) / 'models' / MODEL_REVISION / 'ditto_pytorch/models'
    with Path(args.prepared).open('rb') as source:
        prepared = pickle.load(source)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'code_revision': CODE_REVISION, 'model_revision': MODEL_REVISION,
              'scope': 'isolated warp+decoder GPU latency and output drift on one prepared identity',
              'state': 'running',
              'iterations': args.iterations, 'backends': {}, 'comparisons': []}
    outputs = {}
    rng = np.random.default_rng(1729)
    source = prepared['x_s_info_lst'][0]['kp'].reshape(1, 21, 3)
    motions = [source.copy()] + [source + rng.normal(0, scale, source.shape).astype('float32')
                                for scale in (.01, .03)]
    for backend in args.backends:
        os.environ['OMNI_PORTRAIT_FRAME_BACKEND'] = backend
        warp, _ = load(str(models / 'warp_network.pth'), module_name='WarpingNetwork')
        decoder, _ = load(str(models / 'decoder.pth'), module_name='SPADEDecoder')
        frame = CudaFrame(SimpleNamespace(model=warp, device='cuda'),
                          SimpleNamespace(model=decoder, device='cuda'), torch)
        frame.prepare(prepared)
        outputs[backend] = []
        for motion in motions:
            frame.kp_driving.copy_(torch.from_numpy(motion))
            frame.graph.replay()
            torch.cuda.synchronize()
            value = frame.output.detach().cpu().numpy().copy()
            assert np.isfinite(value).all(), backend
            outputs[backend].append(value)
        for _ in range(10):
            frame.graph.replay()
        torch.cuda.synchronize()
        timings = []
        for _ in range(args.iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            frame.graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        free, total = torch.cuda.mem_get_info()
        result['backends'][backend] = {'mean_ms': float(np.mean(timings)),
            'p50_ms': float(np.percentile(timings, 50)), 'p95_ms': float(np.percentile(timings, 95)),
            'torch_allocated_bytes': torch.cuda.memory_allocated(),
            'torch_reserved_bytes': torch.cuda.memory_reserved(),
            'device_free_bytes': free, 'device_total_bytes': total}
        Path(args.report).write_text(json.dumps(result, indent=2))
        print(backend, result['backends'][backend], flush=True)
        del frame, warp, decoder
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if len(args.backends) == 2:
        for index, (a, b) in enumerate(zip(*(outputs[name] for name in args.backends))):
            diff = np.abs(a-b)
            mse = float(np.mean((a-b)**2))
            result['comparisons'].append({'case': index, 'mae': float(diff.mean()),
                'max_abs': float(diff.max()), 'psnr_db': float(-10*np.log10(max(mse, 1e-12)))})
            from PIL import Image
            for name in args.backends:
                rgb = np.transpose(outputs[name][index][0], (1, 2, 0))
                Image.fromarray((rgb.clip(0, 1)*255).astype('uint8')).save(
                    Path(args.report).with_name(f'frame-{name}-{index}.png'))
    result['state'] = 'completed'
    Path(args.report).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/portrait')
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--backends', nargs='+', choices=['cuda-graph', 'tensorrt'],
                        default=['cuda-graph', 'tensorrt'])
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1 or len(args.backends) != len(set(args.backends)):
        parser.error('Use positive iterations and distinct backends')
    main(args)
