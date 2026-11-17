"""Diagnostic: unchanged MiniCPM-o text weights with Transformers SDPA."""
import argparse
import gc
import json
from pathlib import Path
import sys
from time import monotonic

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, DynamicCache, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from accelerate import dispatch_model

root = Path(__file__).resolve().parents[2]
from evaluations.fixtures.long_context import repeated_document, NEEDLE

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, default=Path('/models/fb09628e98b1f55d8d5b863e0eccfcdd11bff499'))
parser.add_argument('--report', type=Path, default=root / 'tmp/torch-text-reference.json')
args = parser.parse_args()
snapshot = args.checkpoint
report_path = args.report
report_path.parent.mkdir(parents=True, exist_ok=True)
report = {'scope': 'bf16_text_backbone_reference_not_omni_acceptance',
          'model': 'openbmb/MiniCPM-o-4_5', 'checkpoint': snapshot.name,
          'weight_precision': 'bfloat16', 'kv_precision': 'bfloat16',
          'attention': 'transformers_sdpa', 'decoding': 'greedy', 'state': 'running', 'cases': []}
report['placement'] = 'sequential layer split across GPU 0 and GPU 1; not tensor parallel'


def save():
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))


@torch.inference_mode()
def run():
    torch.set_num_threads(4)
    data = json.loads((snapshot / 'config.json').read_text())
    config = Qwen3Config(**data)
    config._attn_implementation = 'sdpa'
    with torch.device('meta'):
        model = Qwen3ForCausalLM(config)
    model.model.rotary_emb = Qwen3RotaryEmbedding(config)
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    selected = [key for key in index if key.startswith('llm.')]
    weights = {}
    for shard in sorted({index[key] for key in selected}):
        with safe_open(snapshot / shard, framework='pt', device='cpu') as source:
            for key in selected:
                if index[key] == shard:
                    weights[key.removeprefix('llm.')] = source.get_tensor(key)
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    gc.collect()
    placement = {'model.embed_tokens': 0, 'model.rotary_emb': 0, 'model.norm': 1, 'lm_head': 1}
    placement.update({f'model.layers.{i}': 0 if i < config.num_hidden_layers // 2 else 1
                      for i in range(config.num_hidden_layers)})
    model = dispatch_model(model, device_map=placement, skip_keys=['past_key_values']).eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    report['parameters'] = sum(p.numel() for p in model.parameters())
    save()
    for name, message in [('short', '只回答数字：2+3等于几？'), ('repeated_39932', repeated_document(3069))]:
        started = monotonic()
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': message}],
            add_generation_prompt=True, enable_thinking=False, return_tensors='pt', return_dict=False).to('cuda')
        cache = DynamicCache(config=config)
        for offset in range(0, ids.shape[1], 2048):
            end = min(offset + 2048, ids.shape[1])
            output = model(ids[:, offset:end], past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            torch.cuda.synchronize(1)
            print(name, 'prefill', end, flush=True)
        generated = []
        stop = {tokenizer.eos_token_id, data['eos_token_id'], 151643}
        for _ in range(64):
            token = output.logits[:, -1].argmax(-1)
            generated.append(token.item())
            if generated[-1] in stop:
                break
            output = model(token[:, None], past_key_values=cache, use_cache=True, logits_to_keep=1)
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        item = {'case': name, 'prompt_tokens': ids.shape[1], 'answer': answer,
                'generated_ids': generated, 'seconds': monotonic() - started,
                'correct': answer.strip() == '5' if name == 'short' else NEEDLE in answer,
                'peak_allocated_mib': [torch.cuda.max_memory_allocated(i) / 2**20 for i in (0, 1)]}
        report['cases'].append(item)
        save()
        print(json.dumps(item, ensure_ascii=False), flush=True)
        del output, cache, ids
        gc.collect()
        torch.cuda.empty_cache()
    report['state'] = 'completed'


save()
try:
    run()
except BaseException as error:
    report['state'] = 'error'
    report['failure'] = type(error).__name__ + ': ' + str(error)
    raise
finally:
    save()
