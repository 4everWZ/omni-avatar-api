"""Exercise MiniCPM-o's window against the Triton packed INT4 wire layout."""

import ast
from itertools import product
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from minicpmo_int4_window import rotate_packed_int4_keys
from patch_minicpmo_int4_window import patch_window_source


HEAD_DIM, HEADS, PAGE = 128, 8, 16
PACKED_HEAD = HEAD_DIM // 2
PADDED_HEAD = PACKED_HEAD + 4


def reference_rht(value, inverse=False):
    block_width = value.shape[-1] & -value.shape[-1]
    generator = torch.Generator().manual_seed(0x9E3779B9)
    signs = (2 * torch.bernoulli(torch.full((block_width,), 0.5), generator=generator) - 1).to(value.device)
    basis = torch.ones((1, 1), device=value.device)
    while basis.shape[0] < block_width:
        basis = torch.cat((torch.cat((basis, basis), 1), torch.cat((basis, -basis), 1)), 0)
    rows = value.reshape(*value.shape[:-1], -1, block_width)
    changed = (rows @ basis) * signs if inverse else (rows * signs) @ basis
    return changed.reshape_as(value)


def oracle_quantize(transformed):
    def nearest(x):
        return torch.copysign(torch.floor(x.abs() + 0.5), x).int()

    low, high = transformed.min(dim=-1).values, transformed.max(dim=-1).values
    scales = ((high - low) / 15).clamp(min=1e-6)
    zeros = nearest(-low / scales).clamp(0, 15)
    codes = nearest(transformed / scales[..., None] + zeros[..., None]).clamp(0, 15)
    nibbles = codes[..., ::2] + 16 * codes[..., 1::2]
    scale_bits = (scales.view(torch.int32) & 0xFFFFFFF0) + zeros
    return nibbles.byte(), scale_bits


def scale_word(pool):
    return pool[..., PACKED_HEAD:PADDED_HEAD].view(torch.int32).squeeze(-1)


def decode_keys(packed, word):
    pairs = torch.stack((packed % 16, packed // 16), -1).flatten(-2)
    matrix_value = (pairs.int() - (word % 16)[..., None]).float() * (word & -16).view(torch.float32)[..., None]
    return reference_rht(matrix_value, inverse=True) / HEAD_DIM


def make_pool(*, physical_nhd, device="cpu"):
    shape = (5, PAGE, HEADS, 2 * PADDED_HEAD) if physical_nhd else (5, HEADS, PAGE, 2 * PADDED_HEAD)
    rng = torch.Generator().manual_seed(823)
    backing = torch.randint(0, 256, shape, generator=rng, dtype=torch.uint8).to(device)
    pool = backing[1:].transpose(1, 2) if physical_nhd else backing[1:]
    keys = torch.randn((*pool.shape[:3], HEAD_DIM), generator=rng).to(device)
    packed, metadata = oracle_quantize(reference_rht(keys))
    pool[..., :PACKED_HEAD] = packed
    scale_word(pool).copy_(metadata)
    return backing, pool


def reference_rope(keys, delta, frequencies):
    first, second = keys.chunk(2, dim=-1)
    phase = torch.polar(torch.ones_like(frequencies), -delta * frequencies)
    result = torch.complex(first, second) * phase
    return torch.cat((result.real, result.imag), dim=-1)


class MinicpmInt4WindowTests(unittest.TestCase):
    def test_rotate_matches_writer_oracle_and_preserves_values_and_other_slots(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for physical_nhd, device in product((False, True), devices):
            with self.subTest(physical_nhd=physical_nhd, device=device):
                backing, pool = make_pool(physical_nhd=physical_nhd, device=device)
                before = pool.clone()
                preceding = backing[0].clone()
                blocks = torch.tensor([3, 1, 0], device=device)[:, None]
                heads = torch.arange(HEADS, device=device)[None, :]
                offsets = torch.tensor([15, 2, 7], device=device)[:, None]
                indices = blocks, heads, offsets
                frequencies = 10000 ** -(torch.arange(HEAD_DIM // 2, device=device).float() / (HEAD_DIM // 2))
                original = decode_keys(pool[(*indices, slice(PACKED_HEAD))], scale_word(pool)[indices])
                expected = reference_rope(original, 1536, frequencies)
                expected_codes, expected_words = oracle_quantize(reference_rht(expected))

                rotate_packed_int4_keys(pool, indices, HEAD_DIM,
                                        lambda keys: reference_rope(keys, 1536, frequencies), reference_rht)

                torch.testing.assert_close(scale_word(pool)[indices], expected_words, atol=0, rtol=0)
                torch.testing.assert_close(pool[(*indices, slice(PACKED_HEAD))], expected_codes, atol=0, rtol=0)
                mutable = torch.zeros_like(pool, dtype=torch.bool)
                mutable[(*indices, slice(PADDED_HEAD))] = True
                self.assertTrue(torch.equal(pool[~mutable], before[~mutable]))
                self.assertTrue(torch.equal(backing[0], preceding))

    def test_nibbles_scale_floor_and_zero_point_bits(self):
        pool = torch.zeros(1, 1, 2, 2 * PADDED_HEAD, dtype=torch.uint8)
        indices = torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), torch.arange(2)
        rotated = torch.zeros(2, HEAD_DIM)
        rotated[1, :6] = torch.tensor([-7., 8., -6.5, 1.5, -6., -5.])
        rotate_packed_int4_keys(pool, indices, HEAD_DIM, lambda _keys: rotated,
                                lambda keys, inverse=False: keys)
        self.assertEqual(pool[0, 0, 1, :3].tolist(), [0xF0, 0x91, 0x21])
        self.assertEqual((scale_word(pool) & 15).flatten().tolist(), [0, 7])
        actual_scale = (scale_word(pool) & -16).view(torch.float32).flatten()
        torch.testing.assert_close(actual_scale, torch.tensor([1e-6, 1.]), atol=0, rtol=2e-6)

    @unittest.skipUnless(os.environ.get("VLLM_INT4_CACHE_SOURCE"), "requires the pinned Triton INT4 source")
    def test_reference_hadamard_matches_pinned_backend(self):
        upstream = Path(os.environ["VLLM_INT4_CACHE_SOURCE"]).read_text(encoding="utf-8")
        functions = {"_get_rht_signs", "single_rht", "fast_hadamard_transform"}
        definitions = [node for node in ast.parse(upstream).body
                       if isinstance(node, ast.FunctionDef) and node.name in functions]
        context = {
            "torch": torch, "largest_power_of_2_divisor": lambda n: n & -n,
            "_RHT_SIGNS_CACHE": {}, "_hadacore_available": lambda: False,
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "<pinned-int4-rht>", "exec"), context)
        upstream_rht = context["single_rht"]
        for head_dim in (64, 96, 128):
            with self.subTest(head_dim=head_dim):
                sample = torch.randn((4, 8, head_dim), generator=torch.Generator().manual_seed(45))
                torch.testing.assert_close(reference_rht(sample), upstream_rht(sample), atol=1e-5, rtol=2e-6)
                torch.testing.assert_close(reference_rht(sample, inverse=True), upstream_rht(sample, inverse=True),
                                           atol=1e-5, rtol=2e-6)

    @unittest.skipUnless(os.environ.get("VLLM_OMNI_WINDOW_SOURCE"), "requires the pinned upstream source")
    def test_pinned_reanchor_dispatch_and_floating_cache(self):
        upstream = Path(os.environ["VLLM_OMNI_WINDOW_SOURCE"]).read_text(encoding="utf-8")
        patched = patch_window_source(upstream)
        definitions = [node for node in ast.parse(patched).body
                       if isinstance(node, ast.FunctionDef) and node.name in {"rotate_keys", "rotate_cached_keys"}]
        context = {"torch": torch, "PositionReanchor": object, "rotate_packed_int4_keys": rotate_packed_int4_keys}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "<pinned-window-rotate>", "exec"), context)
        frequencies = torch.linspace(0.01, 1, HEAD_DIM // 2)
        old_positions = torch.tensor([16, 31, 33])
        plan = SimpleNamespace(delta=16, moved_from=16)
        indices = torch.tensor([2, 2, 0])[:, None], torch.arange(HEADS)[None, :], torch.tensor([0, 15, 1])[:, None]
        rht_module = "vllm.v1.attention.ops.int4_per_token_head"
        _, int4_pool = make_pool(physical_nhd=True)
        dense = torch.randn(4, HEADS, PAGE, 2 * HEAD_DIM, generator=torch.Generator().manual_seed(719))

        for pool in (int4_pool, dense):
            with self.subTest(dtype=pool.dtype):
                before = pool.clone()
                packed = pool.dtype == torch.uint8
                content = PACKED_HEAD if packed else HEAD_DIM
                original = before[(*indices, slice(content))]
                keys = decode_keys(original, scale_word(before)[indices]) if packed else original
                expected = reference_rope(keys, plan.delta, frequencies)
                with patch.dict(sys.modules, {rht_module: SimpleNamespace(single_rht=reference_rht)}):
                    rows = context["rotate_cached_keys"](
                        pool, block_ids=[2, 0, 3], positions=old_positions,
                        plan=plan, inv_freq=frequencies, block_size=PAGE,
                    )
                self.assertEqual(rows, 3)
                actual = pool[(*indices, slice(content))]
                if packed:
                    _, expected_words = oracle_quantize(reference_rht(expected))
                    torch.testing.assert_close(scale_word(pool)[indices], expected_words, atol=0, rtol=0)
                    # Bound after dequantization in the same Hadamard domain used by attention.
                    actual_rht = reference_rht(decode_keys(actual, scale_word(pool)[indices]))
                    scale = (expected_words & -16).view(torch.float32)
                    self.assertTrue(bool(((actual_rht - reference_rht(expected)).abs() <= scale[..., None] * 0.501).all()))
                else:
                    torch.testing.assert_close(actual, expected)
                selected = torch.zeros_like(pool, dtype=torch.bool)
                selected[(*indices, slice(PADDED_HEAD if packed else HEAD_DIM))] = True
                self.assertTrue(torch.equal(pool[~selected], before[~selected]))


if __name__ == "__main__":
    unittest.main()
