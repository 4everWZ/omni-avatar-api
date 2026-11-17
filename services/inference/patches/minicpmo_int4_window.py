"""Re-anchor MiniCPM-o keys stored by vLLM 0.32's packed Triton INT4 cache."""

import torch


def _round_away_from_zero(value):
    return torch.where(value >= 0, value + 0.5, value - 0.5).to(torch.int32)


def rotate_packed_int4_keys(pool, indices, head_dim, rotate, rht):
    """Undo the write transform, apply floating RoPE, and repack only K.

    Each logical BHND row contains K nibbles, a K scale word, V nibbles and
    a V scale word. The zero point is carried in the scale's low four bits.
    The physical storage may be HND or NHD.
    """
    packed_head = head_dim // 2
    if pool.dtype != torch.uint8 or pool.ndim != 4 or pool.shape[-1] != 2 * (packed_head + 4):
        raise ValueError("Expected the packed Triton INT4 KV cache layout")
    scale_offset = pool.storage_offset() + packed_head
    strides = pool.stride()[:3]
    if pool.stride(-1) != 1 or scale_offset % 4 or any(stride % 4 for stride in strides):
        raise ValueError("Packed Triton INT4 KV scale words must be 32-bit-aligned")

    scale_storage = torch.empty(0, dtype=torch.int32, device=pool.device).set_(pool.untyped_storage())
    scale_words = torch.as_strided(
        scale_storage,
        size=pool.shape[:3],
        stride=tuple(stride // 4 for stride in strides),
        storage_offset=scale_offset // 4,
    )
    rows = (*indices, slice(0, packed_head))
    packed = pool[rows].to(torch.int32)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    metadata = scale_words[indices]
    scales = (metadata & -16).view(torch.float32)
    transformed = (codes - (metadata & 15).unsqueeze(-1)).float() * scales.unsqueeze(-1)

    # Single RHT uses an unnormalized Hadamard matrix, block-diagonal for
    # non-power-of-two dimensions. Its inverse divides by the block width.
    hadamard_width = head_dim & -head_dim
    keys = rht(transformed, inverse=True) / hadamard_width
    transformed = rht(rotate(keys))

    # Match the upstream INT4 writer's asymmetric scale, round-half-away,
    # even/odd nibble order, and zero point encoding.
    minimum = transformed.amin(dim=-1)
    scale = (transformed.amax(dim=-1) - minimum).div_(15).clamp_min_(1e-6)
    zero = _round_away_from_zero(-minimum / scale).clamp_(0, 15)
    codes = _round_away_from_zero(transformed * scale.reciprocal().unsqueeze(-1) + zero.unsqueeze(-1))
    codes.clamp_(0, 15)
    pool[rows] = (codes[..., ::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    scale_words[indices] = (scale.view(torch.int32) & -16) | zero
