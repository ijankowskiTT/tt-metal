# Repro for the ttnn.sigmoid_bw cancellation.
#
# sigmoid_bw computes sigmoid(x) * (1 - sigmoid(x)) as separate ttnn ops, so the intermediate
# sigmoid(x) is rounded to the tensor dtype. Once it rounds to exactly 1.0 the subtraction gives
# exactly 0 and the gradient is lost, even though the true derivative is a normal representable
# number there.
#
# Sweeps every bf16 code in [1, 40) and (-40, -1), then locates the fp32 onset at fp32 resolution.
# Reports, per range: first point above 10% relative error, first exact zero, how many codes
# return zero, the worst relative error, and the worst error of sigmoid(x) * sigmoid(-x) -- the
# same quantity written without a subtraction.
import math
import struct

import torch

import ttnn


def bf16_codes(lo, hi, start, stop):
    """Every bf16 value in [lo, hi), enumerated as the top 16 bits of a float32."""
    vs = (struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in range(start, stop))
    return sorted({v for v in vs if lo <= v < hi})


def exact(v):
    return 1.0 / (1.0 + math.exp(-v)) * 1.0 / (1.0 + math.exp(v))


def run(device, xs, torch_dtype, ttnn_dtype):
    n = len(xs)
    pad = ((n + 1023) // 1024) * 1024
    buf = torch.zeros(pad, dtype=torch_dtype)
    buf[:n] = torch.tensor(xs, dtype=torch_dtype)
    t = buf.reshape(1, 1, pad // 32, 32)

    x = ttnn.from_torch(t, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    g = ttnn.from_torch(torch.ones_like(t), dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)

    cur = ttnn.to_torch(ttnn.sigmoid_bw(g, x)[0]).reshape(-1)[:n].double().tolist()
    fix = ttnn.to_torch(ttnn.multiply(ttnn.sigmoid(x), ttnn.sigmoid(ttnn.neg(x)))).reshape(-1)[:n].double().tolist()
    return cur, fix


def report(label, xs, cur, fix):
    first_10pct = first_zero = None
    n_zero = 0
    max_cur_err = max_fix_err = 0.0
    for v, c, f in zip(xs, cur, fix):
        ex = exact(v)
        if abs(c - ex) / ex > 0.10 and first_10pct is None:
            first_10pct = v
        if c == 0.0:
            n_zero += 1
            if first_zero is None:
                first_zero = v
        max_cur_err = max(max_cur_err, abs(c - ex) / ex)
        max_fix_err = max(max_fix_err, abs(f - ex) / ex)

    print(f"[{label}]  {len(xs)} codes")
    print(f"  first >10% rel err             : {first_10pct}")
    print(f"  first exact zero               : {first_zero}")
    print(f"  exact zeros where value exists : {n_zero}")
    print(f"  worst rel err, current         : {max_cur_err:.3%}")
    print(f"  max rel err of sig(x)*sig(-x)  : {max_fix_err:.3%}")


def fp32_onset(device):
    """Locate the fp32 onset at fp32 resolution, not on the coarse bf16 grid."""
    step = 2.0**-8
    xs = [16.0 + i * step for i in range(int(1.0 / step) + 1)]
    cur, _ = run(device, xs, torch.float32, ttnn.float32)

    last_nonzero = first_zero = None
    for v, c in zip(xs, cur):
        if c != 0.0:
            last_nonzero = v
        elif first_zero is None:
            first_zero = v

    plateau = sorted({f"{c:.6e}" for c in cur if c != 0.0})
    print("[fp32 onset]")
    print(f"  plateau value before the drop  : {plateau[0] if len(plateau) == 1 else plateau}")
    print(f"  last non-zero                  : {last_nonzero}")
    print(f"  first exact zero               : {first_zero}")
    print(f"  predicted ln(2^24)             : {24 * math.log(2):.4f}")
    print("  (1 + exp(-x) is summed in fp32; once exp(-x) < 2^-24 it rounds to 1.0)")


def main():
    device = ttnn.open_device(device_id=0)
    try:
        pos = bf16_codes(1.0, 40.0, 0x3F80, 0x4220)
        neg = bf16_codes(-40.0, -1.0, 0xBF80, 0xC220)
        for label, xs in (("bf16 x in [1, 40)", pos), ("bf16 x in (-40, -1)", neg)):
            cur, fix = run(device, xs, torch.bfloat16, ttnn.bfloat16)
            report(label, xs, cur, fix)
        fp32_onset(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
