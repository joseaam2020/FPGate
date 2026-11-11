#!/usr/bin/env python3
"""Turn a model package's manifest.json into a flat text index that fpgate-accel reads
without a JSON parser. Only reads the package; nothing under model/ is modified.

    python tools/gen_pkg_index.py ../../model/quant/release/e1_r7 pkg/e1_r7.idx

Floats are written as C hex floats (exact doubles). Offsets are byte offsets into the
file named in the section header of the package (weights_i8.bin, epilogue_i32.bin, ...).
"""
import json
import sys
from pathlib import Path


def hx(v):
    return float(v).hex()


def main():
    pkg, out = Path(sys.argv[1]), Path(sys.argv[2])
    m = json.load(open(pkg / "manifest.json"))
    o = []
    o.append("version 1")
    o.append(f"name {m['name']}")
    o.append(f"ln_eps {hx(m['ln_eps'])}")
    im = m["image"]
    o.append(f"image {im['size']} {im['patch']} {im['tokens']} {im['channels']}")
    o.append(f"rope {m['rope']['frac_bits']} {m['rope']['pre_shift']}")
    o.append(f"softmax_bits {m['softmax_bits']}")
    o.append(f"silu_in_bits {m['silu_in_bits']}")
    for n, d in m["files"].items():
        o.append(f"file {n} {d['sha256']}")
    t = m["tables"]
    o.append(f"tables {t['exp']['offset']} {t['recip']['offset']} "
             f"{t['rope_cos_q14']['offset']} {t['rope_sin_q14']['offset']}")
    h = m["hps"]
    o.append(f"hps {h['cls']['offset']} {h['pos']['offset']} {h['fcn_g']['offset']} {h['fcn_b']['offset']}")
    for name, d in m["layers"].items():
        bias = d["bias"]["offset"] if d["bias"] is not None else -1
        if d["kind"] == "float":
            o.append(f"layer {name} float {d['out']} {d['in_']} {d['in_bits']} 0 {hx(d['s_x'])} "
                     f"{d['weights']['offset']} {bias} {d['deq_f64']['offset']} 0")
        else:
            o.append(f"layer {name} int {d['out']} {d['in_']} {d['in_bits']} {d['out_bits']} {hx(d['s_x'])} "
                     f"{d['weights']['offset']} {bias} {d['M0']['offset']} {d['shift']['offset']}")
    for i, b in enumerate(m["blocks"]):
        hp = b["hps"]
        o.append(
            f"block {i} {b['heads']} {b['neurons']} {hx(b['s_ctx'])} "
            f"{b['softmax_idx']['M0']} {b['softmax_idx']['shift']} {b['av_requant']['M0']} {b['av_requant']['shift']} "
            f"{b['silu_table']['offset']} {b['swiglu_requant']['M0']} {b['swiglu_requant']['shift']} "
            f"{hx(b['s_swiglu_h'])} {hx(b['s_proj_out'])} {hx(b['s_fc2_out'])} "
            f"{hp['n1_g']['offset']} {hp['n1_b']['offset']} {hp['n2_g']['offset']} {hp['n2_b']['offset']} "
            f"{hp['mlpn_g']['offset']} {hp['mlpn_b']['offset']}")
    for v in m["vectors"]:
        lg = " ".join(str(x) for x in v["logits_hex"])
        o.append(f"vector {v['index']} {v['input']} {v['label']} {lg} {v['stage_sha256']['patch_embed']}")
    out.write_text("\n".join(o) + "\n")
    print(f"{len(o)} lines -> {out}")


if __name__ == "__main__":
    main()
