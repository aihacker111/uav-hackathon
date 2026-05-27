"""
tools/count_model.py
====================
Count GFLOPs and parameter breakdown for any model in this project.

Supported archs (--arch):
  dla_34        DLA-34 + DCNv2  (original AMOT backbone)
  deimv2        DINOv3STAs + HybridEncoder + CenterNet JDE heads

Usage examples:
  # DEIMv2-JDE (default, VisDrone 5-class)
  python tools/count_model.py --arch deimv2

  # DLA-34 baseline
  python tools/count_model.py --arch dla_34

  # Custom input size & class count
  python tools/count_model.py --arch deimv2 --input-h 864 --input-w 1536 --num-classes 7

  # Count per-module breakdown
  python tools/count_model.py --arch deimv2 --verbose

  # Load a trained checkpoint
  python tools/count_model.py --arch deimv2 --weights ./pretrained_model/ecdet_s.pth
"""

import argparse
import sys
import os

# ── path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "lib"))
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from thop import profile, clever_format


# ── helpers ───────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module):
    """Return (total, trainable, frozen) parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    return total, trainable, frozen


def module_param_table(model: nn.Module, top_n: int = 15):
    """
    Build a per-top-level-submodule parameter table.
    Returns list of (name, params_M, trainable_M, frozen_M).
    """
    rows = []
    for name, module in model.named_children():
        total     = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen    = total - trainable
        rows.append((name, total / 1e6, trainable / 1e6, frozen / 1e6))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top_n]


def count_flops(model: nn.Module, input_tensor: torch.Tensor, verbose: bool = False):
    """
    Run thop.profile and return (macs, params) as raw numbers.
    macs × 2 ≈ FLOPs  (thop reports MACs by default).
    """
    # thop may print per-layer info; suppress unless verbose
    if not verbose:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    else:
        macs, params = profile(model, inputs=(input_tensor,), verbose=True)
    return macs, params


def print_banner(title: str):
    w = 60
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)


def print_summary(arch, input_h, input_w, model, macs, verbose):
    total, trainable, frozen = count_parameters(model)

    flops = macs * 2                         # MACs → FLOPs
    gflops = flops / 1e9
    gmacs  = macs  / 1e9

    print_banner(f"Model: {arch}  |  Input: {input_h}×{input_w}")

    # ── Parameter summary ───────────────────────────────────────────────────
    print(f"\n{'Parameter Summary':}")
    print(f"  {'Total':30s}  {total/1e6:>8.3f} M")
    print(f"  {'Trainable':30s}  {trainable/1e6:>8.3f} M")
    print(f"  {'Frozen':30s}  {frozen/1e6:>8.3f} M")

    # ── FLOPs / MACs ────────────────────────────────────────────────────────
    print(f"\n{'Compute (single image forward)':}")
    print(f"  {'GMACs  (thop)':30s}  {gmacs:>8.3f} GMACs")
    print(f"  {'GFLOPs (MACs×2)':30s}  {gflops:>8.3f} GFLOPs")

    # ── Per-module breakdown ─────────────────────────────────────────────────
    if verbose:
        rows = module_param_table(model)
        print(f"\n{'Per-module breakdown (top-level submodules)':}")
        header = f"  {'Module':<18} {'Total (M)':>10} {'Trainable (M)':>15} {'Frozen (M)':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, tot, train, frz in rows:
            print(f"  {name:<18} {tot:>10.3f} {train:>15.3f} {frz:>12.3f}")

    print("\n" + "═" * 60 + "\n")


# ── model builders ────────────────────────────────────────────────────────────

def build_dla34(num_classes: int, head_conv: int, reid_dim: int):
    from lib.models.model import create_model
    heads = {
        "hm":  num_classes,
        "wh":  2,
        "reg": 2,
        "id":  reid_dim,
    }
    return create_model("dla_34", heads, head_conv)


def build_deimv2(num_classes: int, head_conv: int, reid_dim: int, weights: str = ""):
    from lib.models.networks.deimv2_jde import get_deimv2_jde
    heads = {
        "hm":  num_classes,
        "wh":  2,
        "reg": 2,
        "id":  reid_dim,
    }
    model = get_deimv2_jde(
        heads=heads,
        head_conv=head_conv,
        deimv2_pretrained=weights if weights else None,
        freeze_backbone=False,
    )
    return model


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Count GFLOPs and parameters for project models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--arch",        default="deimv2",
                   choices=["dla_34", "deimv2"],
                   help="Model architecture to profile")
    p.add_argument("--input-h",     type=int,   default=608,
                   help="Input image height (pixels)")
    p.add_argument("--input-w",     type=int,   default=1088,
                   help="Input image width (pixels)")
    p.add_argument("--num-classes", type=int,   default=10,
                   help="Number of detection classes (VisDrone=5, UAVDT=5, etc.)")
    p.add_argument("--reid-dim",    type=int,   default=128,
                   help="ReID embedding dimension")
    p.add_argument("--head-conv",   type=int,   default=256,
                   help="Intermediate channels in CenterNet heads")
    p.add_argument("--weights",     type=str,   default="",
                   help="Optional path to a checkpoint to load before counting")
    p.add_argument("--device",      type=str,   default="cpu",
                   help="Device for dummy forward: cpu | cuda | cuda:0")
    p.add_argument("--verbose",     action="store_true",
                   help="Print per-module parameter table and per-layer thop output")
    p.add_argument("--compare",     action="store_true",
                   help="Profile BOTH dla_34 and deimv2 and print side-by-side")
    return p.parse_args()


def profile_one(arch, args, device):
    """Build model, create dummy input, run thop. Returns (model, macs)."""
    print(f"\n  Building {arch} …", end=" ", flush=True)

    if arch == "dla_34":
        try:
            model = build_dla34(args.num_classes, args.head_conv, args.reid_dim)
        except Exception as e:
            print(f"\n  [SKIP] {arch}: {e}")
            return None, None
    elif arch == "deimv2":
        model = build_deimv2(args.num_classes, args.head_conv, args.reid_dim, args.weights)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.eval()
    model = model.to(device)
    print("done.")

    dummy = torch.zeros(1, 3, args.input_h, args.input_w, device=device)

    print(f"  Running thop forward on {device} …", end=" ", flush=True)
    macs, _ = count_flops(model, dummy, verbose=args.verbose)
    print("done.")

    return model, macs


def compare_table(results):
    """Print side-by-side comparison of multiple (arch, model, macs) tuples."""
    print_banner("Side-by-side Comparison")
    col_w = 18
    header = f"  {'Metric':<28}" + "".join(f"{arch:>{col_w}}" for arch, _, _ in results)
    print(header)
    print("  " + "-" * (len(header) - 2))

    def row(label, values):
        print(f"  {label:<28}" + "".join(f"{v:>{col_w}}" for v in values))

    totals     = [count_parameters(m)[0] / 1e6 for _, m, _ in results]
    trainables = [count_parameters(m)[1] / 1e6 for _, m, _ in results]
    gmacs      = [macs / 1e9 for _, _, macs in results]
    gflops     = [macs * 2 / 1e9 for _, _, macs in results]

    row("Total params (M)",     [f"{v:.3f}" for v in totals])
    row("Trainable params (M)", [f"{v:.3f}" for v in trainables])
    row("GMACs",                [f"{v:.3f}" for v in gmacs])
    row("GFLOPs (MACs×2)",      [f"{v:.3f}" for v in gflops])
    print()


def main():
    args = parse_args()
    device = torch.device(args.device)

    archs = ["dla_34", "deimv2"] if args.compare else [args.arch]
    results = []

    for arch in archs:
        model, macs = profile_one(arch, args, device)
        if model is None:
            continue
        print_summary(arch, args.input_h, args.input_w, model, macs, args.verbose)
        results.append((arch, model, macs))

    if args.compare and len(results) > 1:
        compare_table(results)


if __name__ == "__main__":
    main()
