# -*- coding: utf-8 -*-
"""Validate the C1+C2 inference fixes on a processed .npy case.

Runs the trained BaselineUNet on an ROI-cropped, HU-normalised processed
tensor (same distribution as training) and reports the heatmap distribution
with and without the erroneous second sigmoid. This is the Task-12
"eval processed .npy first" validation path — it isolates the checkpoint-load
and double-sigmoid fixes from the unresolved train/inference crop mismatch (H1).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.baseline_unet import BaselineUNet


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    mcfg = cfg.get("model", {})
    feats = tuple(mcfg.get("features", (32, 32, 64, 128, 256, 32)))
    model = BaselineUNet(
        in_channels=mcfg.get("in_channels", 1),
        features=feats,
        dropout=0.0,
        use_uncertainty=False,
        norm=mcfg.get("norm", "instance"),
    )
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    res = model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("uncertainty_head")]
    assert not missing and not res.unexpected_keys, (
        f"STRICT-LOAD FAILED: {len(missing)} missing, {len(res.unexpected_keys)} unexpected"
    )
    n_loaded = len(sd) - len(res.unexpected_keys)
    model.eval().to(device)
    return model, n_loaded, len(sd), cfg, feats


def hist_summary(a, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)):
    h, _ = np.histogram(a, bins=bins)
    frac = h / a.size
    return {f"[{bins[i]:.1f},{bins[i+1]:.1f})": round(float(frac[i]), 4) for i in range(len(h))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/baseline_p40/best_model.pth")
    ap.add_argument("--case", default="CHUM-012")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--out", default="results/model0_validation")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(4)

    ct_path = Path(args.processed_dir) / f"{args.case}_ct.npy"
    tgt_path = Path(args.processed_dir) / "targets" / f"{args.case}_gaussian_heatmap.npy"
    mask_path = Path(args.processed_dir) / "targets" / f"{args.case}_binary.npy"
    ct = np.load(ct_path).astype(np.float32)
    tgt = np.load(tgt_path).astype(np.float32) if tgt_path.exists() else None
    mask = np.load(mask_path).astype(np.float32) if mask_path.exists() else None

    model, n_loaded, n_total, cfg, feats = load_model(args.ckpt, device)

    x = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    heat_prob = out["heatmap"].squeeze().cpu().numpy().astype(np.float32)   # correct (single sigmoid)
    heat_double = 1.0 / (1.0 + np.exp(-heat_prob))                          # buggy (double sigmoid)

    report = {
        "case": args.case,
        "checkpoint": args.ckpt,
        "features": list(feats),
        "params_loaded": f"{n_loaded}/{n_total}",
        "ct_shape": list(ct.shape),
        "ct_range": [round(float(ct.min()), 4), round(float(ct.max()), 4)],
        "CORRECTED_single_sigmoid": {
            "min": round(float(heat_prob.min()), 4),
            "max": round(float(heat_prob.max()), 4),
            "mean": round(float(heat_prob.mean()), 4),
            "frac_ge_0.5": round(float((heat_prob >= 0.5).mean()), 4),
            "frac_le_0.05": round(float((heat_prob <= 0.05).mean()), 4),
            "histogram": hist_summary(heat_prob),
        },
        "BUGGY_double_sigmoid": {
            "min": round(float(heat_double.min()), 4),
            "max": round(float(heat_double.max()), 4),
            "mean": round(float(heat_double.mean()), 4),
            "frac_ge_0.5": round(float((heat_double >= 0.5).mean()), 4),
            "histogram": hist_summary(heat_double),
        },
    }
    if mask is not None and mask.sum() > 0:
        m = mask > 0.5
        report["lesion_vs_background"] = {
            "pred_mean_INSIDE_lesion": round(float(heat_prob[m].mean()), 4),
            "pred_mean_OUTSIDE_lesion": round(float(heat_prob[~m].mean()), 4),
            "lesion_voxel_count": int(m.sum()),
        }
    if tgt is not None:
        # soft dice between prediction and GT gaussian target thresholded at 0.5
        p = (heat_prob >= 0.5).astype(np.float32)
        g = (tgt >= 0.5).astype(np.float32)
        inter = float((p * g).sum())
        dice = (2 * inter) / (p.sum() + g.sum() + 1e-8)
        report["dice_vs_gt_gaussian@0.5"] = round(float(dice), 4)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"{args.case}_heatmap_corrected.npy", heat_prob)
    with open(outdir / f"{args.case}_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
