"""Validate the H1 CT-only body crop + head-neck z-window (Model 0 baseline).

For each case: resample to target spacing, run crop_head_neck_roi(method='body')
with the GT mask supplied ONLY to measure GTV retention (the crop itself uses
no mask), and report volume reduction + GTV retention. Expect z-dim ==
round(z_extent_mm/z_spacing)+1 and GTV_retained == 100%.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, SimpleITK as sitk
from src.dataio.preprocess import (
    standardize_orientation, resample_volume, crop_head_neck_roi, align_volumes,
)

ROOT = "/data/lowngworkspace/hecktor_raw"
TS = [1.0, 1.0, 1.0]

def check(cid, z_extent_mm=360.0):
    ct = standardize_orientation(sitk.ReadImage(f"{ROOT}/{cid}/{cid}__CT.nii.gz"), "RAS")
    gt = standardize_orientation(sitk.ReadImage(f"{ROOT}/{cid}/{cid}.nii.gz"), "RAS")
    ct, _, gt = align_volumes(ct, None, gt)
    ct = resample_volume(ct, TS, "linear"); gt = resample_volume(gt, TS, "nearest")
    a = sitk.GetArrayFromImage(ct).astype(np.float32)
    g = sitk.GetArrayFromImage(gt).astype(np.uint8)
    cropped, info = crop_head_neck_roi(
        a, mask=g, margin=(10, 10, 10), method="body",
        hu_threshold=-500.0, z_extent_mm=z_extent_mm, z_spacing_mm=TS[2])
    sl = [slice(s[0], s[1]) for s in info["slices"]]
    ret = 100.0 * (g[sl[0], sl[1], sl[2]] > 0).sum() / max(1, (g > 0).sum())
    return dict(cid=cid, full=a.shape, crop=cropped.shape,
                vox_pct=round(100*cropped.size/a.size, 1), gtv_retained=round(ret, 2))

if __name__ == "__main__":
    cases = sys.argv[1:] or ["CHUM-012", "CHUP-000", "CHUV-001", "MDA-001"]
    for cid in cases:
        print(check(cid))
