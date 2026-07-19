import sys, glob, os, json, numpy as np, SimpleITK as sitk
sys.path.insert(0, ".")
from src.dataio.preprocess import (standardize_orientation, resample_volume,
                                    align_volumes, compute_body_mask)
root="/data/lowngworkspace/hecktor_raw"
centers=["CHUM","CHUP","CHUS","CHUV","HGJ","HMR","MDA","USZ"]
cases=[]
for c in centers:
    m=sorted(glob.glob(f"{root}/{c}-*"))
    cases += [os.path.basename(x) for x in m[:6]]
print("n_cases:", len(cases), flush=True)
rows=[]; top_narrow=0
for i,cid in enumerate(cases):
    d=f"{root}/{cid}"
    try:
        ct=sitk.ReadImage(f"{d}/{cid}__CT.nii.gz"); gtv=sitk.ReadImage(f"{d}/{cid}.nii.gz")
        ct=standardize_orientation(ct,"RAS"); gtv=standardize_orientation(gtv,"RAS")
        ct,_,gtv=align_volumes(ct,None,gtv)
        ct=resample_volume(ct,(1.0,1.0,1.0),"linear"); gtv=resample_volume(gtv,(1.0,1.0,1.0),"nearest")
        a=sitk.GetArrayFromImage(ct).astype(np.float32); g=sitk.GetArrayFromImage(gtv).astype(np.uint8)
        body=compute_body_mask(a,-500.0)
        zc=body.sum(axis=(1,2))
        bz=np.argwhere(zc>0); bz0,bz1=int(bz.min()),int(bz.max()); top=bz1
        area_top=float(zc[max(bz0,top-30):top+1].mean()); area_mid=float(zc[bz0:bz1].mean())
        if area_top<area_mid: top_narrow+=1
        gz=np.argwhere(g.any(axis=(1,2))); gz0,gz1=int(gz.min()),int(gz.max())
        rows.append(dict(cid=cid, Z=int(a.shape[0]), top=top, gz0=gz0, gz1=gz1,
                         depth_near=int(top-gz1), depth_far=int(top-gz0),
                         area_top=area_top, area_mid=area_mid))
    except Exception as e:
        print("ERR", cid, e, flush=True)
    if (i+1)%8==0: print(f"  {i+1}/{len(cases)}", flush=True)
depths=np.array([r["depth_far"] for r in rows]); near=np.array([r["depth_near"] for r in rows])
print("=== top narrower than middle in", top_narrow, "/", len(rows), "cases ===")
print("GTV inferior-most depth below body-top (mm): min=%.0f p50=%.0f p95=%.0f max=%.0f"%(
    depths.min(), np.percentile(depths,50), np.percentile(depths,95), depths.max()))
print("GTV superior-most depth below body-top (mm): min=%.0f max=%.0f"%(near.min(), near.max()))
json.dump(rows, open("scripts/_hn_crop_diag.json","w"), indent=1)
print("saved")
