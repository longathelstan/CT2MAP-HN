"""Fast head-neck depth diagnostic: measure GTV depth below body-top in mm,
WITHOUT 1mm resampling (uses original z-spacing). CT-only body-top landmark."""
import sys, glob, os, json, numpy as np, SimpleITK as sitk
sys.path.insert(0, ".")
from src.dataio.preprocess import standardize_orientation, compute_body_mask

root="/data/lowngworkspace/hecktor_raw"
all_cases=sorted([os.path.basename(x) for x in glob.glob(f"{root}/*-*") if os.path.isdir(x)])
# stratified: take up to 20 per center
from collections import defaultdict
bycen=defaultdict(list)
for c in all_cases: bycen[c.rsplit("-",1)[0]].append(c)
cases=[]
for cen,lst in bycen.items(): cases+=lst[:20]
print("n_cases:", len(cases), "centers:", list(bycen.keys()), flush=True)

rows=[]; top_narrow=0; errs=0
for i,cid in enumerate(cases):
    d=f"{root}/{cid}"
    try:
        ct=sitk.ReadImage(f"{d}/{cid}__CT.nii.gz")
        gt=sitk.ReadImage(f"{d}/{cid}.nii.gz")
        ct=standardize_orientation(ct,"RAS"); gt=standardize_orientation(gt,"RAS")
        zsp=ct.GetSpacing()[2]  # mm per slice along z (index 2 in sitk xyz -> array axis0)
        a=sitk.GetArrayFromImage(ct).astype(np.float32)
        g=sitk.GetArrayFromImage(gt).astype(np.uint8)
        if a.shape!=g.shape:
            # resample mask onto ct grid via nearest
            gt2=sitk.Resample(gt,ct,sitk.Transform(),sitk.sitkNearestNeighbor,0,gt.GetPixelID())
            g=sitk.GetArrayFromImage(gt2).astype(np.uint8)
        body=compute_body_mask(a,-500.0)
        zc=body.sum(axis=(1,2))
        bz=np.argwhere(zc>0); bz0,bz1=int(bz.min()),int(bz.max()); top=bz1
        at=float(zc[max(bz0,top-int(30/zsp)):top+1].mean()); am=float(zc[bz0:bz1+1].mean())
        if at<am: top_narrow+=1
        gz=np.argwhere(g.any(axis=(1,2)))
        if gz.size==0:
            errs+=1; continue
        gz0,gz1=int(gz.min()),int(gz.max())
        rows.append(dict(cid=cid, Z=int(a.shape[0]), zsp=round(zsp,3), top=top,
                         gz0=gz0, gz1=gz1,
                         depth_near_mm=round((top-gz1)*zsp,1),
                         depth_far_mm=round((top-gz0)*zsp,1)))
    except Exception as e:
        errs+=1
        print("ERR", cid, repr(e), flush=True)
    if (i+1)%20==0: print(f"  {i+1}/{len(cases)}", flush=True)

far=np.array([r["depth_far_mm"] for r in rows])
near=np.array([r["depth_near_mm"] for r in rows])
print(f"=== n_ok={len(rows)} errs={errs} top_narrower_than_mid={top_narrow}/{len(rows)} ===")
print("GTV inferior-most depth below body-top (mm): min=%.0f p50=%.0f p90=%.0f p95=%.0f p99=%.0f max=%.0f"%(
    far.min(),np.percentile(far,50),np.percentile(far,90),np.percentile(far,95),np.percentile(far,99),far.max()))
print("GTV superior-most depth (mm): min=%.0f max=%.0f"%(near.min(),near.max()))
json.dump(rows, open("scripts/_hn_depth_fast.json","w"), indent=1)
print("saved scripts/_hn_depth_fast.json")
