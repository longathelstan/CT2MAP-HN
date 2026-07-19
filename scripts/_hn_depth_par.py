"""Parallel head-neck depth diagnostic (mm-based, no 1mm resample).
Measures GTV depth below CT-only body-top landmark across many cases."""
import sys, glob, os, json, numpy as np, SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, ".")
from src.dataio.preprocess import standardize_orientation, compute_body_mask

root="/data/lowngworkspace/hecktor_raw"

def one(cid):
    d=f"{root}/{cid}"
    try:
        ct=sitk.ReadImage(f"{d}/{cid}__CT.nii.gz")
        gt=sitk.ReadImage(f"{d}/{cid}.nii.gz")
        ct=standardize_orientation(ct,"RAS"); gt=standardize_orientation(gt,"RAS")
        zsp=ct.GetSpacing()[2]
        a=sitk.GetArrayFromImage(ct).astype(np.float32)
        g=sitk.GetArrayFromImage(gt).astype(np.uint8)
        if a.shape!=g.shape:
            gt2=sitk.Resample(gt,ct,sitk.Transform(),sitk.sitkNearestNeighbor,0,gt.GetPixelID())
            g=sitk.GetArrayFromImage(gt2).astype(np.uint8)
        body=compute_body_mask(a,-500.0)
        zc=body.sum(axis=(1,2))
        bz=np.argwhere(zc>0); bz0,bz1=int(bz.min()),int(bz.max()); top=bz1
        nt=int(30/zsp)
        at=float(zc[max(bz0,top-nt):top+1].mean()); am=float(zc[bz0:bz1+1].mean())
        gz=np.argwhere(g.any(axis=(1,2)))
        if gz.size==0: return {"cid":cid,"err":"empty_gtv"}
        gz0,gz1=int(gz.min()),int(gz.max())
        return dict(cid=cid, Z=int(a.shape[0]), zsp=round(zsp,3), top=top,
                    gz0=gz0, gz1=gz1,
                    depth_near_mm=round((top-gz1)*zsp,1),
                    depth_far_mm=round((top-gz0)*zsp,1),
                    top_narrow=int(at<am))
    except Exception as e:
        return {"cid":cid,"err":repr(e)}

if __name__=="__main__":
    all_cases=sorted([os.path.basename(x) for x in glob.glob(f"{root}/*-*") if os.path.isdir(x)])
    from collections import defaultdict
    bycen=defaultdict(list)
    for c in all_cases: bycen[c.rsplit("-",1)[0]].append(c)
    cases=[]
    for cen,lst in bycen.items(): cases+=lst[:20]
    print("n_cases:", len(cases), "centers:", list(bycen.keys()), flush=True)
    rows=[]; errs=[]
    with ProcessPoolExecutor(max_workers=16) as ex:
        futs={ex.submit(one,c):c for c in cases}
        done=0
        for f in as_completed(futs):
            r=f.result(); done+=1
            if "err" in r: errs.append(r)
            else: rows.append(r)
            if done%20==0: print(f"  {done}/{len(cases)}", flush=True)
    far=np.array([r["depth_far_mm"] for r in rows])
    near=np.array([r["depth_near_mm"] for r in rows])
    tn=sum(r["top_narrow"] for r in rows)
    print(f"=== n_ok={len(rows)} errs={len(errs)} top_narrower={tn}/{len(rows)} ===")
    print("GTV inferior-most depth below body-top (mm): min=%.0f p50=%.0f p90=%.0f p95=%.0f p99=%.0f max=%.0f"%(
        far.min(),np.percentile(far,50),np.percentile(far,90),np.percentile(far,95),np.percentile(far,99),far.max()))
    print("GTV superior-most depth (mm): min=%.0f max=%.0f"%(near.min(),near.max()))
    if errs: print("ERRS:", errs[:5])
    json.dump({"rows":rows,"errs":errs}, open("scripts/_hn_depth_par.json","w"), indent=1)
    print("saved")
