import numpy as np
from geom import fk, project
R=np.load("../runs/sim_paths.npz")
for name in ("diffusion","flow"):
    d=np.load(f"../runs/denoise_v4/{name}_iterates.npz"); val=R[f"{name}_val"]; idx=R[f"{name}_val_idx"]; qs=d["query_step"]
    px=project(fk(val.astype(float)))[0]
    err=np.full((8,32),np.nan); err16=err.copy()
    for (e,c),p in zip(idx,px):
        s=qs[c]; n=min(16,250-s-1); act=project(fk(d["qpos"][e].astype(float)))[0]
        err[e,c]=np.linalg.norm(p[:min(8,n)]-act[s+1:s+1+min(8,n)],axis=-1).mean()          # executed 8 steps
        err16[e,c]=np.linalg.norm(p[:n]-act[s+1:s+1+n],axis=-1).mean()                         # 16 steps (the last 8 are not executed as planned)
    print(name,"executed-8 error px: calls 1-8 mean %.2f max %.2f | calls 9+ mean %.2f | all-envs median %.2f"%(np.nanmean(err[:,:8]),np.nanmax(err[:,:8]),np.nanmean(err[:,8:]),np.nanmedian(err)))
    for e in (4,6): print("   env",e,"calls 1-8 mean %.2f"%np.nanmean(err[e,:8]), np.round(err[e,:8],1))
