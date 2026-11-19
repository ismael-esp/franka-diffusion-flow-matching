import numpy as np, sys
from geom import fk, norm_to_q, project
from grav import settle
MODE=sys.argv[3] if len(sys.argv)>3 else 'settle'
LAG=int(sys.argv[4]) if len(sys.argv)>4 else 3
map_=(lambda q:settle(q)) if MODE=='settle' else (lambda q:q)
D=sys.argv[1] if len(sys.argv)>1 else "../runs/denoise_v3"; env=int(sys.argv[2]) if len(sys.argv)>2 else 2
for name in ("diffusion","flow"):
    d=np.load(f"{D}/{name}_iterates.npz"); ke=list(d["keep_envs"]).index(env)
    lo,hi=d["joint_low"].astype(float),d["joint_high"].astype(float)
    ch=d["many_iter"][:,-1,ke,0].astype(float)       # [Q,16,8] driver final chunk
    ex=d["executed"][env].astype(float); q_act=d["qpos"][env].astype(float); qs=d["query_step"]
    print(name,"chunk-vs-executed (norm, joints) max abs diff:",max(np.abs(ch[c,:min(8,250-s),:7]-ex[s:s+8,:7]).max() for c,s in enumerate(qs)))
    act_px=project(fk(q_act))[0]
    errs=[];far=[]
    for c,s in enumerate(qs):
        pred=project(fk(map_(norm_to_q(ch[c,:,:7],lo,hi))))[0]            # [16,2]
        n=min(8,250-s-1)
        e=[np.linalg.norm(pred[k]-act_px[min(249,s+k+LAG)]) for k in range(n)]   # commanded k vs achieved at step s+k+1
        errs.append((np.mean(e),np.max(e)))
        far.append(np.linalg.norm(pred[15]-act_px[min(249,s+15+LAG)]))
    errs=np.array(errs)
    print(name,"per-call mean px err: median %.1f  max %.1f ; per-call max: median %.1f max %.1f ; end-of-16 err median %.1f max %.1f"%(np.median(errs[:,0]),errs[:,0].max(),np.median(errs[:,1]),errs[:,1].max(),np.median(far),np.max(far)))
    print("  per-call mean err:", " ".join("%d"%round(v) for v in errs[:,0]))
    for c in (0,8,15,29): print("  call",c+1,"mean/max px %.1f/%.1f end16 %.1f"%(errs[c,0],errs[c,1],far[c]))
