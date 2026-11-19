"""Controller tracking model: the joint-position PD of the env (low stiffness, gravity on) does not reach its targets (steady-state sag up to 0.27 rad
on joints 2 and 4, ~140 px at the gripper). Fit q_achieved(t+L) - target(t) = f(target(t)) from recorded rollouts (ridge, quadratic features)."""
import numpy as np, itertools, json, sys
from geom import fk, norm_to_q, project
def feats(q):
    q=np.asarray(q,float); n=q.shape[-1]
    quad=[q[...,i]*q[...,j] for i in range(n) for j in range(i,n)]
    return np.concatenate([np.ones(q.shape[:-1]+(1,)),q,np.stack(quad,-1),np.sin(q),np.cos(q)],-1)
def load(D):
    T,Q=[],[]
    for name in ("diffusion","flow"):
        d=np.load(f"{D}/{name}_iterates.npz"); lo,hi=d["joint_low"].astype(float),d["joint_high"].astype(float)
        for e in range(d["executed"].shape[0]):
            T.append(norm_to_q(d["executed"][e][:,:7].astype(float),lo,hi)); Q.append(d["qpos"][e].astype(float))
    return np.array(T),np.array(Q),lo,hi
def fit(T,Q,L,lam=1e-2):
    X=np.concatenate([feats(T[i,:250-L]) for i in range(len(T))]); Y=np.concatenate([Q[i,L:]-T[i,:250-L] for i in range(len(T))])
    W=np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@Y); return W
if __name__=="__main__":
    D=sys.argv[1]; T,Q,lo,hi=load(D)
    n=len(T)
    for L in (1,2,3,4,6):
        errs=[];base=[]
        for k in range(n):                      # leave-one-rollout-out
            tr=[i for i in range(n) if i!=k]; W=fit(T[tr],Q[tr],L)
            pred=T[k,:250-L]+feats(T[k,:250-L])@W; act=Q[k,L:]
            pa=project(fk(pred))[0]; pb=project(fk(act))[0]; pc=project(fk(T[k,:250-L]))[0]
            errs.append(np.linalg.norm(pa-pb,axis=-1)); base.append(np.linalg.norm(pc-pb,axis=-1))
        e=np.concatenate(errs); b=np.concatenate(base)
        print(f"L={L}: model median {np.median(e):.1f} px mean {e.mean():.1f} p95 {np.percentile(e,95):.1f} | no-model median {np.median(b):.1f} mean {b.mean():.1f}")
