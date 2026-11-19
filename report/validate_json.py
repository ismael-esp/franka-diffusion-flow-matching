"""End-to-end check of data/overlay_lift_e2.json (what the scenes draw) against the simulator: predicted gripper path of the executed chunk
(final_all[q][0], 16 points = positions after actions 0..15) vs actual fingertip pixels at env steps s+1..s+16; montage of calls."""
import json, numpy as np, imageio.v2 as io, PIL.Image as I, PIL.ImageDraw as D, sys
J=json.load(open("data/overlay_lift_e2.json")); D_="../runs/denoise_v3"; S=sys.argv[1]
for name in ("diffusion","flow"):
    m=J["models"][name]; fin=np.array(m["final_all"])[:,0]; act=np.array(m["actual_px"]); qs=np.load(f"{D_}/{name}_iterates.npz")["query_step"]
    e8=[];e16=[]
    for c,s in enumerate(qs):
        if s+16>=250: continue
        e8.append(np.linalg.norm(fin[c,:8]-act[s+1:s+9],axis=-1).mean()); e16.append(np.linalg.norm(fin[c,:16]-act[s+1:s+17],axis=-1).mean())
    e8,e16=np.array(e8),np.array(e16)
    print(name,"mean px err over actions 0-7: median %.1f mean %.1f max %.1f | over all 16 steps: median %.1f mean %.1f max %.1f"%(np.median(e8),e8.mean(),e8.max(),np.median(e16),e16.mean(),e16.max()))
    print("   calls 1-8:",[round(v) for v in e8[:8]]," calls 9-30 median %.1f max %.1f"%(np.median(e8[8:30]),e8[8:30].max()))
    print("   16-step end point err calls 9-30 median %.1f"%np.median([np.linalg.norm(fin[c,15]-act[qs[c]+16]) for c in range(8,30)]))
    r=io.get_reader(f"{D_}/{name}_e2_ok.mp4"); tiles=[]
    for c in (0,8,15,29):
        s=int(qs[c]); fr=I.fromarray(r.get_data(s)).resize((640,480)); dr=D.Draw(fr)
        dr.line([tuple(p) for p in act[s:s+17]],fill=(255,255,255),width=2)      # actual future path
        pts=[tuple(p) for p in fin[c]]; dr.line(pts,fill=(255,60,0) if name=="diffusion" else (0,200,170),width=5)
        x,y=act[s]; dr.ellipse([x-8,y-8,x+8,y+8],outline=(255,255,0),width=3); tiles.append(fr)
    sh=I.new("RGB",(1280,960)); [sh.paste(t,((i%2)*640,(i//2)*480)) for i,t in enumerate(tiles)]; sh.save(f"{S}/val_{name}.png")
