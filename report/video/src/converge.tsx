import React, { useContext } from "react";
import { Freeze, Interactive, Sequence, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Video } from "@remotion/media";
import OV from "../../data/overlay2.json";
import OVS from "../../data/overlay2_single.json";
import { BODY, C, DISPLAY, EASE } from "./ui";

// Real recorded data (record_denoise.py, lift, seed 41; sim_chunks.py; make_overlay_data2.py). Every drawn path is the simulator's gripper path for a sampled
// chunk (FK of the joints the env reaches when that chunk is applied from the recorded state), projected into the video camera. Two attempts (env 4, env 6).
type Model = "diffusion" | "flow";
type Pt = number[];
type Metrics = { kind: string; call: number; step: number; end16_to_cube_m: number; end16_to_goal_m: number; close_index: number; close_to_cube_m: number | null; samples_end16_std_mm: number; tcp_to_cube_now_m: number };
type Replay = { call: number; step: number; kind: string; iter: Pt[][][]; vel?: number[][][]; field?: number[][][][]; metrics: Metrics };
type Attempt = { env: number; t_succ: number; t_close: number; cut: number; last_call: number; actual_px: Pt[]; cube_px: Pt[]; goal_px: Pt; finals: Pt[][][]; replays: Replay[] };
type Variant = "multi" | "single";
type DT = { alphas_cumprod: number[]; grid: { gx: number; gy: number; x0: number; x1: number; y0: number; y1: number }; models: Record<Model, { attempts: Attempt[] }> };
const DS: Record<Variant, DT> = { multi: OV as unknown as DT, single: OVS as unknown as DT };
const VariantCtx = React.createContext<Variant>("multi");
const useD = () => DS[useContext(VariantCtx)];

const VW = 1440, PD = 45, HOLD0 = 4, PER = 3, HOLDEND = 22;
type Seg = { kind: "pause" | "play" | "hold"; f0: number; f1: number; att: number; ri: number; step0: number; rate: number };
const buildSegs = (variant: Variant, model: Model): Seg[] => {
  const D = DS[variant], R1 = variant === "single" ? 1 : 2;
  const segs: Seg[] = [];
  let f = 0;
  D.models[model].attempts.forEach((at, ai) => {
    let cur = 0;
    const cutE = at.cut - (at.cut % 2);
    at.replays.forEach((r, ri) => {
      if (cur < r.step) { const n = Math.round((r.step - cur) / R1); segs.push({ kind: "play", f0: f, f1: f + n, att: ai, ri: -1, step0: cur, rate: R1 }); f += n; }
      segs.push({ kind: "pause", f0: f, f1: f + PD, att: ai, ri, step0: r.step, rate: 0 }); f += PD;
      segs.push({ kind: "play", f0: f, f1: f + 8, att: ai, ri: -1, step0: r.step, rate: 1 }); f += 8; cur = r.step + 8;
    });
    if (cur < cutE) { const n = Math.round((cutE - cur) / R1); segs.push({ kind: "play", f0: f, f1: f + n, att: ai, ri: -1, step0: cur, rate: R1 }); f += n; }
    segs.push({ kind: "hold", f0: f, f1: f + HOLDEND, att: ai, ri: -1, step0: cutE, rate: 0 }); f += HOLDEND;
  });
  return segs;
};
const SEGS_BY: Record<Variant, Record<Model, Seg[]>> = { multi: { diffusion: buildSegs("multi", "diffusion"), flow: buildSegs("multi", "flow") }, single: { diffusion: buildSegs("single", "diffusion"), flow: buildSegs("single", "flow") } };
const lastF = (v: Variant, m: Model) => SEGS_BY[v][m][SEGS_BY[v][m].length - 1].f1;
export const FRAMES_BY: Record<Model, number> = { diffusion: lastF("multi", "diffusion"), flow: lastF("multi", "flow") };
export const FRAMES_SINGLE: Record<Model, number> = { diffusion: lastF("single", "diffusion"), flow: lastF("single", "flow") };
export const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
export const ease = (p: number) => interpolate(p, [0, 1], [0, 1], { easing: EASE, extrapolateLeft: "clamp", extrapolateRight: "clamp" });
export const lerp = (a: number, b: number, p: number) => a + (b - a) * p;

type State = { seg: Seg; si: number; mode: "pause" | "play"; att: number; q: number; iter: number; step: number };
const stateAt = (variant: Variant, model: Model, f: number): State => {
  const D = DS[variant], SEGS = SEGS_BY[variant][model];
  let si = SEGS.findIndex((s) => f >= s.f0 && f < s.f1); if (si < 0) si = SEGS.length - 1;
  const seg = SEGS[si], at = D.models[model].attempts[seg.att];
  if (seg.kind === "pause") {
    const r = at.replays[seg.ri];
    const u = clamp((f - seg.f0 - HOLD0) / PER, 0, 10), i = Math.min(9, Math.floor(u));
    return { seg, si, mode: "pause", att: seg.att, q: r.call, iter: u >= 10 ? 10 : i + ease(clamp((u - i) * 1.25, 0, 1)), step: r.step };
  }
  const step = seg.step0 + (f - seg.f0) * seg.rate;
  return { seg, si, mode: "play", att: seg.att, q: Math.min(at.last_call, Math.floor(step / 8)), iter: 10, step };
};

// plasma-like ramp, dark purple (noise) -> orange -> yellow (clean)
export const STOPS = ["#2a0a6b", "#7e03a8", "#cc4778", "#f89540", "#f9d62e"];
const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
export const ramp = (p: number) => {
  const x = clamp(p, 0, 1) * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(x)), a = hex(STOPS[i]), b = hex(STOPS[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(lerp(v, b[k], x - i))).join(",")})`;
};
// Catmull-Rom spline through the real points (display only: the points themselves are untouched)
export const sm = (p: Pt[]) => {
  let d = `M${p[0][0].toFixed(1)},${p[0][1].toFixed(1)}`;
  for (let i = 0; i < p.length - 1; i++) {
    const a = p[Math.max(0, i - 1)], b = p[i], c = p[i + 1], e = p[Math.min(p.length - 1, i + 2)];
    d += `C${(b[0] + (c[0] - a[0]) / 6).toFixed(1)},${(b[1] + (c[1] - a[1]) / 6).toFixed(1)} ${(c[0] - (e[0] - b[0]) / 6).toFixed(1)},${(c[1] - (e[1] - b[1]) / 6).toFixed(1)} ${c[0].toFixed(1)},${c[1].toFixed(1)}`;
  }
  return d;
};
export const line = (pts: Pt[]) => pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");

// ---- overlay drawn in 640 x 480 video pixels -------------------------------------------------------------------------------
export const Tip: React.FC<{ p: Pt }> = ({ p }) => <g><circle cx={p[0]} cy={p[1]} r={7} fill="none" stroke="#000" strokeWidth={4} /><circle cx={p[0]} cy={p[1]} r={7} fill="none" stroke="#fff" strokeWidth={2} /></g>;

export const Arrow: React.FC<{ x: number; y: number; dx: number; dy: number; color: string; w?: number; op?: number }> = ({ x, y, dx, dy, color, w = 2, op = 1 }) => {
  const L = Math.hypot(dx, dy);
  if (L < 3) return null;
  const ux = dx / L, uy = dy / L, hd = Math.min(8, L * 0.45);
  return <g opacity={op} stroke={color} strokeWidth={w} fill={color}>
    <line x1={x} y1={y} x2={x + dx} y2={y + dy} strokeLinecap="round" />
    <polygon stroke="none" points={`${x + dx},${y + dy} ${x + dx - ux * hd - uy * hd * 0.5},${y + dy - uy * hd + ux * hd * 0.5} ${x + dx - ux * hd + uy * hd * 0.5},${y + dy - uy * hd - ux * hd * 0.5}`} />
  </g>;
};

export const FlowField: React.FC<{ pz: Replay; iter: number; op: number }> = ({ pz, iter, op }) => {
  const D = useD();
  const g = D.grid, i = Math.min(9, Math.floor(iter)), fr = iter - i;
  const fa = pz.field![i], fb = pz.field![Math.min(9, i + 1)];
  const at = (gy: number, gx: number): number[] => fa[gy][gx].map((v, k) => (k < 2 ? lerp(v, fb[gy][gx][k], fr) : v));
  const cw = (g.x1 - g.x0) / (g.gx - 1), ch = (g.y1 - g.y0) / (g.gy - 1);
  const sample = (x: number, y: number) => {                 // bilinear field lookup
    const u = (x - g.x0) / cw, v = (y - g.y0) / ch;
    if (u < 0 || v < 0 || u > g.gx - 1 || v > g.gy - 1) return null;
    const x0 = Math.min(g.gx - 2, Math.floor(u)), y0 = Math.min(g.gy - 2, Math.floor(v)), a = u - x0, b = v - y0;
    const c = [at(y0, x0), at(y0, x0 + 1), at(y0 + 1, x0), at(y0 + 1, x0 + 1)];
    const w = [(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b];
    return [0, 1, 2].map((k) => c.reduce((s, cc, j) => s + cc[k] * w[j], 0));
  };
  const cells: React.ReactNode[] = [];
  for (let gy = 0; gy < g.gy; gy++) for (let gx = 0; gx < g.gx; gx++) {
    const [dx, dy, w] = at(gy, gx);
    const conf = clamp(w / 25, 0, 1), sp = clamp(Math.hypot(dx, dy) / 12, 0, 1);
    if (conf < 0.4) continue;
    const cx = g.x0 + gx * cw, cy = g.y0 + gy * ch;
    cells.push(<rect key={`h${gx}-${gy}`} x={cx - cw / 2} y={cy - ch / 2} width={cw} height={ch} fill={C.flow} opacity={0.3 * conf * sp} />);
    if (gx % 2 === 0 && gy % 2 === 0) cells.push(<Arrow key={`a${gx}-${gy}`} x={cx} y={cy} dx={dx * 4} dy={dy * 4} color="#0b3b36" w={1.8} op={clamp((conf - 0.3) * 1.5, 0, 0.85)} />);
  }
  const streams: React.ReactNode[] = [];
  for (let gy = 1; gy < g.gy - 1; gy += 4) for (let gx = 1; gx < g.gx - 1; gx += 4) {
    let x = g.x0 + gx * cw, y = g.y0 + gy * ch;
    const pts: Pt[] = [[x, y]];
    for (let s = 0; s < 22; s++) {
      const v = sample(x, y);
      if (!v || v[2] < 8) break;
      const L = Math.hypot(v[0], v[1]);
      if (L < 1) break;
      x += (v[0] / L) * 5; y += (v[1] / L) * 5; pts.push([x, y]);
    }
    if (pts.length > 4) streams.push(<polyline key={`s${gx}-${gy}`} points={line(pts)} fill="none" stroke="#7fe3d6" strokeWidth={1.6} opacity={0.75} />);
  }
  return <g opacity={op}>{cells.slice(0, 1e9)}{streams}</g>;
};

const Markers: React.FC<{ cube: Pt; goal: Pt }> = ({ cube, goal }) => (
  <g fontFamily={BODY} fontSize={13} fontWeight={700}>
    <rect x={cube[0] - 11} y={cube[1] - 11} width={22} height={22} fill="none" stroke="#000" strokeWidth={4.5} />
    <rect x={cube[0] - 11} y={cube[1] - 11} width={22} height={22} fill="none" stroke="#ffe34d" strokeWidth={2.5} />
    <text x={cube[0] + 15} y={cube[1] + 16} stroke="#000" strokeWidth={3.5} paintOrder="stroke" fill="#ffe34d">cube</text>
    <g stroke="#000" strokeWidth={5} fill="none"><circle cx={goal[0]} cy={goal[1]} r={9} /><line x1={goal[0] - 15} x2={goal[0] + 15} y1={goal[1]} y2={goal[1]} /><line x1={goal[0]} x2={goal[0]} y1={goal[1] - 15} y2={goal[1] + 15} /></g>
    <g stroke="#fff" strokeWidth={2.4} fill="none"><circle cx={goal[0]} cy={goal[1]} r={9} /><line x1={goal[0] - 15} x2={goal[0] + 15} y1={goal[1]} y2={goal[1]} /><line x1={goal[0]} x2={goal[0]} y1={goal[1] - 15} y2={goal[1] + 15} /></g>
    <text x={goal[0] + 17} y={goal[1] - 10} stroke="#000" strokeWidth={3.5} paintOrder="stroke" fill="#fff">goal</text>
  </g>
);

const Overlay: React.FC<{ model: Model; st: State }> = ({ model, st }) => {
  const D = useD();
  const at = D.models[model].attempts[st.att];
  const dp = model === "diffusion";
  const step = Math.min(at.cut, Math.round(st.step));
  const pausing = st.mode === "pause";
  const rp = pausing ? at.replays[st.seg.ri] : null;
  const i = Math.min(9, Math.floor(st.iter)), fr = st.iter >= 10 ? 1 : st.iter - i;
  const trail = at.actual_px.slice(0, step + 1);
  let body: React.ReactNode;
  if (rp) {
    const a = rp.iter[i], b = rp.iter[Math.min(10, i + 1)];
    const cur = a.map((s, k) => s.map((p, j) => [lerp(p[0], b[k][j][0], fr), lerp(p[1], b[k][j][1], fr)]));
    const c = dp ? ramp(st.iter / 10) : C.flow;
    body = (
      <g>
        {!dp && st.iter < 10 ? <FlowField pz={rp} iter={st.iter} op={1} /> : null}
        {cur.slice(1).filter((_, k) => dp || k % 3 === 0).map((s, k) => <path key={k} d={sm(s)} fill="none" stroke={c} strokeWidth={dp ? 1.3 : 1.1} opacity={dp ? 0.6 : 0.4} />)}
        <path d={sm(cur[0])} fill="none" stroke="#000" strokeWidth={4.6} opacity={0.55} />
        <path d={sm(cur[0])} fill="none" stroke={dp ? c : "#fff"} strokeWidth={2.6} />
        {!dp ? cur.filter((_, k) => k % 2 === 0).map((s, k2) => {
          const k = k2 * 2;
          const tr = Array.from({ length: i + 1 }, (_, j) => rp.iter[j][k][9]).concat([s[9]]);
          const v = rp.vel![i][k];
          return <g key={k}>
            <polyline points={line(tr)} fill="none" stroke="#fff" strokeWidth={1.2} opacity={0.55} />
            {st.iter < 10 ? <Arrow x={s[9][0]} y={s[9][1]} dx={v[0] * 4} dy={v[1] * 4} color="#fff" w={1.4} op={0.9} /> : null}
            <circle cx={s[9][0]} cy={s[9][1]} r={k === 0 ? 5 : 3.4} fill={k === 0 ? "#fff" : C.flow} stroke="#0b3b36" strokeWidth={1.2} />
          </g>;
        }) : null}
      </g>
    );
  } else {
    const fin = at.finals[st.q];
    body = (
      <g>
        {fin.slice(1).map((s, k) => <path key={k} d={sm(s)} fill="none" stroke={dp ? ramp(1) : C.flow} strokeWidth={1.8} opacity={0.6} />)}
        <path d={sm(fin[0])} fill="none" stroke="#000" strokeWidth={6} opacity={0.6} />
        <path d={sm(fin[0])} fill="none" stroke={dp ? ramp(1) : "#fff"} strokeWidth={3.2} />
        <polyline points={line(trail)} fill="none" stroke="#000" strokeWidth={5} opacity={0.5} />
        <polyline points={line(trail)} fill="none" stroke="#fff" strokeWidth={2.4} />
      </g>
    );
  }
  return (
    <svg viewBox="0 0 640 480" width={VW} height={1080} style={{ position: "absolute", left: 0, top: 0 }}>
      <clipPath id="vclip"><rect width={640} height={480} /></clipPath>
      <g clipPath="url(#vclip)">{body}<Markers cube={at.cube_px[step]} goal={at.goal_px} /><Tip p={at.actual_px[step]} /></g>
    </svg>
  );
};

// ---- side panel ----------------------------------------------------------------------------------------------------------
export const Inset: React.FC<{ model: Model; st: State }> = ({ model, st }) => {
  const D = useD();
  const w = 420, h = 200;
  if (model === "flow") {
    const t = Math.min(10, st.iter) / 10;
    return (
      <svg width={w} height={h}>
        <rect width={w} height={h} rx={10} fill={C.panel} stroke={C.line} strokeWidth={2} />
        <text x={14} y={34} fontFamily={BODY} fontSize={26} fontWeight={600} fill={C.flow}>Flow time t</text>
        <line x1={30} x2={w - 30} y1={110} y2={110} stroke={C.line} strokeWidth={6} strokeLinecap="round" />
        <line x1={30} x2={30 + t * (w - 60)} y1={110} y2={110} stroke={C.flow} strokeWidth={6} strokeLinecap="round" />
        {Array.from({ length: 11 }, (_, k) => <circle key={k} cx={30 + (k / 10) * (w - 60)} cy={110} r={k / 10 <= t + 1e-6 ? 7 : 5} fill={k / 10 <= t + 1e-6 ? C.flow : C.panel} stroke={C.flow} strokeWidth={2.5} />)}
        <text x={30} y={158} fontFamily={BODY} fontSize={24} fill={C.muted}>0 noise</text>
        <text x={w - 30} y={158} fontFamily={BODY} fontSize={24} fill={C.muted} textAnchor="end">1 action</text>
        <text x={w / 2} y={186} fontFamily={BODY} fontSize={24} fill={C.ink} textAnchor="middle">10 Euler steps</text>
      </svg>
    );
  }
  const ab = D.alphas_cumprod;
  const P = { l: 16, r: 16, t: 52, b: 40 };
  const X = (t: number) => P.l + ((99 - t) / 99) * (w - P.l - P.r);
  const Y = (v: number) => P.t + (1 - v) * (h - P.t - P.b);
  const sig = (t: number) => Math.sqrt(ab[clamp(Math.round(t), 0, 99)]);
  const noi = (t: number) => Math.sqrt(1 - ab[clamp(Math.round(t), 0, 99)]);
  const tNow = Math.max(0, 90 - 10 * Math.min(10, st.iter));
  const curve = (fn: (t: number) => number) => Array.from({ length: 100 }, (_, k) => `${X(99 - k).toFixed(1)},${Y(fn(99 - k)).toFixed(1)}`).join(" ");
  return (
    <svg width={w} height={h}>
      <rect width={w} height={h} rx={10} fill={C.panel} stroke={C.line} strokeWidth={2} />
      <text x={14} y={34} fontFamily={BODY} fontSize={26} fontWeight={600} fill={C.dp}>Noise → signal</text>
      <polyline points={curve(sig)} fill="none" stroke={C.dp} strokeWidth={4} />
      <polyline points={curve(noi)} fill="none" stroke={C.teacher} strokeWidth={4} strokeDasharray="10 7" />
      {[90, 80, 70, 60, 50, 40, 30, 20, 10, 0].map((t) => <circle key={t} cx={X(t)} cy={Y(sig(t))} r={4} fill={C.dp} />)}
      <line x1={X(tNow)} x2={X(tNow)} y1={P.t - 6} y2={h - P.b} stroke={C.ink} strokeWidth={3} />
      <circle cx={X(tNow)} cy={Y(sig(tNow))} r={8} fill={C.ink} />
      <text x={P.l} y={h - 10} fontFamily={BODY} fontSize={24} fill={C.muted}>t = 99</text>
      <text x={w - P.r} y={h - 10} fontFamily={BODY} fontSize={24} fill={C.muted} textAnchor="end">t = 0</text>
      <text x={w - 14} y={34} fontFamily={BODY} fontSize={22} textAnchor="end"><tspan fill={C.dp}>signal</tspan><tspan fill={C.teacher}> - - noise</tspan></text>
    </svg>
  );
};

const cm = (m: number) => Math.round(m * 100);
const Inner: React.FC<{ model: Model }> = ({ model }) => {
  const variant = useContext(VariantCtx), D = useD();
  const f = useCurrentFrame();
  const dp = model === "diffusion";
  const col = dp ? C.dp : C.flow;
  const SEGS = SEGS_BY[variant][model];
  const st = stateAt(variant, model, Math.min(f, lastF(variant, model) - 1));
  const at = D.models[model].attempts[st.att];
  const seg = st.seg;
  const i = Math.min(9, Math.floor(st.iter));
  const pausing = st.mode === "pause";
  const done = st.iter >= 10;
  const step = Math.round(st.step);
  const rp = pausing ? at.replays[seg.ri] : null;
  const times = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0];
  const nextPause = st.si + 1 < SEGS.length && SEGS[st.si + 1].kind === "pause" && SEGS[st.si + 1].att === seg.att;
  const ov = seg.kind === "pause" ? Math.min(1, (f - seg.f0) / 5) : seg.kind === "play" && nextPause ? Math.min(1, Math.max(0, (seg.f1 - f) / 6)) : 1;
  const firstOfAtt = SEGS.findIndex((x) => x.att === seg.att), lastOfAtt = SEGS.length - 1 - [...SEGS].reverse().findIndex((x) => x.att === seg.att);
  const fadeIn = interpolate(f - SEGS[firstOfAtt].f0, [0, 12], [0, 1], { extrapolateRight: "clamp", extrapolateLeft: "clamp" });
  const fadeOut = seg.att === 0 ? interpolate(f, [SEGS[lastOfAtt].f1 - 10, SEGS[lastOfAtt].f1], [1, 0], { extrapolateRight: "clamp", extrapolateLeft: "clamp" }) : 1;
  const vis = Math.min(fadeIn, fadeOut) * interpolate(f, [0, 10], [0.0, 1], { extrapolateRight: "clamp" });
  const success = step >= at.t_succ;

  let big: string, formula: string, note: string;
  if (rp && !done) {
    big = dp ? `noise level t = ${times[i]} → ${i < 9 ? times[i + 1] : "clean"}` : `t: ${(i / 10).toFixed(1)} → ${((i + 1) / 10).toFixed(1)}`;
    formula = dp ? "x ← √ᾱprev · x̂0 + √(1 − ᾱprev) · ε̂" : "x(t + dt) = x(t) + dt · v̂(x, t),  dt = 0.1";
    note = `policy call ${rp.call + 1}, env step ${rp.step}: the robot waits while the policy samples`;
  } else if (rp) {
    const m = rp.metrics;
    big = dp ? "t = 0: clean chunks" : "t = 1: action chunks";
    formula = m.kind === "closest" ? `Closest approach before the grasp: the 16 steps (0.32 s) end ${(m.end16_to_cube_m * 100).toFixed(1)} cm from the cube (the gripper is now ${cm(m.tcp_to_cube_now_m)} cm away)`
      : m.kind === "approach" ? `Approach: the 16 steps (0.32 s) end ${cm(m.end16_to_cube_m)} cm from the cube`
      : m.kind === "grasp" ? `Grasp: gripper closes at action ${m.close_index + 1}, ${m.close_to_cube_m === null ? "-" : cm(m.close_to_cube_m)} cm from the cube centre; the chunk ends ${cm(m.end16_to_cube_m)} cm from it`
      : `Lift: the chunk ends ${cm(m.end16_to_goal_m)} cm from the goal marker`;
    note = `32 samples; their end points spread ${Math.round(m.samples_end16_std_mm)} mm`;
  } else {
    big = `env step ${step}`;
    formula = success ? `Success at step ${at.t_succ}: cube within 5 cm of the goal` : `policy call ${st.q + 1}: 32 samples, one is run`;
    note = variant === "single" ? "robot running (1 env step = 1 frame, about 0.6 x real time)" : "robot running (1 env step = 1 frame after a replay, then 2 per frame)";
  }
  const HUD: React.CSSProperties = { position: "absolute", background: "rgba(255,255,255,0.88)", borderRadius: 10, padding: "6px 16px", fontFamily: BODY, fontSize: 30, fontWeight: 600, color: C.ink };
  return (
    <div style={{ position: "absolute", inset: 0, fontFamily: BODY }}>
      <div style={{ position: "absolute", left: 0, top: 0, width: VW, height: 1080, overflow: "hidden", background: "#000", opacity: vis }}>
        {SEGS.map((s, k) => (
          <Sequence key={k} from={s.f0} durationInFrames={s.f1 - s.f0} layout="none">
            {s.kind === "play" && s.rate === 2 ? (
              <Video src={src2sel(variant, model, s.att)} muted trimBefore={s.step0 / 2} style={{ width: VW, height: 1080, objectFit: "cover" }} />
            ) : s.kind === "play" ? (
              <Video src={src1sel(variant, model, s.att)} muted trimBefore={s.step0} style={{ width: VW, height: 1080, objectFit: "cover" }} />
            ) : (
              <Freeze frame={0}><Video src={src1sel(variant, model, s.att)} muted trimBefore={s.step0} style={{ width: VW, height: 1080, objectFit: "cover" }} /></Freeze>
            )}
          </Sequence>
        ))}
        <div style={{ opacity: ov }}><Overlay model={model} st={st} /></div>
        <div style={{ ...HUD, left: 20, top: 18, background: C.ink, color: "#fff" }}>Attempt {st.att + 1} / 2</div>
        <div style={{ ...HUD, left: 250, top: 18 }}>{pausing ? (dp ? `DDIM step ${Math.min(10, i + 1)} / 10` : `Euler step ${Math.min(10, i + 1)} / 10`) : `env step ${step}`}</div>
        {pausing ? <div style={{ ...HUD, left: 20, top: 74, background: col, color: "#fff" }}>paused at policy call {rp!.call + 1}</div> : null}
        {success && !pausing ? <div style={{ ...HUD, left: 20, top: 74, background: C.ok, color: "#fff" }}>success: cube at the goal</div> : null}
        <div style={{ ...HUD, left: 20, bottom: 18, fontSize: 24, fontWeight: 500, lineHeight: 1.25, maxWidth: 1380 }}>
          {dp ? (
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <span>noise t = 90</span>
              <div style={{ width: 220, height: 16, borderRadius: 8, background: `linear-gradient(90deg, ${STOPS.join(",")})` }} />
              <span>clean t = 0. Lines: gripper path of each of 32 sampled chunks; thick = the one that runs</span>
            </div>
          ) : (
            <div>Heat, arrows (length = 4 × one Euler step), flow lines: how the gripper path moves per Euler step. Dots: chunk step 9 of each sample; thick white = the one that runs</div>
          )}
          <div>Yellow square: the cube (its real pose). White cross: the lift goal (target position of the cube). Ring: the gripper.</div>
        </div>
      </div>
      <div style={{ position: "absolute", left: VW + 24, top: 26, width: 1920 - VW - 48 }}>
        <div style={{ fontFamily: DISPLAY, fontWeight: 800, fontSize: 64, lineHeight: 1.0, color: col }}>{dp ? "Diffusion" : "Flow matching"}</div>
        <div style={{ fontSize: 26, color: C.muted, marginTop: 8 }}>{dp ? "Gaussian noise → action chunk, DDIM" : "noise → action chunk, Euler steps along v̂"}</div>
        <div style={{ marginTop: 20 }}><Inset model={model} st={st} /></div>
        <div style={{ fontFamily: DISPLAY, fontWeight: 800, fontSize: 38, lineHeight: 1.05, color: pausing ? col : C.ink, marginTop: 22, whiteSpace: "nowrap" }}>{big}</div>
        <div style={{ fontSize: 27, color: C.ink, marginTop: 10, lineHeight: 1.22, fontWeight: 500 }}>{formula}</div>
        <div style={{ fontSize: 23, color: C.muted, marginTop: 12, lineHeight: 1.25 }}>{note}</div>
        <div style={{ fontSize: 22, color: C.muted, marginTop: 14, lineHeight: 1.25 }}>
          Lines are the gripper path the simulator produces for each sampled chunk (FK of the joints reached), drawn in the camera. Curves are smoothed for display (spline through the real 16 points). A chunk covers only 16 steps = 0.32 s, so during the approach it ends part-way to the cube.
        </div>
      </div>
    </div>
  );
};
const src1sel = (v: Variant, m: Model, a: number) => staticFile(`denoise_${m}_e${DS[v].models[m].attempts[a].env}.mp4`);
const src2sel = (v: Variant, m: Model, a: number) => staticFile(`denoise_${m}_e${DS[v].models[m].attempts[a].env}_x2.mp4`);

export const ConvergeInner: React.FC<{ model: Model }> = Inner;
const DiffInner: React.FC = () => <Inner model="diffusion" />;
const withV = (v: Variant, model: Model): React.FC => () => <VariantCtx.Provider value={v}><Inner model={model} /></VariantCtx.Provider>;
const FlowInnerS: React.FC = () => <Inner model="flow" />;
export const DiffusionConvergeScene = Interactive.create({ Component: DiffInner, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
export const FlowIntegrateScene = Interactive.create({ Component: FlowInnerS, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
export const CONVERGE_FRAMES = FRAMES_BY;

export const DiffusionConvergeSingleScene = Interactive.create({ Component: withV("single", "diffusion"), layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
export const FlowIntegrateSingleScene = Interactive.create({ Component: withV("single", "flow"), layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
