import React from "react";
import { Freeze, Interactive, Sequence, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Video } from "@remotion/media";
import POUR from "../../data/task_pour.json";
import { BODY, C, DISPLAY, EASE } from "./ui";
import { STOPS, Tip, clamp, lerp, line, ramp, sm } from "./converge";

// Pour convergence video: diffusion attempt(s), then flow-matching attempt(s). Each attempt plays the real rollout (recording camera), pauses once at one policy call and replays the
// sampler steps of 32 sampled chunks. Drawn lines = path of the RIGHT HAND (it holds the bottle) that the simulator produces when the sampled chunk is applied from the recorded state
// (pour_sim.py), projected into the recording camera. There are no water particles in this env; the pour is judged from hand pose, tilt and bottle mouth vs cup opening.
type Model = "diffusion" | "flow";
type Pt = number[];
type Marker = { label: string; kind: "square" | "cross"; t0: number; t1: number; px: number[] | number[][] };
type Metrics = { end16_to_opening_m: number; mouth_now_to_opening_m: number; spread_mm: number; end16_move_m: number };
type Replay = { call: number; step: number; iter: Pt[][][]; labels: number[]; metrics: Metrics };
type Attempt = { env: number; clip: string; success: boolean; t_succ: number | null; t_stage4: number | null; t_pour_done: number | null; cut: number; last_call: number; rate_pre: number; rate_post: number; actual_px: Pt[]; region_px: Pt[][]; stage: number[]; finals: Pt[][][]; markers: Marker[]; replay: Replay };
type Data = { hz: number; vw: number; vh: number; models: Record<Model, { ddim_steps: number; checkpoint: string; attempts: Attempt[] }> };
const D = POUR as unknown as Data;
const VW = 1440, VH = Math.round((1440 * D.vh) / D.vw), K = 1.3;   // K: overlay line widths are in video pixels (848 wide here, 640 in the other task videos)
const PD = 45, HOLD0 = 4, HOLDEND = 22;
const NIT = (m: Model) => D.models[m].attempts[0].replay.iter.length - 1;
const PER = (m: Model) => (NIT(m) >= 10 ? 3 : 6);
type Seg = { kind: "pause" | "play" | "hold"; f0: number; f1: number; model: Model; att: number; step0: number; rate: number };
const buildSegs = (): Seg[] => {
  const segs: Seg[] = []; let f = 0;
  (["diffusion", "flow"] as Model[]).forEach((model) => D.models[model].attempts.forEach((at, ai) => {
    const r = at.replay, nIt = NIT(model), pd = Math.max(PD, HOLD0 + PER(model) * nIt + 14);
    if (r.step > 0) { const n = Math.round(r.step / at.rate_pre); segs.push({ kind: "play", f0: f, f1: f + n, model, att: ai, step0: 0, rate: at.rate_pre }); f += n; }
    segs.push({ kind: "pause", f0: f, f1: f + pd, model, att: ai, step0: r.step, rate: 0 }); f += pd;
    segs.push({ kind: "play", f0: f, f1: f + 16, model, att: ai, step0: r.step, rate: 1 }); f += 16;
    const n = Math.round((at.cut - r.step - 16) / at.rate_post); segs.push({ kind: "play", f0: f, f1: f + n, model, att: ai, step0: r.step + 16, rate: at.rate_post }); f += n;
    segs.push({ kind: "hold", f0: f, f1: f + HOLDEND, model, att: ai, step0: at.cut, rate: 0 }); f += HOLDEND;
  }));
  return segs;
};
const SEGS = buildSegs();
export const POUR_FRAMES = SEGS[SEGS.length - 1].f1;

type State = { seg: Seg; si: number; mode: "pause" | "play"; iter: number; step: number; q: number; at: Attempt };
const stateAt = (f: number): State => {
  let si = SEGS.findIndex((s) => f >= s.f0 && f < s.f1); if (si < 0) si = SEGS.length - 1;
  const seg = SEGS[si], at = D.models[seg.model].attempts[seg.att], nIt = NIT(seg.model);
  if (seg.kind === "pause") {
    const u = clamp((f - seg.f0 - HOLD0) / PER(seg.model), 0, nIt), i = Math.min(nIt - 1, Math.floor(u));
    const e = interpolate(clamp((u - i) * 1.25, 0, 1), [0, 1], [0, 1], { easing: EASE });
    return { seg, si, mode: "pause", iter: u >= nIt ? nIt : i + e, step: at.replay.step, q: at.replay.call, at };
  }
  const step = seg.step0 + (f - seg.f0) * seg.rate;
  return { seg, si, mode: "play", iter: nIt, step, q: Math.min(at.last_call, Math.floor(step / 16)), at };
};
const STAGE = ["approach to grasp", "grasp", "lift both", "move bottle to the cup", "pour (hold the tilt)", "stop pouring", "move to place pose", "place and release", "retreat after release", "success stage reached", "back at start"];

const MarkerDraw: React.FC<{ m: Marker; step: number }> = ({ m, step }) => {
  if (step < m.t0 || step > m.t1) return null;
  const p = (Array.isArray(m.px[0]) ? (m.px as number[][])[Math.min(step, (m.px as number[][]).length - 1)] : m.px) as Pt;
  const s = 0.8 * K, fs = 15;
  if (m.kind === "square") return (
    <g fontFamily={BODY} fontSize={fs} fontWeight={700}>
      <rect x={p[0] - 9 * K} y={p[1] - 9 * K} width={18 * K} height={18 * K} fill="none" stroke="#000" strokeWidth={4.5 * s} />
      <rect x={p[0] - 9 * K} y={p[1] - 9 * K} width={18 * K} height={18 * K} fill="none" stroke="#ffe34d" strokeWidth={2.5 * s} />
      <text x={p[0] + 13 * K} y={p[1] + 14 * K} stroke="#000" strokeWidth={3.5 * s} paintOrder="stroke" fill="#ffe34d">{m.label}</text>
    </g>);
  return (
    <g fontFamily={BODY} fontSize={fs} fontWeight={700}>
      <g stroke="#000" strokeWidth={5 * s} fill="none"><circle cx={p[0]} cy={p[1]} r={8 * K} /><line x1={p[0] - 13 * K} x2={p[0] + 13 * K} y1={p[1]} y2={p[1]} /><line x1={p[0]} x2={p[0]} y1={p[1] - 13 * K} y2={p[1] + 13 * K} /></g>
      <g stroke="#fff" strokeWidth={2.4 * s} fill="none"><circle cx={p[0]} cy={p[1]} r={8 * K} /><line x1={p[0] - 13 * K} x2={p[0] + 13 * K} y1={p[1]} y2={p[1]} /><line x1={p[0]} x2={p[0]} y1={p[1] - 13 * K} y2={p[1] + 13 * K} /></g>
      <text x={p[0] + 15 * K} y={p[1] - 9 * K} stroke="#000" strokeWidth={3.5 * s} paintOrder="stroke" fill="#fff">{m.label}</text>
    </g>);
};

const Overlay: React.FC<{ model: Model; st: State }> = ({ model, st }) => {
  const at = st.at, dp = model === "diffusion", step = Math.min(at.cut, Math.round(st.step)), pausing = st.mode === "pause", rp = at.replay, nIt = NIT(model);
  const i = Math.min(nIt - 1, Math.floor(st.iter)), fr = st.iter >= nIt ? 1 : st.iter - i, s = 0.8 * K;
  const trail = at.actual_px.slice(Math.max(0, step - 100), step + 1);
  let body: React.ReactNode;
  if (pausing) {
    const a = rp.iter[i], b = rp.iter[Math.min(nIt, i + 1)];
    const cur = a.map((sp, k) => sp.map((p, j) => [lerp(p[0], b[k][j][0], fr), lerp(p[1], b[k][j][1], fr)]));
    const c = dp ? ramp(st.iter / nIt) : C.flow;
    body = (
      <g>
        {cur.slice(1).map((sp, k) => <path key={k} d={sm(sp)} fill="none" stroke={c} strokeWidth={(dp ? 1.3 : 1.2) * s} opacity={dp ? 0.6 : 0.5} />)}
        <path d={sm(cur[0])} fill="none" stroke="#000" strokeWidth={4.6 * s} opacity={0.55} />
        <path d={sm(cur[0])} fill="none" stroke={dp ? c : "#fff"} strokeWidth={2.6 * s} />
        {!dp ? cur.filter((_, k) => k % 2 === 0).map((sp, k2) => <circle key={k2} cx={sp[16][0]} cy={sp[16][1]} r={(k2 === 0 ? 5 : 3.4) * s} fill={k2 === 0 ? "#fff" : C.flow} stroke="#0b3b36" strokeWidth={1.2 * s} />) : null}
      </g>
    );
  } else {
    const fin = at.finals[st.q];
    body = (
      <g>
        {fin.slice(1).map((sp, k) => <path key={k} d={sm(sp)} fill="none" stroke={dp ? ramp(1) : C.flow} strokeWidth={1.8 * s} opacity={0.6} />)}
        <path d={sm(fin[0])} fill="none" stroke="#000" strokeWidth={6 * s} opacity={0.6} />
        <path d={sm(fin[0])} fill="none" stroke={dp ? ramp(1) : "#fff"} strokeWidth={3.2 * s} />
        <polyline points={line(trail)} fill="none" stroke="#000" strokeWidth={5 * s} opacity={0.5} />
        <polyline points={line(trail)} fill="none" stroke="#fff" strokeWidth={2.4 * s} />
      </g>
    );
  }
  const reg = at.region_px[Math.min(step, at.region_px.length - 1)];
  return (
    <svg viewBox={`0 0 ${D.vw} ${D.vh}`} width={VW} height={VH} style={{ position: "absolute", left: 0, top: 0 }}>
      <clipPath id="pclip"><rect width={D.vw} height={D.vh} /></clipPath>
      <g clipPath="url(#pclip)">
        <g><polygon points={line(reg)} fill="#4be38a" fillOpacity={0.22} stroke="#000" strokeWidth={3.6 * s} opacity={0.6} /><polygon points={line(reg)} fill="none" stroke="#4be38a" strokeWidth={2 * s} /></g>
        {body}
        {at.markers.map((m, k) => <MarkerDraw key={k} m={m} step={step} />)}
        <g transform={`translate(${at.actual_px[step][0]} ${at.actual_px[step][1]}) scale(${K * 0.75}) translate(${-at.actual_px[step][0]} ${-at.actual_px[step][1]})`}><Tip p={at.actual_px[step]} /></g>
      </g>
    </svg>
  );
};

const cm = (m: number) => (m * 100).toFixed(1);
const Prog: React.FC<{ model: Model; iter: number }> = ({ model, iter }) => {
  const nIt = NIT(model), dp = model === "diffusion", w = 420, h = 150, col = dp ? C.dp : C.flow;
  return (
    <svg width={w} height={h}>
      <rect width={w} height={h} rx={10} fill={C.panel} stroke={C.line} strokeWidth={2} />
      <text x={14} y={34} fontFamily={BODY} fontSize={26} fontWeight={600} fill={col}>{dp ? "DDIM: noise → action" : "Flow time t"}</text>
      <line x1={30} x2={w - 30} y1={78} y2={78} stroke={C.line} strokeWidth={6} strokeLinecap="round" />
      <line x1={30} x2={30 + (iter / nIt) * (w - 60)} y1={78} y2={78} stroke={col} strokeWidth={6} strokeLinecap="round" />
      {Array.from({ length: nIt + 1 }, (_, k) => <circle key={k} cx={30 + (k / nIt) * (w - 60)} cy={78} r={k <= iter + 1e-6 ? 7 : 5} fill={k <= iter + 1e-6 ? col : C.panel} stroke={col} strokeWidth={2.5} />)}
      <text x={w / 2} y={122} fontFamily={BODY} fontSize={23} fill={C.ink} textAnchor="middle">{dp ? `${D.models.diffusion.ddim_steps} DDIM steps, 10 of them shown` : `${D.models.flow.ddim_steps} Euler steps`}</text>
    </svg>
  );
};

const INFO: Record<Model, string> = { diffusion: "bc_lag1200, 51.5 % held-out success (50 DDIM steps, seed 201, 200 eps)", flow: "fm_lag1200, 57.0 % held-out success (5 Euler steps, seed 201, 200 eps)" };
const Inner: React.FC = () => {
  const f = useCurrentFrame();
  const st = stateAt(Math.min(f, POUR_FRAMES - 1)), seg = st.seg, at = st.at, model = seg.model;
  const dp = model === "diffusion", col = dp ? C.dp : C.flow, nAtt = D.models[model].attempts.length;
  const nIt = NIT(model), i = Math.min(nIt - 1, Math.floor(st.iter)), pausing = st.mode === "pause", done = st.iter >= nIt, step = Math.round(st.step), rp = at.replay, m = rp.metrics;
  const nextPause = st.si + 1 < SEGS.length && SEGS[st.si + 1].kind === "pause" && SEGS[st.si + 1].model === model && SEGS[st.si + 1].att === seg.att;
  const ov = seg.kind === "pause" ? Math.min(1, (f - seg.f0) / 5) : seg.kind === "play" && nextPause ? Math.min(1, Math.max(0, (seg.f1 - f) / 6)) : 1;
  const first = SEGS.findIndex((x) => x.model === model && x.att === seg.att), last = SEGS.length - 1 - [...SEGS].reverse().findIndex((x) => x.model === model && x.att === seg.att);
  const vis = Math.min(interpolate(f - SEGS[first].f0, [0, 12], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }), interpolate(f, [SEGS[last].f1 - 10, SEGS[last].f1], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  const ok = at.success && at.t_succ !== null && step >= at.t_succ;
  const lab = rp.labels[Math.min(nIt, i + 1)], lab0 = rp.labels[i];
  const stg = STAGE[Math.min(10, at.stage[Math.min(at.cut, step)])];
  let big: string, formula: string, note: string;
  if (pausing && !done) {
    big = dp ? `t = ${lab0} → ${lab}` : `t: ${lab0.toFixed(1)} → ${lab.toFixed(1)}`;
    formula = dp ? "x ← √ᾱprev · x̂0 + √(1 − ᾱprev) · ε̂   (every 5th of 50 steps shown)" : "x(t + dt) = x(t) + dt · v̂(x, t),  dt = 0.2";
    note = `policy call ${rp.call + 1}, env step ${rp.step}: the robot waits while the policy samples`;
  } else if (pausing) {
    big = dp ? "t = 0: clean chunks" : "t = 1: action chunks";
    formula = `The 16 steps (${(16 / D.hz).toFixed(2)} s) move the bottle mouth ${cm(m.end16_move_m)} cm and end ${cm(m.end16_to_opening_m)} cm from the cup opening (now ${cm(m.mouth_now_to_opening_m)} cm)`;
    note = `32 samples; their end points spread ${Math.round(m.spread_mm)} mm. Stage: ${stg}`;
  } else {
    big = `env step ${step}`;
    formula = ok ? `Success at step ${at.t_succ}` : `policy call ${st.q + 1}: 32 samples, one is run`;
    note = `stage: ${stg}`;
  }
  const HUD: React.CSSProperties = { position: "absolute", background: "rgba(255,255,255,0.88)", borderRadius: 10, padding: "6px 16px", fontFamily: BODY, fontSize: 30, fontWeight: 600, color: C.ink };
  return (
    <div style={{ position: "absolute", inset: 0, fontFamily: BODY }}>
      <div style={{ position: "absolute", left: 0, top: 0, width: VW, height: VH, overflow: "hidden", background: "#000", opacity: vis }}>
        {SEGS.map((s, k) => {
          const a = D.models[s.model].attempts[s.att], base = `task_pour_${s.model}_${a.clip}`;
          return (
            <Sequence key={k} from={s.f0} durationInFrames={s.f1 - s.f0} layout="none">
              {s.kind === "play" && s.rate > 1 ? <Video src={staticFile(`${base}_x${s.rate}.mp4`)} muted trimBefore={s.step0 / s.rate} style={{ width: VW, height: VH, objectFit: "cover" }} />
                : s.kind === "play" ? <Video src={staticFile(`${base}.mp4`)} muted trimBefore={s.step0} style={{ width: VW, height: VH, objectFit: "cover" }} />
                : <Freeze frame={0}><Video src={staticFile(`${base}.mp4`)} muted trimBefore={s.step0} style={{ width: VW, height: VH, objectFit: "cover" }} /></Freeze>}
            </Sequence>);
        })}
        <div style={{ opacity: ov }}><Overlay model={model} st={st} /></div>
        <div style={{ ...HUD, left: 20, top: 18, background: C.ink, color: "#fff" }}>Pour · attempt {seg.att + 1} / {nAtt}</div>
        <div style={{ ...HUD, left: 330, top: 18 }}>{pausing ? (dp ? `DDIM state ${Math.min(nIt, i + 1)} / ${nIt}` : `Euler step ${Math.min(nIt, i + 1)} / ${nIt}`) : `env step ${step}`}</div>
        {pausing ? <div style={{ ...HUD, left: 20, top: 74, background: col, color: "#fff" }}>paused at policy call {rp.call + 1}</div> : null}
        {ok && !pausing ? <div style={{ ...HUD, left: 20, top: 74, background: C.ok, color: "#fff" }}>success</div> : null}
      </div>
      <div style={{ position: "absolute", left: 0, top: VH, width: VW, height: 1080 - VH, background: C.panel, borderTop: `2px solid ${C.line}`, padding: "14px 22px", boxSizing: "border-box", opacity: vis, fontSize: 23, lineHeight: 1.28, color: C.ink }}>
        {dp ? (
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <span>noise</span><div style={{ width: 220, height: 16, borderRadius: 8, background: `linear-gradient(90deg, ${STOPS.join(",")})` }} /><span>clean. Lines: path of the bottle mouth for each of 32 sampled chunks; thick = the one that runs.</span>
          </div>
        ) : <div>Lines: path of the bottle mouth for each of 32 sampled chunks; thick white = the one that runs; dots: mouth after 16 steps.</div>}
        <div style={{ marginTop: 6 }}>Yellow square: cup opening (the cup is held by the left hand). Green shape: pour region of the task, the half cylinder (radius 3.75 cm, from 2 cm below to 10 cm above the cup opening, bottle side) that the bottle mouth must enter during the pour stage. White ring: bottle mouth.</div>
        <div style={{ marginTop: 6, color: C.muted, fontSize: 21 }}>Pour success = the env's stage 9: both objects released upright on the table and both hands retreated a few cm (returning to the start pose is not required). Not a water measure: this simulator has no water. Early noisy sampler states snap the grip to open/closed, so in those lines the bottle can be dropped.</div>
      </div>
      <div style={{ position: "absolute", left: VW + 24, top: 26, width: 1920 - VW - 48, opacity: vis > 0.2 ? 1 : 0 }}>
        <div style={{ fontFamily: DISPLAY, fontWeight: 800, fontSize: 64, lineHeight: 1.0, color: col }}>{dp ? "Diffusion" : "Flow matching"}</div>
        <div style={{ fontSize: 24, color: C.muted, marginTop: 8, lineHeight: 1.2 }}>{INFO[model]}</div>
        <div style={{ marginTop: 16 }}><Prog model={model} iter={st.iter} /></div>
        <div style={{ fontFamily: DISPLAY, fontWeight: 800, fontSize: 34, lineHeight: 1.05, color: pausing ? col : C.ink, marginTop: 20 }}>{big}</div>
        <div style={{ fontSize: 26, color: C.ink, marginTop: 10, lineHeight: 1.22, fontWeight: 500 }}>{formula}</div>
        <div style={{ fontSize: 22, color: C.muted, marginTop: 10, lineHeight: 1.25 }}>{note}</div>
        <div style={{ fontSize: 21, color: C.muted, marginTop: 12, lineHeight: 1.25 }}>
          Lines are the bottle-mouth path (carried by the right hand) the simulator produces for each sampled chunk (both arms are commanded; the 14-D chunk is applied from the recorded state: joints, cup, bottle and actuator lag restored), drawn in the camera. Smoothed for display. A chunk covers 16 steps = {(16 / D.hz).toFixed(2)} s.
        </div>
      </div>
    </div>
  );
};
export const TaskPourScene = Interactive.create({ Component: Inner, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
