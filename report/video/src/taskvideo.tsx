import React from "react";
import { Freeze, Interactive, Sequence, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Video } from "@remotion/media";
import LIFT from "../../data/task_lift.json";
import DRAWER from "../../data/task_drawer.json";
import STACK from "../../data/task_stack.json";
import { BODY, C, DISPLAY, EASE } from "./ui";
import { FlowField, Inset, STOPS, Tip, clamp, lerp, line, ramp, sm } from "./converge";

// One merged convergence video per task: diffusion attempts, then flow-matching attempts, each with ONE replay of the 10 sampling steps at the policy call
// whose simulator-predicted 16-step chunk end gets closest to the marker before the grasp. Same data pipeline as the lift clips (record_task.py ->
// sim_task.py -> make_task_data.py): every drawn line is the gripper path the simulator produces for a sampled chunk, projected into the task's video camera.
type Model = "diffusion" | "flow";
type Pt = number[];
type Marker = { label: string; kind: "square" | "cross"; t0: number; t1: number; px: number[] | number[][] };
type Metrics = { end16_to_marker_m: number; end16_to_target_m: number; now_to_marker_m: number; close_index: number; close_to_marker_m: number | null; spread_mm: number };
type Replay = { call: number; step: number; iter: Pt[][][]; vel?: number[][][]; field?: number[][][][]; metrics: Metrics };
type Attempt = { env: number; clip: string; success: boolean; t_succ: number | null; cut: number; last_call: number; rate: number; actual_px: Pt[]; finals: Pt[][][]; markers: Marker[]; replay: Replay };
type Data = { hz: number; models: Record<Model, { attempts: Attempt[] }> };
const DATA: Record<string, Data> = { lift: LIFT as unknown as Data, drawer: DRAWER as unknown as Data, stack: STACK as unknown as Data };
export type TaskKey = "lift" | "drawer" | "stack";

const INFO: Record<TaskKey, { title: string; marker: string; legend: string; ckpt: Record<Model, string>; note: string }> = {
  lift: { title: "Lift", marker: "cube", legend: "Yellow square: the cube. White cross: the lift goal (target position of the cube).",
    ckpt: { diffusion: "dp_lift, 62.8 % held-out success", flow: "fm_lift, 78.2 % held-out success" }, note: "Both models drive the same two initial states (env 4, env 6 of an 8-env batch); both succeed." },
  drawer: { title: "Drawer", marker: "handle", legend: "Yellow square: the handle (grasp point). White cross: the handle position when the drawer is 30 cm open (open target, the success condition).",
    ckpt: { diffusion: "drawer_s/dp, 92.2 % held-out success", flow: "fm_drawer, 92.3 % held-out success" }, note: "Both models drive the same two initial cabinet poses (env 0, env 4 of a 12-env batch); both succeed." },
  stack: { title: "Stack", marker: "cube 2", legend: "Yellow square: the cube to place. White cross: its target pose on the cube below (4.68 cm above it).",
    ckpt: { diffusion: "DAgger round 2, 42.1 % held-out success", flow: "fm_stack (demos only), 7.3 % held-out success" }, note: "Flow succeeds in about 1 of 16 envs, so its two successes were searched in 32 envs; the diffusion attempts have similar cube starts but are not the same states (both models rarely succeed on one state)." },
};
export const TASK_NOTE = INFO;

const VW = 1440, PD = 45, HOLD0 = 4, PER = 3, HOLDEND = 22;
type Seg = { kind: "pause" | "play" | "hold"; f0: number; f1: number; model: Model; att: number; step0: number; rate: number };
const buildSegs = (task: TaskKey): Seg[] => {
  const segs: Seg[] = []; let f = 0;
  (["diffusion", "flow"] as Model[]).forEach((model) => DATA[task].models[model].attempts.forEach((at, ai) => {
    const R = at.rate, r = at.replay, cutE = at.cut - (at.cut % 2);
    if (r.step > 0) { const n = Math.round(r.step / R); segs.push({ kind: "play", f0: f, f1: f + n, model, att: ai, step0: 0, rate: R }); f += n; }
    segs.push({ kind: "pause", f0: f, f1: f + PD, model, att: ai, step0: r.step, rate: 0 }); f += PD;
    segs.push({ kind: "play", f0: f, f1: f + 8, model, att: ai, step0: r.step, rate: 1 }); f += 8;
    const n = Math.round((cutE - r.step - 8) / R); segs.push({ kind: "play", f0: f, f1: f + n, model, att: ai, step0: r.step + 8, rate: R }); f += n;
    segs.push({ kind: "hold", f0: f, f1: f + HOLDEND, model, att: ai, step0: cutE, rate: 0 }); f += HOLDEND;
  }));
  return segs;
};
const SEGS: Record<TaskKey, Seg[]> = { lift: buildSegs("lift"), drawer: buildSegs("drawer"), stack: buildSegs("stack") };
export const TASK_FRAMES: Record<TaskKey, number> = { lift: SEGS.lift[SEGS.lift.length - 1].f1, drawer: SEGS.drawer[SEGS.drawer.length - 1].f1, stack: SEGS.stack[SEGS.stack.length - 1].f1 };

type State = { seg: Seg; si: number; mode: "pause" | "play"; iter: number; step: number; q: number; at: Attempt };
const stateAt = (task: TaskKey, f: number): State => {
  const S = SEGS[task]; let si = S.findIndex((s) => f >= s.f0 && f < s.f1); if (si < 0) si = S.length - 1;
  const seg = S[si], at = DATA[task].models[seg.model].attempts[seg.att];
  if (seg.kind === "pause") {
    const u = clamp((f - seg.f0 - HOLD0) / PER, 0, 10), i = Math.min(9, Math.floor(u));
    const e = interpolate(clamp((u - i) * 1.25, 0, 1), [0, 1], [0, 1], { easing: EASE });
    return { seg, si, mode: "pause", iter: u >= 10 ? 10 : i + e, step: at.replay.step, q: at.replay.call, at };
  }
  const step = seg.step0 + (f - seg.f0) * seg.rate;
  return { seg, si, mode: "play", iter: 10, step, q: Math.min(at.last_call, Math.floor(step / 8)), at };
};

const MarkerDraw: React.FC<{ m: Marker; step: number }> = ({ m, step }) => {
  if (step < m.t0 || step > m.t1) return null;
  const p = (Array.isArray(m.px[0]) ? (m.px as number[][])[Math.min(step, (m.px as number[][]).length - 1)] : m.px) as Pt;
  if (m.kind === "square") return (
    <g fontFamily={BODY} fontSize={13} fontWeight={700}>
      <rect x={p[0] - 11} y={p[1] - 11} width={22} height={22} fill="none" stroke="#000" strokeWidth={4.5} />
      <rect x={p[0] - 11} y={p[1] - 11} width={22} height={22} fill="none" stroke="#ffe34d" strokeWidth={2.5} />
      <text x={p[0] + 15} y={p[1] + 16} stroke="#000" strokeWidth={3.5} paintOrder="stroke" fill="#ffe34d">{m.label}</text>
    </g>);
  return (
    <g fontFamily={BODY} fontSize={13} fontWeight={700}>
      <g stroke="#000" strokeWidth={5} fill="none"><circle cx={p[0]} cy={p[1]} r={9} /><line x1={p[0] - 15} x2={p[0] + 15} y1={p[1]} y2={p[1]} /><line x1={p[0]} x2={p[0]} y1={p[1] - 15} y2={p[1] + 15} /></g>
      <g stroke="#fff" strokeWidth={2.4} fill="none"><circle cx={p[0]} cy={p[1]} r={9} /><line x1={p[0] - 15} x2={p[0] + 15} y1={p[1]} y2={p[1]} /><line x1={p[0]} x2={p[0]} y1={p[1] - 15} y2={p[1] + 15} /></g>
      <text x={p[0] + 17} y={p[1] - 10} stroke="#000" strokeWidth={3.5} paintOrder="stroke" fill="#fff">{m.label}</text>
    </g>);
};

const Overlay: React.FC<{ model: Model; st: State }> = ({ model, st }) => {
  const at = st.at, dp = model === "diffusion", step = Math.min(at.cut, Math.round(st.step)), pausing = st.mode === "pause", rp = at.replay;
  const i = Math.min(9, Math.floor(st.iter)), fr = st.iter >= 10 ? 1 : st.iter - i;
  const trail = at.actual_px.slice(0, step + 1);
  let body: React.ReactNode;
  if (pausing) {
    const a = rp.iter[i], b = rp.iter[Math.min(10, i + 1)];
    const cur = a.map((s, k) => s.map((p, j) => [lerp(p[0], b[k][j][0], fr), lerp(p[1], b[k][j][1], fr)]));
    const c = dp ? ramp(st.iter / 10) : C.flow;
    body = (
      <g>
        {!dp && st.iter < 10 ? <FlowField pz={rp as never} iter={st.iter} op={1} /> : null}
        {cur.slice(1).filter((_, k) => dp || k % 3 === 0).map((s, k) => <path key={k} d={sm(s)} fill="none" stroke={c} strokeWidth={dp ? 1.3 : 1.1} opacity={dp ? 0.6 : 0.4} />)}
        <path d={sm(cur[0])} fill="none" stroke="#000" strokeWidth={4.6} opacity={0.55} />
        <path d={sm(cur[0])} fill="none" stroke={dp ? c : "#fff"} strokeWidth={2.6} />
        {!dp ? cur.filter((_, k) => k % 2 === 0).map((s, k2) => {
          const k = k2 * 2, tr = Array.from({ length: i + 1 }, (_, j) => rp.iter[j][k][9]).concat([s[9]]), v = rp.vel![i][k];
          return <g key={k}>
            <polyline points={line(tr)} fill="none" stroke="#fff" strokeWidth={1.2} opacity={0.55} />
            {st.iter < 10 ? <Arrow4 x={s[9][0]} y={s[9][1]} dx={v[0] * 4} dy={v[1] * 4} /> : null}
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
      <clipPath id="tclip"><rect width={640} height={480} /></clipPath>
      <g clipPath="url(#tclip)">{body}{at.markers.map((m, k) => <MarkerDraw key={k} m={m} step={step} />)}<Tip p={at.actual_px[step]} /></g>
    </svg>
  );
};
const Arrow4: React.FC<{ x: number; y: number; dx: number; dy: number }> = ({ x, y, dx, dy }) => {
  const L = Math.hypot(dx, dy); if (L < 3) return null;
  const ux = dx / L, uy = dy / L, hd = Math.min(8, L * 0.45);
  return <g opacity={0.9} stroke="#fff" strokeWidth={1.4} fill="#fff"><line x1={x} y1={y} x2={x + dx} y2={y + dy} strokeLinecap="round" />
    <polygon stroke="none" points={`${x + dx},${y + dy} ${x + dx - ux * hd - uy * hd * 0.5},${y + dy - uy * hd + ux * hd * 0.5} ${x + dx - ux * hd + uy * hd * 0.5},${y + dy - uy * hd - ux * hd * 0.5}`} /></g>;
};

const cm = (m: number) => Math.round(m * 100);
const Inner: React.FC<{ task: TaskKey }> = ({ task }) => {
  const f = useCurrentFrame();
  const info = INFO[task], D = DATA[task], SG = SEGS[task];
  const st = stateAt(task, Math.min(f, TASK_FRAMES[task] - 1)), seg = st.seg, at = st.at, model = seg.model;
  const dp = model === "diffusion", col = dp ? C.dp : C.flow;
  const nAtt = D.models[model].attempts.length;
  const i = Math.min(9, Math.floor(st.iter)), pausing = st.mode === "pause", done = st.iter >= 10, step = Math.round(st.step), rp = at.replay, m = rp.metrics;
  const times = [90, 80, 70, 60, 50, 40, 30, 20, 10, 0];
  const horizon = (16 / D.hz).toFixed(2);
  const nextPause = st.si + 1 < SG.length && SG[st.si + 1].kind === "pause" && SG[st.si + 1].model === model && SG[st.si + 1].att === seg.att;
  const ov = seg.kind === "pause" ? Math.min(1, (f - seg.f0) / 5) : seg.kind === "play" && nextPause ? Math.min(1, Math.max(0, (seg.f1 - f) / 6)) : 1;
  const first = SG.findIndex((x) => x.model === model && x.att === seg.att), last = SG.length - 1 - [...SG].reverse().findIndex((x) => x.model === model && x.att === seg.att);
  const vis = Math.min(interpolate(f - SG[first].f0, [0, 12], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }), interpolate(f, [SG[last].f1 - 10, SG[last].f1], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  const ok = at.success && at.t_succ !== null && step >= at.t_succ;
  let big: string, formula: string, note: string;
  if (pausing && !done) {
    big = dp ? `noise level t = ${times[i]} → ${i < 9 ? times[i + 1] : "clean"}` : `t: ${(i / 10).toFixed(1)} → ${((i + 1) / 10).toFixed(1)}`;
    formula = dp ? "x ← √ᾱprev · x̂0 + √(1 − ᾱprev) · ε̂" : "x(t + dt) = x(t) + dt · v̂(x, t),  dt = 0.1";
    note = `policy call ${rp.call + 1}, env step ${rp.step}: the robot waits while the policy samples`;
  } else if (pausing) {
    big = dp ? "t = 0: clean chunks" : "t = 1: action chunks";
    formula = `Closest approach before the grasp: the 16 steps (${horizon} s) end ${(m.end16_to_marker_m * 100).toFixed(1)} cm from the ${info.marker} (the gripper is now ${cm(m.now_to_marker_m)} cm away)`;
    note = `32 samples; their end points spread ${Math.round(m.spread_mm)} mm`;
  } else {
    big = `env step ${step}`;
    formula = ok ? `Success at step ${at.t_succ}` : at.success ? `policy call ${st.q + 1}: 32 samples, one is run` : `policy call ${st.q + 1}: 32 samples, one is run (this attempt fails)`;
    note = `robot running (${at.rate === 1 ? "1 env step per frame" : "1 env step per frame after the replay, then 2"})`;
  }
  const HUD: React.CSSProperties = { position: "absolute", background: "rgba(255,255,255,0.88)", borderRadius: 10, padding: "6px 16px", fontFamily: BODY, fontSize: 30, fontWeight: 600, color: C.ink };
  return (
    <div style={{ position: "absolute", inset: 0, fontFamily: BODY }}>
      <div style={{ position: "absolute", left: 0, top: 0, width: VW, height: 1080, overflow: "hidden", background: "#000", opacity: vis }}>
        {SG.map((s, k) => (
          <Sequence key={k} from={s.f0} durationInFrames={s.f1 - s.f0} layout="none">
            {s.kind === "play" && s.rate === 2 ? <Video src={staticFile(`task_${task}_${s.model}_${DATA[task].models[s.model].attempts[s.att].clip}_x2.mp4`)} muted trimBefore={s.step0 / 2} style={{ width: VW, height: 1080, objectFit: "cover" }} />
              : s.kind === "play" ? <Video src={staticFile(`task_${task}_${s.model}_${DATA[task].models[s.model].attempts[s.att].clip}.mp4`)} muted trimBefore={s.step0} style={{ width: VW, height: 1080, objectFit: "cover" }} />
              : <Freeze frame={0}><Video src={staticFile(`task_${task}_${s.model}_${DATA[task].models[s.model].attempts[s.att].clip}.mp4`)} muted trimBefore={s.step0} style={{ width: VW, height: 1080, objectFit: "cover" }} /></Freeze>}
          </Sequence>
        ))}
        <div style={{ opacity: ov }}><Overlay model={model} st={st} /></div>
        <div style={{ ...HUD, left: 20, top: 18, background: C.ink, color: "#fff" }}>{info.title} · attempt {seg.att + 1} / {nAtt}</div>
        <div style={{ ...HUD, left: 360, top: 18 }}>{pausing ? (dp ? `DDIM step ${Math.min(10, i + 1)} / 10` : `Euler step ${Math.min(10, i + 1)} / 10`) : `env step ${step}`}</div>
        {pausing ? <div style={{ ...HUD, left: 20, top: 74, background: col, color: "#fff" }}>paused at policy call {rp.call + 1}</div> : null}
        {ok && !pausing ? <div style={{ ...HUD, left: 20, top: 74, background: C.ok, color: "#fff" }}>success</div> : null}
        {!at.success && !pausing && step >= at.cut ? <div style={{ ...HUD, left: 20, top: 74, background: C.fail, color: "#fff" }}>no success in this episode</div> : null}
        <div style={{ ...HUD, left: 20, bottom: 18, fontSize: 24, fontWeight: 500, lineHeight: 1.25, maxWidth: 1380 }}>
          {dp ? (
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <span>noise t = 90</span>
              <div style={{ width: 220, height: 16, borderRadius: 8, background: `linear-gradient(90deg, ${STOPS.join(",")})` }} />
              <span>clean t = 0. Lines: gripper path of each of 32 sampled chunks; thick = the one that runs</span>
            </div>
          ) : <div>Heat, arrows (length = 4 × one Euler step), flow lines: how the gripper path moves per Euler step. Dots: chunk step 9 of each sample; thick white = the one that runs</div>}
          <div>{info.legend} Ring: the gripper.</div>
        </div>
      </div>
      <div style={{ position: "absolute", left: VW + 24, top: 26, width: 1920 - VW - 48, opacity: vis > 0.2 ? 1 : 0 }}>
        <div style={{ fontFamily: DISPLAY, fontWeight: 800, fontSize: 64, lineHeight: 1.0, color: col }}>{dp ? "Diffusion" : "Flow matching"}</div>
        <div style={{ fontSize: 24, color: C.muted, marginTop: 8 }}>{info.ckpt[model]}</div>
        <div style={{ marginTop: 16 }}><Inset model={model} st={{ iter: st.iter } as never} /></div>
        <div style={{ fontFamily: DISPLAY, fontWeight: 800, fontSize: 38, lineHeight: 1.05, color: pausing ? col : C.ink, marginTop: 20, whiteSpace: "nowrap" }}>{big}</div>
        <div style={{ fontSize: 26, color: C.ink, marginTop: 10, lineHeight: 1.22, fontWeight: 500 }}>{formula}</div>
        <div style={{ fontSize: 22, color: C.muted, marginTop: 10, lineHeight: 1.25 }}>{note}</div>
        <div style={{ fontSize: 21, color: C.muted, marginTop: 12, lineHeight: 1.25 }}>
          Lines are the gripper path the simulator produces for each sampled chunk (FK of the joints reached), drawn in the camera. Curves are smoothed for display (spline through the real 16 points). A chunk covers 16 steps = {horizon} s. {info.note}
        </div>
      </div>
    </div>
  );
};
const mk = (t: TaskKey): React.FC => () => <Inner task={t} />;
export const TaskLiftScene = Interactive.create({ Component: mk("lift"), layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
export const TaskDrawerScene = Interactive.create({ Component: mk("drawer"), layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
export const TaskStackScene = Interactive.create({ Component: mk("stack"), layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
