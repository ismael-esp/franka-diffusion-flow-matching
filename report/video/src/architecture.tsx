import React from "react";
import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { ARCH, LEARN_DIFFUSION, LEARN_FLOW, type Node } from "./diagrams";
import { BODY, C, DISPLAY, EASE, Headline } from "./ui";

// ---- renders the real diagrams/*.svg geometry (text stripped: the labels below are set large in Remotion) ----------
type Anim = (n: Node, circle: number, circles: number) => { op: number; dash: number | null };
const camel = (k: string) => k.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
const countCircles = (ns: Node[]): number => ns.reduce((s, n) => s + (n.t === "circle" ? 1 : 0) + (n.c ? countCircles(n.c) : 0), 0);

const Nodes: React.FC<{ nodes: Node[]; anim: Anim; counter: { i: number }; total: number }> = ({ nodes, anim, counter, total }) => (
  <>
    {nodes.map((n, k) => {
      const ci = n.t === "circle" ? counter.i++ : -1;
      const { op, dash } = anim(n, ci, total);
      const props: Record<string, unknown> = { key: k };
      for (const [a, v] of Object.entries(n.a)) props[camel(a)] = v;
      props.opacity = op;
      if (dash !== null) { props.pathLength = 1; props.strokeDasharray = 1; props.strokeDashoffset = 1 - dash; props.opacity = 1; }
      return React.createElement(n.t, props, n.c ? <Nodes nodes={n.c} anim={anim} counter={counter} total={total} /> : null);
    })}
  </>
);

const Diagram: React.FC<{ nodes: Node[]; viewBox: string; w: number; h: number; anim: Anim }> = ({ nodes, viewBox, w, h, anim }) => (
  <svg viewBox={viewBox} width={w} height={h} style={{ display: "block", overflow: "hidden" }}>
    <Nodes nodes={nodes} anim={anim} counter={{ i: 0 }} total={countCircles(nodes)} />
  </svg>
);

const firstX = (n: Node): number => {
  if (n.a.x) return parseFloat(n.a.x);
  if (n.a.points) return parseFloat(n.a.points.split(/[ ,]/)[0]);
  return 0;
};
const ramp = (f: number, a: number, d: number) => interpolate(f, [a, a + d], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: EASE });

// Architecture: wipe from left to right (inputs -> encoder -> decoder -> output).
const archAnim = (f: number): Anim => (n) => {
  const p = ramp(f, 16 + (firstX(n) / 1536) * 95, 20);
  return n.t === "polyline" ? { op: 1, dash: p } : { op: p, dash: null };
};

// Learning plots: chunk curves draw in one after another, then the sampling path and its dots.
const learnAnim = (f: number): Anim => (n, ci, total) => {
  if (n.t === "polyline" && n.draw !== undefined) {
    const start = n.draw >= 0 ? 196 + n.draw * 40 : 252;
    return { op: 1, dash: ramp(f, start, n.draw >= 0 ? 28 : 46) };
  }
  if (n.t === "circle") return { op: ramp(f, 252 + (ci / Math.max(1, total - 1)) * 46, 8), dash: null };
  return { op: ramp(f, 176, 18), dash: null };
};

const Cap: React.FC<{ x: number; y: number; w: number; at: number; children: React.ReactNode; color?: string }> = ({ x, y, w, at, children, color = C.ink }) => {
  const f = useCurrentFrame();
  const p = ramp(f, at, 18);
  return <div style={{ position: "absolute", left: x, top: y, width: w, opacity: p, fontFamily: BODY, fontSize: 40, lineHeight: 1.18, fontWeight: 500, color }}>{children}</div>;
};

const Column: React.FC<{ x: number; nodes: Node[]; color: string; title: string; labels: string[]; desc: string; stat: string }> = ({ x, nodes, color, title, labels, desc, stat }) => {
  const f = useCurrentFrame();
  const W = 820, S = W / 745;
  const centres = [68, 220, 372, 524, 676];
  return (
    <div style={{ position: "absolute", left: x, top: 0, width: W, height: 1080 }}>
      <div style={{ position: "absolute", top: 70, fontFamily: DISPLAY, fontWeight: 800, fontSize: 76, color, letterSpacing: -0.5, whiteSpace: "nowrap" }}>{title}</div>
      <div style={{ position: "absolute", top: 172 }}>
        <Diagram nodes={nodes} viewBox="0 34 745 126" w={W} h={126 * S} anim={learnAnim(f)} />
      </div>
      {labels.map((l, i) => (
        <div key={l} style={{ position: "absolute", top: 172 + 126 * S + 6, left: centres[i] * S - 70, width: 140, textAlign: "center", fontFamily: BODY, fontSize: 40, color: C.muted, opacity: ramp(f, 186 + i * 6, 14) }}>{l}</div>
      ))}
      <div style={{ position: "absolute", top: 172 + 126 * S + 62, width: W, fontFamily: BODY, fontSize: 44, lineHeight: 1.2, color: C.ink, fontWeight: 500, opacity: ramp(f, 196, 18) }}>{desc}</div>
      <div style={{ position: "absolute", top: 560, fontFamily: BODY, fontSize: 40, color: C.muted, opacity: ramp(f, 240, 16) }}>Sampling path, schematic</div>
      <div style={{ position: "absolute", top: 610 }}>
        <Diagram nodes={nodes} viewBox="0 285 480 170" w={480 * S} h={170 * S} anim={learnAnim(f)} />
      </div>
      <div style={{ position: "absolute", top: 820, width: W, fontFamily: DISPLAY, fontWeight: 800, fontSize: 68, lineHeight: 1.0, color, opacity: ramp(f, 292, 18), whiteSpace: "nowrap" }}>{stat}</div>
    </div>
  );
};

const HowInner: React.FC = () => {
  const f = useCurrentFrame();
  const arch = interpolate(f, [0, 16, 160, 182], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const learn = interpolate(f, [170, 192], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const AW = 1680, AS = AW / 1536;
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <div style={{ position: "absolute", inset: 0, opacity: arch }}>
        <div style={{ position: "absolute", left: 120, top: 80 }}>
          <Headline size={100}>One network, two ways to learn</Headline>
        </div>
        <div style={{ position: "absolute", left: 120, top: 250 }}>
          <Diagram nodes={ARCH} viewBox="0 0 1536 404" w={AW} h={404 * AS} anim={archAnim(f)} />
        </div>
        <Cap x={120} y={250 + 404 * AS + 22} w={720} at={34}>Two frames in: ResNet18, 32 keypoints, joint state</Cap>
        <Cap x={120 + 790 * AS} y={250 + 404 * AS + 22} w={310} at={72}>5 conditioning tokens</Cap>
        <Cap x={120 + 1070 * AS} y={250 + 404 * AS + 22} w={340} at={96}>4-layer decoder over 16 action tokens</Cap>
        <Cap x={120 + 1380 * AS} y={250 + 404 * AS + 22} w={200} at={118}>16 × 8 out, run 8</Cap>
        <Cap x={120} y={900} w={1680} at={130} color={C.teacher}>17.5 M parameters, the same network for both. It predicts 16 steps and executes 8.</Cap>
      </div>
      <div style={{ position: "absolute", inset: 0, opacity: learn }}>
        <Column x={100} nodes={LEARN_DIFFUSION} color={C.dp} title="Diffusion" labels={["t = 0", "t = 25", "t = 50", "t = 75", "t = 99"]} desc="A noisy demo chunk at 5 noise levels. The network predicts the noise." stat="10 network calls" />
        <Column x={1000} nodes={LEARN_FLOW} color={C.flow} title="Flow matching" labels={["t = 1", "0.75", "0.5", "0.25", "t = 0"]} desc="A straight blend from noise to the demo. The network predicts the velocity." stat="1, 5 or 10 network calls" />
        <div style={{ position: "absolute", left: 100, top: 935, width: 1720, fontFamily: BODY, fontSize: 40, lineHeight: 1.2, color: C.muted, opacity: ramp(f, 316, 20) }}>
          Held-out lift action error: 0.0140 diffusion, 0.0079 flow matching (one run).
        </div>
      </div>
    </div>
  );
};
export const HowScene = Interactive.create({ Component: HowInner, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
