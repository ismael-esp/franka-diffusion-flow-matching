import React from "react";
import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { BODY, C, DISPLAY, EASE } from "./ui";

// Two separate model diagrams. Facts: dp_core/model.py 70-125 (encoder, denoiser), 157-164 (tokens), 166-171 (diffusion
// loss), 179-198 (DDIM); dp_core/flow.py 19-35 (flow loss, Euler). Slate = literally the same code, colour = what differs.
const ramp = (f: number, a: number, d = 16) => interpolate(f, [a, a + d], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: EASE });

const Box: React.FC<{ w: number; h?: number; kind: "same" | "diff" | "plain"; color: string; title: string; lines: string[]; at: number }> = ({ w, h = 176, kind, color, title, lines, at }) => {
  const f = useCurrentFrame();
  const p = ramp(f, at);
  const bg = kind === "same" ? "#e6ecf1" : kind === "diff" ? color + "22" : C.panel;
  const bd = kind === "same" ? C.teacher : kind === "diff" ? color : C.teacher;
  return (
    <div style={{ width: w, height: h, boxSizing: "border-box", background: bg, border: `3px solid ${bd}`, borderRadius: 12, padding: "10px 12px", opacity: p, translate: `0px ${(1 - p) * 18}px`, display: "flex", flexDirection: "column", justifyContent: "center", textAlign: "center", fontFamily: BODY }}>
      <div style={{ fontSize: 32, fontWeight: 700, color: C.ink, lineHeight: 1.1 }}>{title}</div>
      {lines.map((l) => <div key={l} style={{ fontSize: 28, lineHeight: 1.15, color: C.ink, marginTop: 4 }}>{l}</div>)}
    </div>
  );
};
const Arr: React.FC<{ at: number }> = ({ at }) => {
  const p = ramp(useCurrentFrame(), at, 10);
  return <div style={{ fontSize: 48, color: C.muted, opacity: p, alignSelf: "center", fontFamily: BODY }}>→</div>;
};

const Column: React.FC<{ model: "dp" | "fm"; x: number; t0: number }> = ({ model, x, t0 }) => {
  const dp = model === "dp";
  const col = dp ? C.dp : C.flow;
  return (
    <div style={{ position: "absolute", left: x, top: 150, width: 880 }}>
      <div style={{ fontFamily: DISPLAY, fontWeight: 800, fontSize: 72, color: col, letterSpacing: -0.5 }}>{dp ? "Diffusion" : "Flow matching"}</div>
      <div style={{ fontFamily: BODY, fontSize: 32, fontWeight: 600, color: C.muted, margin: "10px 0 8px" }}>Training: one gradient step</div>
      <div style={{ display: "flex", gap: 6 }}>
        <Box w={290} kind="diff" color={col} at={t0} title={dp ? "Add noise" : "Blend"} lines={dp ? ["t: integer 0 … 99", "x = √α·a + √(1−α)·ε"] : ["t ~ U(0, 1)", "x = (1−t)·ε + t·a"]} />
        <Arr at={t0 + 8} />
        <Box w={290} kind="same" color={col} at={t0 + 14} title="Same network" lines={dp ? ["time input t", "(0 … 99)"] : ["time input t × 100", "(float)"]} />
        <Arr at={t0 + 22} />
        <Box w={230} kind="diff" color={col} at={t0 + 28} title={dp ? "Predict noise ε" : "Predict velocity v"} lines={dp ? ["MSE to ε"] : ["MSE to a − ε"]} />
      </div>
      <div style={{ fontFamily: BODY, fontSize: 32, fontWeight: 600, color: C.muted, margin: "26px 0 8px" }}>Sampling: every 8 steps</div>
      <div style={{ display: "flex", gap: 6 }}>
        <Box w={190} h={190} kind="plain" color={col} at={t0 + 70} title="Noise" lines={["16 × 8"]} />
        <Arr at={t0 + 78} />
        <Box w={350} h={190} kind="diff" color={col} at={t0 + 84} title={dp ? "10 DDIM steps" : "10 Euler steps"} lines={dp ? ["t = 90, 80 … 0", "x0 estimate, clipped"] : ["t = 0, 0.1 … 0.9", "x ← x + v/10"]} />
        <Arr at={t0 + 92} />
        <Box w={190} h={190} kind="plain" color={col} at={t0 + 98} title="Chunk" lines={["run 8 of 16"]} />
      </div>
    </div>
  );
};

const ModelsInner: React.FC = () => {
  const f = useCurrentFrame();
  const p = ramp(f, 250, 20);
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <div style={{ position: "absolute", left: 80, top: 36, fontFamily: DISPLAY, fontWeight: 800, fontSize: 84, color: C.ink, letterSpacing: -1, whiteSpace: "nowrap" }}>Two models, two objectives</div>
      <Column model="dp" x={80} t0={10} />
      <Column model="fm" x={990} t0={60} />
      <div style={{ position: "absolute", left: 80, top: 800, width: 1760, opacity: p, fontFamily: BODY, fontSize: 32, lineHeight: 1.25, color: C.ink }}>
        <div><b style={{ color: C.teacher }}>Shared code:</b> ResNet18 + spatial softmax, 5 tokens, 4-layer decoder, Linear 256 → 8, data, optimiser, EMA.</div>
        <div><b style={{ color: C.fail }}>Not shared:</b> loss target, what the time input means, the sampler. Flow matching subclasses it and overrides only those.</div>
        <div style={{ color: C.muted }}>Grey boxes are the same code in both. Coloured boxes differ.</div>
      </div>
    </div>
  );
};
export const ModelsScene = Interactive.create({ Component: ModelsInner, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
