import React from "react";
import { Easing, interpolate, useCurrentFrame } from "remotion";
import { Video } from "@remotion/media";
import { staticFile } from "remotion";
import { loadFont as loadArchivo } from "@remotion/google-fonts/Archivo";
import { loadFont as loadPlex } from "@remotion/google-fonts/IBMPlexSans";

const archivo = loadArchivo("normal", { weights: ["700", "800"] });
const plex = loadPlex("normal", { weights: ["400", "500", "600"] });
export const DISPLAY = archivo.fontFamily;
export const BODY = plex.fontFamily;

// Same tokens as report/shared.css
export const C = {
  paper: "#f5f7f9", panel: "#ffffff", ink: "#12202b", muted: "#52626f", line: "#d3dbe2",
  teacher: "#3d5568", dp: "#df6327", flow: "#149484", fail: "#b8473f", ok: "#2f8a57",
};
export const EASE = Easing.bezier(0.16, 1, 0.3, 1);

// One entrance per scene: progress 0 -> 1 starting at `delay` frames.
export const useIn = (delay = 0, dur = 24) => {
  const f = useCurrentFrame();
  return interpolate(f, [delay, delay + dur], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: EASE });
};

export const Headline: React.FC<{ children: React.ReactNode; size?: number }> = ({ children, size = 112 }) => (
  <div style={{ fontFamily: DISPLAY, fontWeight: 800, fontSize: size, lineHeight: 1.02, letterSpacing: -1.5, color: C.ink, textWrap: "balance" }}>
    {children}
  </div>
);

export const Clip: React.FC<{ src: string; w: number; label: string; sub?: string; color?: string; ratio?: number }> = ({ src, w, label, sub, color = C.ink, ratio = 488 / 640 }) => (
  <div style={{ width: w, background: C.panel, border: `2px solid ${C.line}`, borderRadius: 14, overflow: "hidden" }}>
    <Video src={staticFile(src)} muted loop style={{ width: w, height: Math.round(w * ratio), display: "block", objectFit: "cover" }} />
    <div style={{ padding: "12px 18px 14px", fontFamily: BODY }}>
      <div style={{ fontSize: 40, fontWeight: 600, color }}>{label}</div>
      {sub ? <div style={{ fontSize: 34, color: C.muted }}>{sub}</div> : null}
    </div>
  </div>
);

// Horizontal bar that grows from zero while its number counts up.
export const Bar: React.FC<{ label: string; value: number; color: string; maxW: number; delay: number; h?: number; fs?: number }> = ({ label, value, color, maxW, delay, h = 72, fs = 46 }) => {
  const p = useIn(delay, 36);
  return (
    <div style={{ fontFamily: BODY }}>
      <div style={{ fontSize: fs - 6, color: C.muted, fontWeight: 500, marginBottom: 6 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
        <div style={{ width: (maxW * value * p) / 100, height: h, background: color, borderRadius: 6, minWidth: 4 }} />
        <div style={{ fontSize: fs + 14, fontWeight: 600, color: C.ink, fontVariantNumeric: "tabular-nums" }}>{(value * p).toFixed(1)}%</div>
      </div>
    </div>
  );
};
