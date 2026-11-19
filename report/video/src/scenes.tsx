import React from "react";
import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { BODY, C, Bar, Clip, DISPLAY, Headline, useIn } from "./ui";

const PAD = { padding: "100px 120px" } as const;
const Sub: React.FC<{ children: React.ReactNode; color?: string }> = ({ children, color = C.muted }) => (
  <div style={{ fontFamily: BODY, fontSize: 52, lineHeight: 1.3, color, maxWidth: 1500 }}>{children}</div>
);

// ---- 1 Question -------------------------------------------------------------------------------------------------
const QuestionInner: React.FC = () => {
  const tiles = [
    { src: "fm_pour_2x2.mp4", label: "Pour", ratio: 480 / 848 },
    { src: "fm_lift_2x2.mp4", label: "Lift", ratio: 480 / 640 },
    { src: "fm_drawer_2x2.mp4", label: "Drawer", ratio: 488 / 640 },
    { src: "fm_stack_2x2.mp4", label: "Stack", ratio: 488 / 640 },
  ];
  const f = useCurrentFrame();
  return (
    <div style={{ ...PAD, display: "flex", flexDirection: "column", gap: 48, width: "100%", boxSizing: "border-box" }}>
      <div style={{ opacity: useIn(0, 20), translate: `0px ${(1 - useIn(0, 20)) * 30}px` }}>
        <Headline size={132}>Can a diffusion policy learn manipulation from a teacher?</Headline>
      </div>
      <div style={{ display: "flex", gap: 32, marginTop: 24 }}>
        {tiles.map((t, i) => {
          const p = interpolate(f, [30 + i * 8, 30 + i * 8 + 24], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          return (
            <div key={t.label} style={{ opacity: p, translate: `0px ${(1 - p) * 40}px` }}>
              <Clip src={t.src} w={402} label={t.label} ratio={t.ratio} />
            </div>
          );
        })}
      </div>
      <Sub>Four tasks, one recipe: teacher, demonstrations, student.</Sub>
    </div>
  );
};
export const QuestionScene = Interactive.create({ Component: QuestionInner, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });

// ---- 2 Teacher ceiling ------------------------------------------------------------------------------------------
const CeilingInner: React.FC = () => (
  <div style={{ ...PAD, display: "grid", gridTemplateColumns: "1fr 700px", gap: 80, width: "100%", boxSizing: "border-box", alignContent: "center" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: 44 }}>
      <Headline size={120}>The teacher sets the ceiling</Headline>
      <Sub>Drawer student, same network, different teacher.</Sub>
      <div style={{ display: "flex", flexDirection: "column", gap: 34, marginTop: 8 }}>
        <Bar label="Student of the jerky PPO teacher" value={8.7} color={C.fail} maxW={780} delay={20} />
        <Bar label="Student of the smooth scripted teacher" value={92.2} color={C.dp} maxW={780} delay={50} />
      </div>
    </div>
    <div style={{ alignSelf: "center", opacity: useIn(10, 24) }}>
      <Clip src="drawer_dp_2x2.mp4" w={700} label="Drawer student" sub="scripted teacher, 92.2%" color={C.dp} />
    </div>
  </div>
);
export const CeilingScene = Interactive.create({ Component: CeilingInner, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });

// ---- 3 Stacking -------------------------------------------------------------------------------------------------
const StackInner: React.FC = () => (
  <div style={{ ...PAD, display: "grid", gridTemplateColumns: "860px 1fr", gap: 60, width: "100%", boxSizing: "border-box", alignContent: "center" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: 34 }}>
      <Headline size={104}>Stacking needed corrections</Headline>
      <Sub>The student drives, the teacher relabels (DAgger).</Sub>
      <div style={{ display: "flex", flexDirection: "column", gap: 26, marginTop: 4 }}>
        <Bar label="Demonstrations only" value={10.9} color={C.dp} maxW={620} delay={18} h={56} fs={42} />
        <Bar label="DAgger, round 2" value={42.1} color={C.dp} maxW={620} delay={44} h={56} fs={42} />
        <Bar label="Scripted teacher" value={94.0} color={C.teacher} maxW={620} delay={70} h={56} fs={42} />
      </div>
      <div style={{ fontFamily: BODY, fontSize: 42, color: C.muted }}>From scratch on all DAgger data: 2%.</div>
    </div>
    <div style={{ display: "flex", gap: 24, alignSelf: "center", opacity: useIn(10, 24) }}>
      <Clip src="stack_dp_2x2.mp4" w={380} label="Demos only" sub="10.9%" color={C.fail} />
      <Clip src="stack_dagger_2x2.mp4" w={380} label="DAgger round 2" sub="42.1%" color={C.ok} />
    </div>
  </div>
);
export const StackScene = Interactive.create({ Component: StackInner, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });

// ---- 4 Flow vs diffusion ----------------------------------------------------------------------------------------
const GROUPS = [
  { task: "Stack", dp: 10.2, fm: 7.3 },
  { task: "Drawer", dp: 92.2, fm: 92.3 },
  { task: "Lift", dp: 62.8, fm: 78.2 },
  { task: "Pour", dp: 27.5, fm: 57.0 },
];
const GroupBars: React.FC<{ g: { task: string; dp: number; fm: number }; i: number; H: number }> = ({ g, i, H }) => {
  const a = useIn(16 + i * 8, 34);
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, fontFamily: BODY }}>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 14, height: H }}>
        {[{ v: g.dp, c: C.dp }, { v: g.fm, c: C.flow }].map((b) => (
          <div key={b.c} style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end", height: H }}>
            <div style={{ fontSize: 38, fontWeight: 600, color: C.ink, fontVariantNumeric: "tabular-nums", marginBottom: 6 }}>{(b.v * a).toFixed(1)}</div>
            <div style={{ width: 78, height: (H - 50) * (b.v / 100) * a, background: b.c, borderRadius: "6px 6px 0 0" }} />
          </div>
        ))}
      </div>
      <div style={{ fontSize: 46, fontWeight: 600, color: C.ink }}>{g.task}</div>
    </div>
  );
};
const FlowInner: React.FC = () => {
  const H = 470;
  return (
    <div style={{ ...PAD, display: "grid", gridTemplateColumns: "1fr 480px", gap: 60, width: "100%", boxSizing: "border-box", alignContent: "center" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 30 }}>
        <Headline size={96}>Flow matching helps lift and pouring</Headline>
        <div style={{ display: "flex", gap: 36, fontFamily: BODY, fontSize: 38, color: C.muted }}>
          <span><b style={{ color: C.dp }}>Diffusion</b></span>
          <span><b style={{ color: C.flow }}>Flow matching</b></span>
          <span>Pour: both at 5 sampling steps</span>
        </div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 44, height: H + 70 }}>
          {GROUPS.map((g, i) => (
            <GroupBars key={g.task} g={g} i={i} H={H} />
          ))}
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 28, alignSelf: "center", opacity: useIn(8, 24) }}>
        <Clip src="fm_lift_2x2.mp4" w={480} label="Lift, flow matching" sub="78.2% at 10 steps" color={C.flow} ratio={480 / 640} />
        <Clip src="fm_pour_2x2.mp4" w={480} label="Pour, flow matching" sub="57.0%" color={C.flow} ratio={480 / 848} />
      </div>
    </div>
  );
};
export const FlowScene = Interactive.create({ Component: FlowInner, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });

// ---- 5 Few steps ------------------------------------------------------------------------------------------------
const StepsInner: React.FC = () => (
  <div style={{ ...PAD, width: "100%", boxSizing: "border-box", display: "flex", flexDirection: "column", gap: 44, justifyContent: "center" }}>
    <Headline size={120}>Flow matching works with few steps</Headline>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 90 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 30 }}>
        <div style={{ fontFamily: DISPLAY, fontWeight: 700, fontSize: 64, color: C.ink }}>Lift, flow matching, one sampling step</div>
        <Bar label="Flow matching" value={66} color={C.flow} maxW={700} delay={40} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 30 }}>
        <div style={{ fontFamily: DISPLAY, fontWeight: 700, fontSize: 64, color: C.ink }}>Pour, flow matching</div>
        <Bar label="5 sampling steps" value={57} color={C.flow} maxW={700} delay={64} />
        <Bar label="50 sampling steps" value={41.5} color={C.teacher} maxW={700} delay={88} />
      </div>
    </div>
    <div style={{ fontFamily: BODY, fontSize: 40, color: C.muted }}>Lift: one test seed, 200 episodes. Pour: one run of 200 episodes. DDIM with one step is not comparable (it only evaluates t=0 on pure noise).</div>
  </div>
);
export const StepsScene = Interactive.create({ Component: StepsInner, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });

// ---- 6 Takeaway -------------------------------------------------------------------------------------------------
const TakeawayInner: React.FC = () => {
  const a = useIn(0, 24);
  const b = useIn(40, 24);
  return (
    <div style={{ ...PAD, width: "100%", boxSizing: "border-box", display: "flex", flexDirection: "column", gap: 54, justifyContent: "center" }}>
      <div style={{ opacity: a, translate: `0px ${(1 - a) * 30}px` }}>
        <Headline size={116}>The teacher and its labels set the limit, more than the training method</Headline>
      </div>
      <div style={{ opacity: b, translate: `0px ${(1 - b) * 30}px`, fontFamily: BODY, fontSize: 42, lineHeight: 1.35, color: C.muted, display: "flex", flexDirection: "column", gap: 14, maxWidth: 1560 }}>
        <div>Held-out test seeds, 200 episodes each. The pour result is one run.</div>
        <div>Stacking is still far from the 94% teacher.</div>
        <div>Why DAgger from scratch failed is a hypothesis, not tested.</div>
      </div>
    </div>
  );
};
export const TakeawayScene = Interactive.create({ Component: TakeawayInner, layout: "absolute-fill", schema: { ...Interactive.backgroundSchema } });
