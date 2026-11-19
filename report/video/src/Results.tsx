import React from "react";
import { Track } from "remotion";
import { C } from "./ui";
import { CONVERGE_FRAMES, DiffusionConvergeScene, FlowIntegrateScene } from "./converge";
import { HowScene } from "./architecture";
import { ModelsScene } from "./models";
import { CeilingScene, FlowScene, QuestionScene, StackScene, StepsScene, TakeawayScene } from "./scenes";

const bg = { backgroundColor: C.paper, alignItems: "center", justifyContent: "center" } as const;
// seconds per scene: question, ceiling, stacking, flow vs diffusion, two models, how they learn, diffusion converge, flow integrate, few steps, takeaway
const SEC = [6, 8, 8, 9, 10, 6, 0, 0, 8, 8];
const FR = SEC.map((s, i) => (i === 6 ? CONVERGE_FRAMES.diffusion : i === 7 ? CONVERGE_FRAMES.flow : s * 30));
export const RESULTS_FRAMES = FR.reduce((a, b) => a + b, 0);

export const Results: React.FC = () => {
  const at = (i: number) => FR.slice(0, i).reduce((a, b) => a + b, 0);
  const d = (i: number) => FR[i];
  return (
    <Track name="Scenes">
      <QuestionScene name="Question" durationInFrames={d(0)} style={bg} />
      <CeilingScene name="Teacher ceiling" from={at(1)} durationInFrames={d(1)} style={bg} />
      <StackScene name="Stacking" from={at(2)} durationInFrames={d(2)} style={bg} />
      <FlowScene name="Flow vs diffusion" from={at(3)} durationInFrames={d(3)} style={bg} />
      <ModelsScene name="Two models, two objectives" from={at(4)} durationInFrames={d(4)} style={bg} />
      <HowScene name="How they learn (corrupted chunks, sampling paths)" from={at(5)} durationInFrames={d(5)} trimBefore={190} style={bg} />
      <DiffusionConvergeScene name="Diffusion converges" from={at(6)} durationInFrames={d(6)} style={bg} />
      <FlowIntegrateScene name="Flow integrates" from={at(7)} durationInFrames={d(7)} style={bg} />
      <StepsScene name="Few steps" from={at(8)} durationInFrames={d(8)} style={bg} />
      <TakeawayScene name="Takeaway" from={at(9)} durationInFrames={d(9)} style={bg} />
    </Track>
  );
};
