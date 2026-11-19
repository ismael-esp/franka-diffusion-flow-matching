import { Composition } from "remotion";
import { Results, RESULTS_FRAMES } from "./Results";
import { CONVERGE_FRAMES, DiffusionConvergeScene, DiffusionConvergeSingleScene, FlowIntegrateScene, FlowIntegrateSingleScene, FRAMES_SINGLE } from "./converge";
import { ModelsScene } from "./models";
import { POUR_FRAMES, TaskPourScene } from "./pourvideo";
import { TASK_FRAMES, TaskDrawerScene, TaskLiftScene, TaskStackScene } from "./taskvideo";

const bg = { backgroundColor: "#f5f7f9", alignItems: "center", justifyContent: "center" } as const;
const DiffusionConverge: React.FC = () => <DiffusionConvergeScene style={bg} />;
const FlowIntegrate: React.FC = () => <FlowIntegrateScene style={bg} />;
const DiffusionConvergeSingle: React.FC = () => <DiffusionConvergeSingleScene style={bg} />;
const FlowIntegrateSingle: React.FC = () => <FlowIntegrateSingleScene style={bg} />;
const TaskLift: React.FC = () => <TaskLiftScene style={bg} />;
const TaskDrawer: React.FC = () => <TaskDrawerScene style={bg} />;
const TaskStack: React.FC = () => <TaskStackScene style={bg} />;
const TaskPour: React.FC = () => <TaskPourScene style={bg} />;
const Models: React.FC = () => <ModelsScene style={bg} />;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition id="Results" component={Results} durationInFrames={RESULTS_FRAMES} fps={30} width={1920} height={1080} />
      <Composition id="DiffusionConverge" component={DiffusionConverge} durationInFrames={CONVERGE_FRAMES.diffusion} fps={30} width={1920} height={1080} />
      <Composition id="FlowIntegrate" component={FlowIntegrate} durationInFrames={CONVERGE_FRAMES.flow} fps={30} width={1920} height={1080} />
      <Composition id="DiffusionConvergeSingle" component={DiffusionConvergeSingle} durationInFrames={FRAMES_SINGLE.diffusion} fps={30} width={1920} height={1080} />
      <Composition id="FlowIntegrateSingle" component={FlowIntegrateSingle} durationInFrames={FRAMES_SINGLE.flow} fps={30} width={1920} height={1080} />
      <Composition id="TaskLift" component={TaskLift} durationInFrames={TASK_FRAMES.lift} fps={30} width={1920} height={1080} />
      <Composition id="TaskDrawer" component={TaskDrawer} durationInFrames={TASK_FRAMES.drawer} fps={30} width={1920} height={1080} />
      <Composition id="TaskStack" component={TaskStack} durationInFrames={TASK_FRAMES.stack} fps={30} width={1920} height={1080} />
      <Composition id="TaskPour" component={TaskPour} durationInFrames={POUR_FRAMES} fps={30} width={1920} height={1080} />
      <Composition id="Models" component={Models} durationInFrames={360} fps={30} width={1920} height={1080} />
    </>
  );
};
