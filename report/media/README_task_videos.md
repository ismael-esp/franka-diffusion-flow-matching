# Task convergence videos (report_v2, merged diffusion + flow matching)

One video per task. Each merges both models in the same video: the diffusion attempts first, then the flow-matching attempts. Every attempt plays the real rollout
(1 env step per frame, 2 per frame after the replay for the long tasks) over the task's video camera, pauses once at one policy call and replays the 10 DDIM / Euler
steps of 32 sampled chunks, continues to success and is cut 1 s after success. All drawn lines are the gripper paths the simulator produces for the sampled chunks
(state restored from the recording, `sim_chunks.py` / `sim_task.py`), projected into the camera; curves are smoothed for display (spline through the 16 real points).
Replay call rule (same for all tasks, models, attempts): among the calls that start at or before the first open-to-closed gripper command, the one whose predicted
16-step chunk end is closest to the marker (lift: cube, drawer: handle, stack: cube 2).

| file | task | length | what is shown |
|---|---|---|---|
| `task_lift_converge.mp4` | lift | 25.3 s | diffusion env 4, env 6, then flow env 4, env 6 (same two initial states, all succeed) |
| `task_drawer_converge.mp4` | drawer | 24.8 s | diffusion env 0, env 4, then flow env 0, env 4 (same two cabinet poses, all succeed) |
| `task_stack_converge.mp4` | stack | 24.4 s | diffusion (2 attempts), then flow (2 attempts); NOT the same initial states, see below |
| `task_pour_converge.mp4` | pour | 22.0 s | diffusion env 2, then flow env 2 (same initial cup/bottle state, both succeed); see the pour section below, different call rule and path than the other tasks |

## Checkpoints, seeds, envs
| task | diffusion | flow matching | recording | attempts |
|---|---|---|---|---|
| lift | `runs/flow/dp_lift/bc_best.pt` (62.8 % over seeds 1001-1003) | `runs/flow/fm_lift/bc_best.pt` (78.2 %) | `record_denoise.py`, seed 41, 8 envs (diffusion 5/8, flow 8/8 succeed) | envs 4 and 6 |
| drawer | `runs/drawer_s/dp/bc_best.pt` (92.2 %) | `runs/flow/fm_drawer/bc_best.pt` (92.3 %) | `record_task.py`, seed 41, 12 envs (diffusion 10/12, flow 12/12) | envs 0 and 4 (cabinet y +0.076 m and -0.095 m) |
| stack | `runs/stack/dp_dagger2/bc_best.pt`, DAgger round 2 (42.1 %, seeds 1002-1005) | `runs/flow/fm_stack/bc_best.pt`, demos only (7.3 %) | `record_task.py`, seeds 41 and 42, 16 envs each (32 envs): diffusion 5/16 and 8/16, flow 1/16 and 1/16, no env where both succeed | diffusion: batch 0 env 10, batch 1 env 12; flow: batch 0 env 8, batch 1 env 6 |

All models: 10 sampling steps, 16-step chunks, first 8 executed. The success rates are the held-out rates from RESULTS.md, not the rate of the shown episodes: every shown
attempt is a success chosen on purpose (searched for), so they are not typical for flow on stack. Stack: flow needed 32 envs to give two successes; the diffusion attempts were
picked to have a similar cube 2 start (flow: (0.568, 0.099) and (0.592, -0.066) m; diffusion: (0.511, 0.087) and (0.584, -0.059) m) but they are different states.

## Verification numbers (all from the data files in `data/task_*.json` and the logs)
* Franka FK vs the simulator's `panda_hand` (env frame): max 4e-7 m for drawer and stack (robot at the env origin, identity), 1e-6 m for lift. Cameras are the tasks' video cameras
  (drawer eye (-0.35, -1.25, 1.35) -> (0.55, 0, 0.5); stack and lift eye (1.45, 0.75, 0.75) -> (0.42, 0, 0.18); focal 18 mm, aperture 20.955 mm).
* Simulator re-simulation of the executed chunk vs the recorded fingertip path (mean over the 8 executed steps of the replayed call, pixels of a 640 x 480 frame): drawer 0.03-0.10 px,
  stack 0.02-0.08 px (max 1.4 px); lift see `RESULTS.md` (0.02-0.07 px). This checks the state restoration, not unexecuted chunks.
* Marker placement, fingertip at the first close command vs the marker (3D distance / pixels): drawer handle 3-7 mm / 0.5-2.2 px; stack cube 2 10-16 mm / 0.8-6.3 px.
* Predicted 16-step chunk end (executed sample) at the replay call, to the marker, and to the target:

| task | model | env | call (env step) | gripper to marker now | chunk end to marker | chunk end to target | spread of 32 end points |
|---|---|---|---|---|---|---|---|
| lift | diffusion | 4 | 2 (8) | 32 cm | 5.1 cm (14 px) | 30.4 cm goal | 19 mm |
| lift | diffusion | 6 | 3 (16) | 34 cm | 2.4 cm (9 px) | 26.7 cm goal | 28 mm |
| lift | flow | 4 | 2 (8) | 31 cm | 4.7 cm (11 px) | 31.4 cm goal | 14 mm |
| lift | flow | 6 | 3 (16) | 34 cm | 1.9 cm (4 px) | 27.0 cm goal | 25 mm |
| drawer | diffusion | 0 | 10 (72) | 1.5 cm | 0.3 cm (1 px) | 30.0 cm open target | 1.6 mm |
| drawer | diffusion | 4 | 11 (80) | 6.8 cm | 0.3 cm (0 px) | 29.8 cm | 2.9 mm |
| drawer | flow | 0 | 9 (64) | 4.7 cm | 0.7 cm (2 px) | 30.5 cm | 1.5 mm |
| drawer | flow | 4 | 12 (88) | 2.0 cm | 0.3 cm (1 px) | 30.2 cm | 1.4 mm |
| stack | diffusion | b0e10 | 4 (24) | 10.2 cm | 1.4 cm (5 px) | 17.6 cm on cube 1 | 8.7 mm |
| stack | diffusion | b1e12 | 4 (24) | 10.0 cm | 1.4 cm (5 px) | 13.6 cm | 6.1 mm |
| stack | flow | b0e8 | 5 (32) | 8.6 cm | 2.1 cm (6 px) | 15.2 cm | 2.1 mm |
| stack | flow | b1e6 | 4 (24) | 9.1 cm | 0.5 cm (2 px) | 13.2 cm | 2.9 mm |

Honest reading: the chunk is only 16 steps (lift 0.32 s, drawer 0.53 s, stack 0.80 s); by the rule the replayed chunk ends within 0.3-5 cm of the marker (the approach is short because
the replay is the closest approach before the grasp), and it never reaches the target, which is 13-30 cm away. For drawer and stack the arm is already near the marker at that call.
The simulator paths of the noisy early iterates stay close to the gripper because the arm low-passes the targets, so the visible convergence is smaller than a plot of the raw
joint targets would be.

## Pour (`task_pour_converge.mp4`), done: what is different
The earlier note that pour was blocked was wrong; it was written from reading code, nothing had been run. Checked by running: the pour env (`TRLDistill_v5`, two UR3 arms with Allegro hands, rigid cup and bottle) has **no water
particles** (pouring is judged from hand pose, bottle tilt and bottle mouth vs cup opening), so everything is restorable: both robots' joints and velocities, cup and bottle root states, and the env's real-match actuator lag filters
(slew, 4-step dead time, first-order filter; stored relative to the ring pointer). No Franka FK is used: the simulator gives the body and object poses directly.
* Models (same seed 201, same 4 initial states, 32 noise seeds per policy call, sample 0 = the executed one; the traced sampler loops reproduce `model.sample()` with max difference 0.0):
  diffusion `pouring_diffusion/runs/bc_lag1200/bc_best.pt`, 50 DDIM steps (51.5 %, `runs/sweep/lag1200_ddim50.json`); flow `pouring_diffusion/runs/fm_lag1200/bc_best.pt`, 5 Euler steps (57.0 %, `runs/sweep/fm_lag1200_k5.json`), 200 episodes, seed 201.
  Eval rules as `eval_policy.py` (stage `eval`, snapped grip, 16 of 32 predicted steps executed, 60 Hz control, so a chunk is 0.27 s).
* Search: 4 envs per model (GPU shared, 4 envs was what fit): diffusion succeeded in envs 0 and 2 (success steps 414, 431), flow in envs 2 and 3 (461, 433). Env 2 succeeds for both and has identical cup/bottle start poses (checked, difference 0.0), so the video has one attempt per model on the same state (cup (-0.081, 0.389), bottle (0.219, 0.463) m).
  Cut 59 env steps (about 1 s) after the env's success flag (cut rounded down to an even step for the 2-steps-per-frame playback; diffusion success step 431, cut 490; flow 461, cut 520). Success in the env is `subgoal >= 9`: all stages up to 8 passed, i.e. the pour stage, both objects placed down upright and released, both hands retreated a few cm (`P_BI`); returning to the start pose (stage 10) is not required. It is NOT a water measure.
* What is drawn: the **bottle mouth** (the env's pour point: bottle root + 0.216 m along the bottle axis; carried by the right hand) for each of the 32 sampled chunks, as the path the simulator produces when the 14-D chunk (both arms) is applied from the restored state.
  Markers: yellow square = cup opening (cup root + 0.157 m, the cup is held by the left hand); green shape = the env's pour region (half cylinder around the cup opening on the bottle side, radius 3.75 cm, from 2 cm below to 10 cm above the opening) that the mouth must enter in stage 4,
  with the hand within 4 cm of the pour pose and tilted within 15 deg. Projection: recording camera of the env (eye (0, 1.85, 1.65), target (0, 0.40, 1.12), focal 24 mm, horizontal aperture 33.24 mm, 848 x 480).
* Replay call rule (same for both models): the call that contains the start of the pour tilt, i.e. the first env step at which the bottle is tilted more than 45 deg from upright, call = step // 16. Both models: step 208, call 14 (1-based) of the executed episode.
  (The closest-approach-to-target rule of the other tasks does not fit: the hand sits at the pour pose for 80 steps, so the chunk ends there are all within millimetres.)
* Sampler states shown: diffusion has 50 DDIM steps, so the video shows 10 of 50 (the noise state, then every 5th: t = 98 -> 88 -> ... -> 0); flow shows all 5 Euler steps (t = 0, 0.2, ... 1.0; no velocity arrows or field in this video). The intermediate states are clamped to [-1, 1] and the grip entries snapped like the executed chunk.
* Replay: 2 env steps per frame before the call, 16 steps at 1 per frame after it, then 2 per frame (not the 1-per-frame / 2-per-frame-after-replay rule of the other tasks); sampled chunk paths of every call are drawn during play. Smoothed for display like the other videos. The line 'All models: 10 sampling steps, 16-step chunks, first 8 executed' above does NOT hold for pour: 50 DDIM (10 shown) / 5 Euler steps, 32-step chunks predicted, first 16 executed.

| model | env | replay call (step) | mouth to cup opening now | chunk end (16 steps) to opening | mouth move in 16 steps | spread of 32 end points: noise, final | success step | sim vs recorded run (executed chunk, all calls) |
|---|---|---|---|---|---|---|---|---|
| diffusion | 2 | 14 (208) | 24.5 cm | 16.1 cm (56 px) | 12.1 cm | 34 mm, 1.6 mm | 431 | mouth 0.48 px mean / 4.2 px max (1.7 mm), right hand 0.22 mm |
| flow | 2 | 14 (208) | 20.5 cm | 12.6 cm (46 px) | 11.0 cm | 36 mm, 2.3 mm | 461 | mouth 0.74 px mean / 6.1 px max (2.4 mm), right hand 0.22 mm |

* Verification: re-simulating the executed chunk (sample 0) of every call from the restored state reproduces the recorded right-hand path to 0.22 mm mean (hand) and the bottle mouth to 1.7-2.4 mm mean. Camera model: colour segmentation of the cup in the recorded frames vs the projected cup centre 0.9-1.1 px, bottle x 0.3 px (y has a 6 px offset because the segmented
  centroid is not at the assumed mid-height). Marker geometry: at the step the pour stage (stage 4) completes the right hand is 4 mm (diffusion) / 1.6 mm (flow) from the pour pose (1.7 / 0.6 px) and the mouth is inside the half cylinder (checked with the env's own test).
  Frames extracted from the encoded mp4 (pause start/middle/end, replay, success cut, last frame of each attempt) were viewed. No file in TRLDistill_v5 or pouring_diffusion was written (checked with find; scripts set `sys.dont_write_bytecode`).
* Caveats, stated honestly: (1) the sampled chunks are applied to the real simulator state, not to water; there is none. (2) The pour is a contact-rich in-hand grasp: re-simulating the same actions in another batch slot gives hand paths identical to 0.2 mm but the mouth can differ by a few mm (median 1.5 mm diffusion, 4.2 mm flow) and, rarely, the bottle slips in one sample (flow, policy calls 15 and 16, 1-based: 1 and 2 of 32 samples end more than 5 cm from sample 0). For the replayed call itself (flow, call 14) the two simulations of the final samples (one in the 'finals' batch, one in the replay batch) differed for one sample by 23 cm (bottle slipped in one of them); the video uses the 'finals' simulation for the last sampler state, so the replay ends exactly on the lines drawn afterwards, and the earlier states use the replay batch.
  (3) The end points of the 32 final samples agree within 2 mm because both models are essentially unimodal here; the visible convergence is mostly from the noisy early states (34-36 mm spread) to the final chunks. (4) Intermediate sampler states with noisy gripper entries snap to open/closed and may drop the bottle in the simulation (lines that fall in the early states), which is what the simulator does with that action.
  (5) Diffusion and flow are both single recordings (one 4-env batch per model); the success rates quoted are the held-out numbers, not the shown episodes.
* Files: `pour_common.py` (build, state get/set, projection), `pour_record.py`, `pour_sim.py`, `pour_select.py`, `make_pour_data.py`, `video/src/pourvideo.tsx` + `Root.tsx` composition `TaskPour`, data `data/task_pour.json`; recordings `runs/pour/dp_t1`, `.../fm_t1` (per-env mp4, `iterates.npz`, `episode.npz`, `meta.json`, logs), simulator paths `runs/sim_pour.npz`.
  Reproduce: `python pour_record.py --kind diffusion --checkpoint .../bc_lag1200/bc_best.pt --ddim-steps 50 --out-dir runs/pour/dp_t1 --seed 201 --num-envs 4` (flow: `--kind flow ... fm_lag1200 --ddim-steps 5`), `python pour_sim.py --out runs/sim_pour.npz --item diffusion:runs/pour/dp_t1:2 --item flow:runs/pour/fm_t1:2 --batch 32`, `python make_pour_data.py --sim runs/sim_pour.npz --rate-pre 2 --item diffusion:runs/pour/dp_t1:2 --item flow:runs/pour/fm_t1:2`, `npx remotion render TaskPour media/task_pour_converge.mp4`.

## Files
* Code: `record_task.py`, `sim_task.py`, `task_select.py`, `make_task_data.py`, `make_lift_task_data.py`, `geom.py` (`project_cam`), `task_geom_check.py`, `video/src/taskvideo.tsx`, `video/src/Root.tsx` (compositions `TaskLift`, `TaskDrawer`, `TaskStack`).
* Data: `data/task_{lift,drawer,stack}.json`; raw recordings in `runs/task_{drawer,stack}`, simulator paths `runs/sim_{drawer,stack}.npz`.

Note: the finished clips are published with the project website (`docs/assets/video`); this repository's `report/media` keeps only the posters and a few clips. The recordings, `runs/` and the simulator-path files are generated by the scripts above and are not published.
