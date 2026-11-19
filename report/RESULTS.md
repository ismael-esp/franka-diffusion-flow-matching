# Results summary: flow matching, DAgger and what limits each task

Success rates on held-out episodes. SE = sqrt(p(1-p)/n). "Diffusion" = DDIM sampling; "flow" = flow matching (rectified flow,
Euler sampling) with the same encoder, transformer, data and recipe; only the training target (velocity instead of noise) and the
sampler differ. The number of sampling steps is the same for both objectives unless stated.
Standard errors are about 1-2 points for the 600-800 episode rows and about 3.5 points for the pouring rows (200 episodes).

## 1. Flow matching vs diffusion on the four tasks

| Task | Diffusion | Flow matching | Verdict |
|---|---|---|---|
| Franka stacking (10 steps, seeds 1001-1003, 3 x 200) | 10.2 % (8.0 / 9.0 / 13.5) | 7.3 % (6.0 / 10.0 / 6.0) | both low; within about 2 SE |
| Franka stacking, like-for-like re-run (10 steps, seeds 1002-1005, 4 x 200, idle GPU) | 10.9 % +- 1.1 | 9.3 % +- 1.0 | no meaningful difference |
| Franka drawer (10 steps, seeds 1001-1003, 3 x 200) | 92.2 % +- 1.1 | 92.3 % (95.0 / 90.5 / 91.5) | tie |
| Franka lift (10 steps, seeds 1001-1003, 3 x 200) | 62.8 % (63.0 / 66.0 / 59.5) | 78.2 % (77.0 / 78.0 / 79.5) | flow +15 points |
| Pouring, 5 steps (seed 201, 200 episodes) | 27.5 % | 57.0 % | flow +29.5 points |
| Pouring, 10 steps | not measured | 50.0 % | |
| Pouring, 50 steps | 51.5 % | 41.5 % | diffusion ahead at 50 steps |

Training: stacking, drawer and lift 30k steps from scratch each (lift: both objectives from scratch on the same zoom demos, so
these lift numbers are lower than the earlier fine-tune-chain lift policies); pouring flow 80k steps on `demos_lag400` +
`demos_lag800b`, the recipe of the diffusion `bc_lag1200`.

### Sampling steps (one seed each)

| Task / seed | Model | 1 step | 5 steps | 10 steps | 50 steps |
|---|---|---|---|---|---|
| Lift, 1001 | diffusion | not comparable (see below) | 60.5 % | 63.0 % | |
| Lift, 1001 | flow | 66.0 % | 81.0 % | 77.0 % | |
| Drawer, 1001 | flow | 75.5 % | 93.0 % | 95.0 % | |
| Stacking, 1001 | flow | 0.0 % | 10.0 % | 6.0 % | |
| Pouring, 201 | diffusion | | 27.5 % | | 51.5 % |
| Pouring, 201 | flow | | 57.0 % | 50.0 % | 41.5 % |

Flow matching keeps working with one Euler step on lift (66.0 %) and drawer (75.5 %); its pouring success is best at 5 steps and falls as steps are
added (no explanation tested).

**The diffusion 1-step cell is not comparable.** `ddim_timesteps(100, 1)` (`pouring_diffusion/dp_core/model.py`) returns `[0]`, so the single network call is made
at t = 0 on pure noise (alpha about 0.9994) and the output is essentially the initial noise; the measured 0 % is a property of that sampling grid, not a fair measure of
what a one-step diffusion sampler could do. A trailing-spaced grid starting at t = 99 was not tested. Likewise the 5-step DDIM grid starts at t = 80 and never sees the
noisiest levels (80, 60, 40, 20, 0); this is an untested possible contributor to the low diffusion pouring result at 5 steps (27.5 %).

## 2. Drawer (from the earlier page)

| Policy | Success |
|---|---|
| PPO teacher | 97.0 % +- 1.2 |
| Diffusion policy from the PPO teacher | 8.7 % +- 1.2 (600 episodes) |
| Scripted IK teacher (GPU eval) | 90.5 % +- 2.1 |
| Diffusion policy from the scripted teacher | 92.2 % +- 1.1 (600 episodes) |
| Flow matching from the scripted teacher | 92.3 % +- 1.1 (600 episodes) |

The PPO teacher's actions are ~25x rougher (mean step-to-step change 0.061 vs 0.0024), which the student cannot copy. With the
scripted teacher the student matches the teacher; the teacher is the ceiling. DAgger was **not** tried on the drawer: the scripted
drawer teacher has no resync mode and has not been checked for sensible labels on student-driven states.

## 3. Stacking and DAgger

DAgger: the student drives, the scripted teacher (resync mode: its phase is re-derived from the scene) labels every visited
state, then the student is fine-tuned on all data so far (`--init` previous model, 15k steps per round).

Held-out seeds 1002-1005 (4 x 200 = 800 episodes, idle GPU, 50 parallel envs). Seed 1001 was used only to pick models.

| Model | 1002 | 1003 | 1004 | 1005 | Mean +- SE |
|---|---|---|---|---|---|
| Scripted teacher (100 episodes) | | | | | 94.0 % +- 2.4 |
| Diffusion, demos only | 9.5 | 14.0 | 10.5 | 9.5 | 10.9 % +- 1.1 |
| Flow, demos only | 10.5 | 5.5 | 9.5 | 11.5 | 9.3 % +- 1.0 |
| **DAgger round 2 (diffusion, fine-tune chain)** | 39.0 | 43.5 | 44.0 | 42.0 | **42.1 % +- 1.7** |
| DAgger round 3 (diffusion, fine-tune chain) | 23.0 | 24.0 | 18.5 | 27.0 | 23.1 % +- 1.5 |
| Diffusion, from scratch on demos + rounds 1-3 | 2.5 | 2.0 | 2.0 | 1.5 | 2.0 % +- 0.5 |
| Flow, from scratch on demos + rounds 1-3 | 0.0 | 5.5 | 1.5 | 4.0 | 2.8 % +- 0.6 |
| Diffusion, from scratch + round 4 | 0.5 | 3.0 | 2.5 | 2.5 | 2.1 % +- 0.5 |
| Flow, from scratch + round 4 | 2.5 | 0.5 | 0.5 | 4.5 | 2.0 % +- 0.5 |

Selection seed 1001: round 1 4.5 %, round 2 39.0 %, round 3 29.5 %; demos-only diffusion 8.0 %.

Outcomes over the 800 episodes (success / none stacked / one stacked): demos diffusion 87 / 491 / 222; demos flow 74 / 540 / 186;
DAgger round 2 337 / 169 / 294; round 3 185 / 174 / 441; scratch diffusion 16 / 432 / 352; scratch flow 22 / 347 / 431.

Best validation action MSE (each run on its own validation split): demos only 0.0060 (flow 0.0040); +round 1 0.0109;
+rounds 1-2 0.0276; +rounds 1-3 0.0472 (chain); from scratch on the same rounds 1-3 data: flow 0.0404, diffusion 0.0560.
Round 4 data: 0.0287 (flow) / 0.0316 (diffusion) on a different mix.

Reading: DAgger round 2 is 4x the demos-only student, but still far below the 94 % teacher. Training from scratch on all the DAgger
data failed for both objectives (about 2 %), and the validation loss rises as DAgger data is added. **Hypothesis, not tested:** the
teacher labels on student-visited states are harder to fit (noisier or more multimodal) than demo states; 30k steps may also simply
be too few for the mixed data. The fine-tune chain probably works in round 2 because it keeps the demo behaviour and corrects only
part of it, but round 3 then degrades. Flow matching did not help stacking; this is consistent with the result being limited by the data and labels rather than the objective (not tested).

## 4. Caveats

- Round 2 was **chosen on seed 1001** (39.0 %); the reported 42.1 % uses seeds 1002-1005 only.
- A first test of round 2 on seed 1003 gave 6 %. It ran beside flow-matching training on an almost full GPU (the other tests that
  ran at the same time hit out-of-memory errors). It was replaced by an idle-GPU rerun: 42.7 % (192 episodes, 16 envs) and then 43.5 %
  (200 episodes, 50 envs). The bad file is kept as `runs/stack/test_dagger2_s1003_BAD_gpu_contention.json`. I cannot say exactly how
  GPU contention produced the low number; the policy itself is fine.
- **Correction:** an earlier draft reported the flow-matching stacking number as 6.3 %; that was an averaging error; the three seeds
  (12/200, 20/200, 12/200) pool to 44/600 = **7.3 %**.
- Diffusion demos-only stacking is 10.2 % on seeds 1001-1003 (3 evals from the first run) and 10.9 % on seeds 1002-1005 (re-run on an
  idle GPU); per-seed values differ by up to 0.5 points between the two runs (different env counts), so numbers are reproducible only
  to about that level.
- Flow matching vs diffusion rows come from different training runs (one seed each); differences of a few points on stacking and
  drawer are not meaningful. The lift (+15) and pouring-at-5-steps (+29.5) gaps are far outside the standard errors but are single
  training runs.
- The diffusion pouring 5-step number (27.5 %) uses the checkpoint's own 5-step DDIM schedule (the evaluation file does not record it).
- DAgger was run with the diffusion student only; flow matching was tested on demos and on the from-scratch DAgger mix, not as a
  DAgger chain.
- The drawer was not improved with DAgger (students 92 % vs 90.5 % teacher); the earlier attempts to fix the scripted drawer teacher
  (posture term, homing, translate-first) gave no real gain and are left off.

## 5. Files

- Pages: `report/index.html`, `report/page2.html`, `report/page3.html`.
- Result JSONs: `runs/flow/`, `runs/improve/`, `runs/stack/`; scripts `run_flow.sh`, `run_improve.sh`, `run_stack_dagger.sh`.
- Flow matching code: `pouring_diffusion/dp_core/flow.py`, `train_fm.py`, `eval_fm.py`; in this repo `train_dp.py --objective flow`.

## Slide media (clips on the project page)
- Flow-matching clips (5 sampling steps, 4 of the recorded episodes each): lift seed 33 (3/4 success shown; 4-env recording), pour seed 301 (6 envs, 3 successes; envs 0, 1, 3, 5 shown), drawer seed 32 (7/8 recorded, plain white cabinet), stack seed 34 (1/8 recorded; demos-only flow, 7.3 % overall). Rates on the slide are the held-out test rates, not these samples.
- Feature visualisations: same spatial-softmax attention method as the first page (`attention_task.py` for drawer/stack, `record_attention.py` for lift, `make_attention_stills.py` for the stills). Lift diffusion vs flow stills come from different random scene draws (not a paired test). No saliency/occlusion maps were re-run for the new policies.

## Page 4: how the two policies are built and how they learn

Every size below is read from code or checkpoints, nothing is estimated.

| Item | Value | Source |
|---|---|---|
| Image per step | workspace and wrist cameras, 128 x 128 each, side by side (128 x 256) | `fd_core/tasks.py` (cat on width), `collect_task.py` meta (image_size 128, wrist true) |
| Image encoder | ResNet18 with GroupNorm, spatial softmax with 32 keypoints (64 numbers), Linear to 256; 4 x 8 feature grid (stride 32) | `dp_core/model.py` ImageEncoder |
| Joint state | 18 numbers (drawer), 21 (lift); normalised, clipped to +-5, Linear to 256 | checkpoint configs, `DiffusionPolicy.encode` |
| Tokens | 2 past steps x (image, joints) = 4 observation tokens, plus 1 timestep token = 5 memory tokens | `encode`, `TransformerDenoiser.forward` |
| Timestep | diffusion: integer 0 to 99; flow: t x 100 (continuous); sinusoidal 256, MLP 256 to 1024 to 256 (Mish) | `dp_core/model.py`, `dp_core/flow.py` |
| Transformer | encoder 2 layers, decoder 4 layers, 4 heads, width 256, feed-forward 1024, pre-norm, GELU, dropout 0; decoder self-attends over 16 action tokens and cross-attends to the memory | `TransformerDenoiser` |
| Chunk | predict 16 steps x 8 values, execute the first 8; 17.5 M parameters | checkpoint configs (pred_horizon 16, exec_horizon 8, agent_action_dim 8) |
| Diffusion | 100 training timesteps, cosine schedule, noise prediction, MSE; DDIM eta 0, 10 sampling steps, x0 clipped | `bc_loss`, `ddim_mean`, `sample` |
| Flow matching | x_t = (1 - t) noise + t action, t ~ U(0,1), velocity target action - noise, MSE; Euler with k steps, clamp to +-1 | `dp_core/flow.py` |
| Training | AdamW lr 1e-4, 500 warm-up steps then cosine decay, weight decay 1e-4, batch 128, grad clip 1.0, bf16, EMA 0.999, 30,000 steps (pouring: 80,000) | `train_dp.py` defaults, checkpoint args |
| Pouring variant | same code, two arms (7 + 7 action values), 32 steps predicted, 16 executed, 5 sampling steps | `pouring_diffusion/runs/{bc,fm}_lag1200` configs |

Real vs schematic: the five small plots per method use a real demo chunk (`runs/drawer_s/demos_clean`, episode 0, steps 75 to 90, the gripper
command, +1 open to -1 closed; saved in `data/arch_chunk.json`) corrupted with the real cosine schedule (alpha = 1.00, 0.84, 0.48, 0.13, 0.00 at
t = 0, 25, 50, 75, 99) or the straight flow blend with one seeded noise draw (seed 20). The sampling-path plots are schematic: one action
coordinate and a perfect network, drawn on the real DDIM grid (90, 80, ..., 0) and the real schedule for diffusion and on the straight line for flow.
Held-out action error (mean squared error of the sampled first 8 actions against held-out lift demos, best checkpoint): diffusion 0.0140, flow matching 0.0079.


## Page 4 update and the convergence overlay (what is real)

- Code differences, verified (dp_core is in pouring_diffusion/): shared = ImageEncoder (model.py:70-90), observation tokens `encode` (157-164), TransformerDenoiser incl. the Linear head (93-125),
  the training loop, optimiser, EMA and checkpoint format (train_dp.py). `FlowPolicy(DiffusionPolicy)` (flow.py:16) overrides only `bc_loss` and `sample`.
  Diffusion: integer t in 0..99, noise eps, x_t = sqrt(a_t) a + sqrt(1 - a_t) eps with the cosine alphas (model.py:48-55, 166-171), target eps; DDIM eta = 0 over t = 90, 80 .. 0 with x0 clipped to +-1 at every step (179-198).
  Flow: t ~ U(0,1), x_t = (1 - t) eps + t a, time input t * 100 (float), target a - eps (flow.py:19-24); Euler x += v / k at t = i / k, clamp to +-1 only at the end (26-35). t = 99 is noise for diffusion, t = 0 is noise for flow.
  A flow checkpoint still carries the unused alphas table. No head or normalisation difference.

Earlier overlay versions (joint-curve panels and FK of the commanded targets, with their intermediate recording batches) were superseded by the convergence
videos below and their numbers are removed here; only the final version is documented.

## Convergence videos, final version

Files: `media/task_{lift,drawer,stack}_converge.mp4` (one per task, diffusion attempts then flow-matching attempts, about 25 s), lift variants `media/diffusion_converge.mp4` + `flow_integrate.mp4`
(several replays per attempt, two attempts) and `media/*_single.mp4` (one replay per attempt), `video/out/results.mp4` (98.6 s). Details per file: `media/README_task_videos.md`. Pour: `media/task_pour_converge.mp4` (see the pour paragraph below).

What the overlays are
- Every drawn line is a gripper path **from the simulator**: the recorded robot (joints and velocities) and task objects are restored (`sim_chunks.py` for lift, `sim_task.py` for drawer / stack) and a sampled 16-step chunk is applied for 16 steps; the line is forward kinematics (`geom.py`) of the joints the env reaches, projected into the task's video camera. 32 noise seeds per policy call; the executed sample is the thick line.
- The curves are **smoothed for display** (Catmull-Rom through the real 16 points); the data are not smoothed. The video says so.
- Each replay shows all 10 DDIM / Euler steps of one policy call (the rollout pauses). Rollout speed: lift 1 env step per frame (0.6 x real time); drawer / stack 1 step per frame for the first 8 steps after the replay, then 2 per frame.
- Episodes are cut 1 s after the first step at which the task success holds.
- Because the arm low-passes the targets (PD controller, gravity), the simulator paths of noisy early iterates stay within a few cm of the gripper: the visible convergence is smaller than a plot of raw joint targets would show.

Horizon caveat: a chunk is 16 steps = 0.32 s (lift, 50 Hz), 0.53 s (drawer, 30 Hz), 0.80 s (stack, 20 Hz). It ends **part-way** to the marker, never at the target (13-30 cm away). The replay call is chosen by one rule for all tasks / models / attempts:
among the calls up to the first open-to-closed gripper command, the one whose predicted 16-step end is closest to the marker (lift cube, drawer handle, stack cube 2). Chunk end to marker at the replay: lift 1.9-5.1 cm, drawer 0.3-0.7 cm, stack 0.5-2.1 cm (table in `media/README_task_videos.md`).

Checkpoints, seeds, envs
| task | diffusion | flow matching | recording | shown |
|---|---|---|---|---|
| lift | `runs/flow/dp_lift/bc_best.pt` (62.8 %) | `runs/flow/fm_lift/bc_best.pt` (78.2 %) | `record_denoise.py` seed 41, 8 envs (5/8, 8/8) | envs 4, 6 both models, all succeed |
| drawer | `runs/drawer_s/dp/bc_best.pt` (92.2 %) | `runs/flow/fm_drawer/bc_best.pt` (92.3 %) | `record_task.py` seed 41, 12 envs (10/12, 12/12) | envs 0, 4 both models, all succeed |
| stack | `runs/stack/dp_dagger2/bc_best.pt` DAgger round 2 (42.1 %) | `runs/flow/fm_stack/bc_best.pt` demos only (7.3 %) | seeds 41, 42, 16 envs each (diffusion 5/16, 8/16; flow 1/16, 1/16) | diffusion b0 env 10, b1 env 12; flow b0 env 8, b1 env 6 |
Rates are the held-out numbers from section 1/3, not the rates of the shown episodes. **Stack caveat:** flow matching is the weak demos-only model (7.3 %); its two successes were searched for in 32 envs, there was no env where both models succeeded, and the diffusion attempts are different (similar) start states.

The earlier bug and its fix: the first overlay version drew forward kinematics of the **commanded** joint targets. The env's joint PD (stiffness 80, gravity on) settles up to 0.27 rad away from its target (joints 2 and 4), about 140 px at the gripper, so the converged lines sat far from the real gripper (median error 143 px; the original `report/media` clips of 13:2x had this bug and were replaced). The fix first modelled gravity sag, then the lines were taken from the simulator itself (restore and re-simulate). Check: re-simulating the executed chunk from the restored state reproduces the recorded fingertip path to 0.02-0.10 px (drawer, stack, lift; mean over the 8 executed steps at 640x480; this tests the restore, not unexecuted chunks); FK vs the simulator's panda_hand: 4e-7 m (drawer, stack), 1e-6 m (lift); the fingertip at the close command is 3-7 mm from the drawer handle and 10-16 mm from stack cube 2.

Pour (added later, `media/task_pour_converge.mp4`, 22 s, data `data/task_pour.json`, code `pour_*.py`, `make_pour_data.py`, `video/src/pourvideo.tsx`, composition TaskPour): diffusion `pouring_diffusion/runs/bc_lag1200` with 50 DDIM steps (51.5 %; only 10 of the 50 states are shown), flow `fm_lag1200` with 5 Euler steps (57.0 %), seed 201, env 2, one attempt per model, both succeed. The env has no water particles: success is `subgoal >= 9` (pour stage done, both objects placed and released, hands retreated), not a water measure. The lines are the **bottle-mouth path** per sampled chunk (simulator-restored, 14-D two-arm chunks, 32 predicted / 16 executed); markers are the cup opening (yellow) and the pour region of the env (green). Replay call = the call containing the start of the tilt (call 14, env step 208, both models). The 32 final samples agree within about 2 mm (unimodal); early noisy states snap the grip, so the bottle can drop in those lines; the flow clip has no velocity arrows or field. The pour code and media were promoted unchanged; see `media/README_task_videos.md` for its checks and caveats.
