# Script inventory

Reference tables for the scripts in this repository. The step-by-step instructions are in the top-level [README](../README.md).
"Checked against the code only" means the arguments were read from the source / parsed with the script's real `parse_args()`; the script was not run.

## `run_*.sh` pipelines (repository root)

They skip steps whose output exists (re-run to resume), log to `runs/**.log`, and use `ISAACLAB_PYTHON` or `$HOME/miniconda3/envs/isaaclab/bin/python`.
Several wait for another script with `pgrep` (one GPU job at a time); the wait is a no-op when that script is not running. A clean clone should use
`scripts/run_repro_minimal.sh` (the shortest working order); the table says what each original script produced and what it depends on.

| Script | What it does | Depends on / notes | Produced |
|---|---|---|---|
| `run_pipeline.sh`, `run_pipeline_wrist.sh` | lift: PPO teacher eval, 300 clean + 200 DART demos, student from scratch, eval on seeds 201/301 (wrist adds the eye-in-hand camera) | `TEACHER=` PPO checkpoint | first lift students (42 % workspace camera, 56.3 % with wrist) |
| `run_dagger.sh` | lift DAgger with the PPO teacher (it hurt: 23-48 %) | `TEACHER`, `runs/dp_wrist`, `runs/demos_wrist_*` from `run_pipeline_wrist.sh` | negative DAgger result |
| `run_more_data.sh` | more clean + DART demos and longer training | `TEACHER`, `run_pipeline_wrist.sh` outputs | 59.8 % model |
| `run_recovery.sh`, `run_robust_teacher.sh` | recovery demos + fine-tune; robust PPO teacher + perturbation suite | `TEACHER`, `run_more_data.sh` outputs | 72.5 % lift model |
| `run_resolution.sh` | zoom camera / no-max-pool tests (fine-tune chains from `runs/dp_recovery`), collects the five zoom demo sets, trains the robust teacher | `TEACHER`, `runs/dp_recovery/bc_best.pt` and the `demos_wrist_*` sets | zoom model (75.2 %); the 1,900 zoom demos used by `run_flow.sh` |
| `run_final.sh` | robust-teacher demos + one DAgger round on the zoom model, test seeds 1001-1005 | `RT=` robust-teacher checkpoint, `runs/res_zoom` | 76.4 % lift model |
| `run_v3.sh` | "v3" literature recipe (`train_dp_v3.py`, `fd_core/model_v3.py`) | `TEACHER`, waits for `run_more_data.sh` | about 45 %, renderer-sensitive, not used |
| `run_drawer.sh` | drawer with the **PPO** teacher | `logs/rsl_rl/franka_drawer` (`train_teacher_task.py`) | the 8.7 % student |
| `run_night.sh` | scripted drawer + stack teachers: demos, student, tests, videos | stack half expects `runs/stack/demos_clean` (from `run_stack.sh`) and trains on clean + DART 0.03, which is **not** the reported stack model | drawer 92.2 % (`runs/drawer_s/`) |
| `run_stack.sh` | stacking student from scripted demos, clean + DART 0.10 | waits for `run_drawer.sh` if running | older stack variant, not the reported one |
| `run_stack2.sh` | stacking student from clean + DART 0.01 (collects both if missing) | none | stack demos-only 10.2 % (`runs/stack/dp`) |
| `run_stack_dagger.sh` | stack DAgger rounds 1-3 (resync teacher), fine-tune chain, test seeds | `runs/stack/dp`, `demos_clean`, `demos_dart01` | DAgger round 2: 42.1 % |
| `run_flow.sh` | flow matching vs diffusion on stack, drawer, lift (both from scratch on the zoom demos), steps sweep, pour | `runs/stack/demos_*`, `runs/drawer_s/demos_*`, `runs/zoom_*`; the pour part needs the external pour environment and its demos (it reports FAILED otherwise and continues) | the main results table |
| `run_improve.sh` | clean re-evaluation on seeds 1002-1005, from-scratch retraining on all DAgger data, round 4 | outputs of `run_stack2.sh`, `run_stack_dagger.sh`, `run_flow.sh` | DAgger table in `RESULTS.md` |

## Visualisation scripts

| Output | Script | Needs |
|---|---|---|
| Attention (spatial-softmax) videos, lift | `record_attention.py --checkpoint CKPT --out-dir DIR`, then `render_attention.py OUT.mp4 EP.npz[:note] ...` | Isaac Sim + checkpoint |
| Attention, drawer / stack | `attention_task.py --task {drawer,stack} --checkpoint CKPT --out-dir DIR` | Isaac Sim + checkpoint |
| Attention stills | `make_attention_stills.py out.png P hz npz:frame:label ...` | the npz of the scripts above (no simulator) |
| Denoising / flow-integration overlay (lift): the real DDIM / Euler iterates | `record_denoise.py --dp DP.pt --fm FM.pt --out-dir runs/denoise --seed 41` | Isaac Sim + both checkpoints |
| Same for drawer / stack | `report/record_task.py --task {drawer,stack} --dp ... --fm ... --out-dir runs/task_<task>` | Isaac Sim + checkpoints |
| Simulator-replayed chunk paths | `report/sim_chunks.py` (lift), `report/sim_task.py` (drawer, stack), `report/pour_sim.py` (pour) | Isaac Sim + the recordings |
| Pour recording | `report/pour_record.py`, `report/pour_select.py`, `report/pour_common.py` | Isaac Sim + the external pour environment + pour checkpoints |
| Video data JSON | `report/make_overlay_data2.py`, `make_task_data.py`, `make_pour_data.py`, `make_lift_task_data.py`, `make_denoise_data.py` | the recordings above (`runs/*`); no simulator |
| Saliency, occlusion, receding-horizon, learning-curve data (`report/data/*.json`) | `report/make_figures_data.py`, `make_figures_data2.py` | a trained lift checkpoint and held-out demos (`runs/dp_wrist_more`, `runs/v3_clean`); GPU, no simulator |
| Recovery-test videos / failure taxonomy | `record_recovery.py`, `diagnose.py` | Isaac Sim + teacher + checkpoint |
| Cosmetic wood-cabinet drawer videos | `record_wood.py` | Isaac Sim + checkpoint |
| Slides (optional) | `report/build_pptx.py`, `report/build_onepager.py` (`python-pptx`), `report/tools/make_pngs.py` (`cairosvg`) | optional pip packages |
| Results video | `report/video/` (Remotion): `cd report/video && npm install && npx remotion studio` | node, the clips in `report/video/public` |

Other helpers: `preview_task.py` (save camera images to set up camera poses), `debug_drawer.py`, `debug_drawer_s.py`, `debug_stack.py` (teacher debugging).

## Paths and environment variables

| Variable | Meaning | Default |
|---|---|---|
| `DP_ROOT` | folder containing `dp_core/` | `pouring_diffusion/` of this repository |
| `TRLDISTILL_V5` | checkout of the external pour environment | `../TRLDistill_v5` (sibling of this repository) |
| `POURING_DIFFUSION` | pour scripts' working folder (`run_flow.sh`) | `pouring_diffusion/` of this repository |
| `ISAACLAB_PYTHON` | python used by the `run_*.sh` scripts | `$HOME/miniconda3/envs/isaaclab/bin/python` |
| `FD_PLOTS_OUT` | output folder of `report/make_plots.py` | `report/export_build` |
| `FD_RENDER=v2` | deterministic-image lift renderer (experimental, not used for reported numbers) | `v1` |

Scripts that read recordings or checkpoints (`runs/...`) expect them relative to the repository root.
