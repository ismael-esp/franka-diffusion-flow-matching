# Third-party notices

The code and text of this repository are MIT-licensed (see `LICENSE`). This repository does not include or redistribute the third-party
software or assets below; each keeps its own licence. Where a licence is not stated here, see the upstream project.

## Simulation

- **NVIDIA Isaac Lab** (https://github.com/isaac-sim/IsaacLab): used as a dependency (Franka lift, drawer and stack tasks, rsl_rl PPO integration).
  The core is BSD-3-Clause; some extensions carry other licences, see upstream.
- **NVIDIA Isaac Sim** (https://developer.nvidia.com/isaac/sim): used as a dependency under NVIDIA's own licence and EULA, see upstream.
- **Robot and scene assets** (Franka Emika Panda, cabinet and cubes, and for the pour task the UR3 and Allegro hand models): loaded at run time from
  NVIDIA's asset library / the Isaac Sim assets. They appear in the rendered videos and posters in `report/media` and on the project website.
  They keep their own licences, see upstream (NVIDIA Isaac Sim asset licence, and the original manufacturers' terms).

## Methods and papers (cited, no code copied from the papers)

- Diffusion Policy (Chi et al., 2023), Flow Matching (Lipman et al., 2022), Rectified Flow (Liu et al., 2022), DDIM (Song et al., 2020), and, for the pour
  experiments in `pouring_diffusion/dp_core/`, DSRL (Wagenmaker et al., 2025) and DPPO (Ren et al., 2024). See the citation list in `README.md`.
  The network and training code in this repository is part of the MIT-licensed code; it follows the methods of these papers. Their reference implementations have
  their own licences, see upstream.

## Video project and fonts

- **Remotion** (https://www.remotion.dev), used by `report/video/`: published under its own source-available licence, not an OSI licence. It is free for individuals and
  small teams; larger organisations may need a paid licence. See https://www.remotion.dev/license for the terms that apply to you before using or rendering the video project.
  Remotion is installed by `npm install`, not included.
- **Archivo** and **IBM Plex Sans**: SIL Open Font License 1.1 (OFL-1.1). They are self-hosted by the project website (`docs/assets/fonts`, with the licence texts next to them);
  not included in the code tree.

## Python and JavaScript dependencies

Used by name only, each under its own licence (see upstream): PyTorch, torchvision, NumPy, Matplotlib, Pillow, imageio, imageio-ffmpeg, gymnasium, rsl-rl,
python-pptx and cairosvg (optional); Node packages of the Remotion project as listed in `report/video/package.json`.

## External environment

The bimanual pour task needs an external Isaac Lab environment (`TRLDistill_v5`) that is not part of this repository and is not released with it; any
licence terms of that environment apply to it separately.
