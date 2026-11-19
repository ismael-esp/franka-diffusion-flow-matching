# Report design (index.html, page2.html, page3.html, one-slide PowerPoint)

Subject: results of a diffusion-policy and flow-matching study on robot manipulation. Audience: robotics readers who scan
a single dense slide. Job: make the headline result of each page readable first, then let the clips and charts prove it.

## Tokens (shared.css)

| Role | Name | Hex |
|---|---|---|
| Page / panel | paper / white | #f5f7f9 / #ffffff |
| Text / muted / rule | ink / muted / line | #12202b / #55646f / #d3dbe2 |
| Scripted teacher | slate | #3d5568 |
| Diffusion student | orange | #df6327 (soft #f3c3a8) |
| Flow matching | teal | #149484 |
| Failure / success | brick / green | #b8473f / #2f8a57 |

Colour follows the entity on every page: teacher slate, diffusion orange, flow matching teal, failure brick.
Type: Archivo condensed (headlines, key numbers; Arial Narrow in PowerPoint) and IBM Plex Sans (everything else; Calibri
in PowerPoint). Charts use Liberation Sans/Arial so the browser and the rasterised PowerPoint charts match. No monospace.
Scale: key number 56 px (pages 2, 3) / 36 px (page 1), headline 42 px, panel title 21, chart title 16, body 14, captions 12.5, labels 12.
Panels: 8 px radius, 1 px line, no shadows. Media is square-cornered because PowerPoint pictures are.

## Decisions

- The key results strip moved from the bottom to directly under the header and carries the largest type on each page.
  Before-and-after results are written as "from, grey" arrow "to, ink" (8.7 to 92.2 %, 10.9 to 42.1 %, 62.8 to 78.2 %, 27.5 to 57.0 %).
- Removed: tracked all-caps labels, mono micro-labels, middle-dot strings, the one-phrase coloured headline word,
  the 01-11 panel numbers (the panels are not a sequence), chip boxes around the pipeline steps (now plain text with arrows).
- Motion (web only, shared.js): one load sequence. Bars grow from their baseline, then the key numbers count up. It is off with
  `?static=1` (used for the PowerPoint capture) and with prefers-reduced-motion. No PowerPoint animation.

## Layouts (1600 px wide, height at most about 1600 px; slide = 16 x 16 in)

Page 1                              Page 2                              Page 3
```
header (headline, pipeline)         header                              header
key results x5 (36 px)              key results x4 (56 px)              key results x4 (56 px)
[pouring+attention][lift][config]   [drawer panel][stacking panel]      [4 flow-matching clips | encoder stills]
[how the policy generates motion]   [why: 2 charts]                     [3 charts: flow vs diffusion, steps, DAgger]
footer                              [encoder stills x3]                 footer
                                    footer
```

## Pipeline

`tools/capture.js` (run in the page with `?static=1`) posts every chart SVG and `layout.json` to `tools/recv.py`;
`tools/make_pngs.py` rasterises the charts (4x); `build_onepager.py` writes the deck from `layout.json` + charts + posters;
`tools/preview_pptx.py` draws a rough PIL preview of the deck (no LibreOffice here).
The builder carries: flat fills with alpha, 1-3 px borders/rules, corner radius, font family/size/weight/colour, paragraphs
(headline line breaks come from the browser). It does not carry gradients, shadows, transforms or pseudo-element text, so none are used on captured elements.

## Page 4: two models, two diagrams, how each learns

Redesigned after feedback ("why one framework for both?"). Same tokens and pipeline (capture with `?static=1`, `tools/make_pngs.py`, `build_onepager.py` with four pages).
```
header (headline, 4 plain steps)
key facts x4 (17.5 M parameters | 10 -> 5 network calls | 0.0140 -> 0.0079 held-out action error | 16 -> 8 actions)
[ diffusion (orange): training step + DDIM sampling loop ] [ flow matching (teal): training step + Euler sampling loop ]
[ caption: what is literally shared code and what is not (with file:line) ]
[ diffusion: how it learns (5 corrupted chunks, sampling path) ] [ flow matching: same ]
footer (training recipe, identical for both)
```
`tools/make_diagrams.py` writes `diagrams/architecture_diffusion.svg`, `architecture_flow.svg` (new, one per model; slate = same code, colour = differs),
`learning_*.svg` and the older single `architecture.svg` (no longer on the page); it inlines them into `page4.html` from `page4.template.html`.
The video scenes `src/models.tsx` (two diagrams), `src/converge.tsx` (DiffusionConverge, FlowIntegrate: real rollout + real iterates) use the same tokens.

## Video: Results composition (83 s)

question 6 s, teacher ceiling 8, stacking 8, flow vs diffusion 9, two models two objectives 12 (models.tsx), diffusion converge 12, flow integrate 12
(converge.tsx), few steps 8, takeaway 8. Compositions `Results`, `DiffusionConverge`, `FlowIntegrate`, `Models`.
Convergence scenes: the rollout video (record_denoise.py, shown at 2 env steps per frame) pauses at policy calls 1, 9 and 16 (env steps 0, 64, 120) and replays that
call's 10 recorded iterates of the 16-step chunk for joints 1, 2 and 7; between pauses the chunk of the current call is shown with a cursor over its 8 executed steps.
Data: `data/denoise_lift_e1.json` (4 of 8 action dims), full iterates in `data/denoise_lift/*.npz`; regenerate with `make_denoise_data.py`.

## Page 4 task-video section and the deck (update)

A section "How each model samples an action chunk, over the real rollout" sits between the two architecture panels / shared-code caption and the two learning panels: four videos in one row (lift, drawer, stack, pour; two columns from 1100 to 1399 px, one column below 1100 px) with checkpoint and success-rate captions, then one caveat paragraph (simulator paths, smoothed for display, 16-step horizon, replay rule, other lift versions linked). The pour figure follows the same `<figure class="tvc">` pattern as the other three; deck slide 4 picks the videos up from a re-capture (`tools/capture.js`, `make_pngs.py`, `build_onepager.py`; posters in `export_build/posters/task_<task>_converge.png`).
The deck has the four videos embedded natively with poster frames on slide 4; slides 1-3 are unchanged (shape-by-shape diff against `export_build/backup_20261116_pre_v2/`), the slide is taller because page 4 is now the tallest page.
Remotion: `src/converge.tsx` (lift scenes, variants multi / single), `src/taskvideo.tsx` (the three task videos, compositions TaskLift / TaskDrawer / TaskStack), `Results` (98.6 s).
