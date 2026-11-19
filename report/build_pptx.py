#!/usr/bin/env python3
"""Build the PowerPoint version of the report slide (python-pptx): 7 slides, embedded videos with poster frames,
native bar/column charts, matplotlib plots for the motion figures. Output: report/export_build/slides/*.pptx"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

R = Path(__file__).resolve().parent
B = R / "export_build"; MEDIA = R / "media"; PLOTS = B / "plots"; POST = B / "posters"
OUT = B / "slides"; OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, LINE, BG2 = "12202B", "5A6A77", "D3DBE2", "EEF1F4"
DP, DPS, TEACH, REF, NEG, OK = "DF6327", "F3C3A8", "3D5568", "8B98A4", "B8473F", "2F8A57"
HEAD, BODY = "Arial", "Calibri"
rgb = lambda h: RGBColor.from_string(h)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
TOTAL = 8


def text(slide, x, y, w, h, runs, size=14, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None):
    """runs: str, or list of paragraphs, each a str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        for t, o in ([(p, {})] if isinstance(p, str) else p):
            r = para.add_run(); r.text = t
            f = r.font; f.name = o.get("font", font); f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def frame(slide, n, eyebrow, title):
    text(slide, 0.5, 0.25, 12.3, 0.24, eyebrow.upper(), size=10, color=MUTED, font=BODY, name="Eyebrow")
    text(slide, 0.5, 0.52, 12.3, 0.7, title, size=28, color=INK, font=HEAD, bold=True, name="Title")
    text(slide, 0.5, 7.08, 9, 0.25, "One recipe, two robots · privileged RL teacher → diffusion policy student", size=9, color=MUTED, name="Footer")
    text(slide, 11.8, 7.08, 1.03, 0.25, f"{n} / {TOTAL}", size=9, color=MUTED, align=PP_ALIGN.RIGHT, name="Slide number")


def video(slide, name, x, y, w, h):
    mv = slide.shapes.add_movie(str(MEDIA / f"{name}.mp4"), Inches(x), Inches(y), Inches(w), Inches(h),
                                poster_frame_image=str(POST / f"{name}.png"), mime_type="video/mp4")
    mv.name = f"Video {name}"
    return mv


def image(slide, path, x, y, w=None, h=None, name=None):
    p = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
    if name:
        p.name = name
    return p


def chip(slide, x, y, w, h, label, fill, fg, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.18
    s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(1)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame; tf.margin_left = tf.margin_right = Inches(0.08); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = label; r.font.size = Pt(12); r.font.name = BODY; r.font.bold = True; r.font.color.rgb = rgb(fg)
    return s


def card(slide, x, y, w, h):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.06; s.fill.solid(); s.fill.fore_color.rgb = rgb(BG2); s.line.fill.background(); s.shadow.inherit = False
    return s


def stat(slide, x, y, w, big, small, color=DP, big_size=30, name=None):
    text(slide, x, y, w, 0.55, big, size=big_size, color=color, font=HEAD, bold=True, name=name)
    text(slide, x, y + 0.58, w, 0.5, small, size=12, color=MUTED)


def hbar(slide, x, y, w, h, rows, title):
    """rows: (label, value, se, color) top-to-bottom; native clustered bar chart, per-bar colours, value labels."""
    rows = rows[::-1]                                         # bar charts draw the first category at the bottom
    cd = CategoryChartData()
    cd.categories = [f"{l}  (±{se:.1f})" if se else l for l, v, se, c in rows]
    cd.add_series("Success (%)", [v for l, v, se, c in rows])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = title
    ch = gf.chart
    ch.has_title = True; ch.chart_title.text_frame.text = title
    tp = ch.chart_title.text_frame.paragraphs[0]; tp.runs[0].font.size = Pt(14); tp.runs[0].font.bold = True
    tp.runs[0].font.name = BODY; tp.runs[0].font.color.rgb = rgb(INK)
    ch.has_legend = False
    pl = ch.plots[0]; pl.gap_width = 55; pl.vary_by_categories = False
    pl.has_data_labels = True
    dl = pl.data_labels; dl.number_format = '0.0"%"'; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(11); dl.font.bold = True; dl.font.name = BODY; dl.font.color.rgb = rgb(INK)
    for i, (l, v, se, c) in enumerate(rows):
        pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
    va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = 100; va.major_unit = 25
    va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb(LINE)
    va.tick_labels.font.size = Pt(10); va.tick_labels.font.color.rgb = rgb(MUTED); va.tick_labels.number_format = '0"%"'
    va.tick_labels.number_format_is_linked = False; va.format.line.fill.background()
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(11); ca.tick_labels.font.color.rgb = rgb(INK); ca.format.line.color.rgb = rgb(LINE)
    return gf


def recovery_chart(slide, x, y, w, h):
    """Clustered bars: success (of 100) under injected failures for teacher / DP before / DP + recovery demos."""
    cats = ["No perturbation", "Missed grasp", "Cube dropped", "Cube knocked", "Pushed in hand (held-out)", "Arm shoved (held-out)", "Late knock (held-out)"]
    cd = CategoryChartData(); cd.categories = cats[::-1]
    for name, vals in (("DP · zoom", [76, 51, 12, 49, 81, 76, 37]), ("DP + robust-teacher DAgger", [74, 54, 38, 56, 75, 75, 43])):   # teacher (100/98/90/84/100/100/90) in the notes
        cd.add_series(name, vals[::-1])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = "Recovery under injected failures"; ch = gf.chart
    ch.has_title = True; ch.chart_title.text_frame.text = "Recovery under injected failures · success of 100 (last 3 held-out)"
    r0 = ch.chart_title.text_frame.paragraphs[0].runs[0]; r0.font.size = Pt(14); r0.font.bold = True; r0.font.name = BODY; r0.font.color.rgb = rgb(INK)
    ch.has_legend = True; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(10); ch.legend.font.name = BODY
    from pptx.enum.chart import XL_LEGEND_POSITION
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    pl = ch.plots[0]; pl.gap_width = 30; pl.overlap = 0; pl.has_data_labels = True
    dl = pl.data_labels; dl.number_format = '0"%"'; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(8); dl.font.name = BODY; dl.font.color.rgb = rgb(INK)
    for ser, c in zip(list(pl.series), (DPS, DP)):     # same order the series were added
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(c)
    va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = 100; va.major_unit = 25
    va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb(LINE)
    va.tick_labels.font.size = Pt(10); va.tick_labels.font.color.rgb = rgb(MUTED); va.format.line.fill.background()
    va.tick_labels.number_format = '0"%"'; va.tick_labels.number_format_is_linked = False
    ch.category_axis.tick_labels.font.size = Pt(11); ch.category_axis.tick_labels.font.color.rgb = rgb(INK)
    return gf


def column(slide, x, y, w, h, cats, vals, colors, title):
    cd = CategoryChartData(); cd.categories = cats; cd.add_series("Wrist share (%)", vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    gf.name = title; ch = gf.chart
    ch.has_title = True; ch.chart_title.text_frame.text = title
    r0 = ch.chart_title.text_frame.paragraphs[0].runs[0]; r0.font.size = Pt(14); r0.font.bold = True; r0.font.name = BODY; r0.font.color.rgb = rgb(INK)
    ch.has_legend = False; pl = ch.plots[0]; pl.gap_width = 70; pl.has_data_labels = True
    dl = pl.data_labels; dl.number_format = '0"%"'; dl.number_format_is_linked = False; dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(12); dl.font.bold = True; dl.font.name = BODY; dl.font.color.rgb = rgb(INK)
    for i, c in enumerate(colors):
        p = pl.series[0].points[i]; p.format.fill.solid(); p.format.fill.fore_color.rgb = rgb(c)
    va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = 100; va.major_unit = 25
    va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb(LINE)
    va.tick_labels.font.size = Pt(10); va.tick_labels.font.color.rgb = rgb(MUTED); va.format.line.fill.background()
    va.tick_labels.number_format = '0"%"'; va.tick_labels.number_format_is_linked = False
    ch.category_axis.tick_labels.font.size = Pt(11); ch.category_axis.tick_labels.font.color.rgb = rgb(INK)
    return gf


# ---------------------------------------------------------------- 1. overview
s = prs.slides.add_slide(BLANK)
text(s, 0.5, 0.3, 12.3, 0.3, "TEACHER–STUDENT DISTILLATION · ISAAC LAB · ISAAC SIM 5.1", size=11, color=MUTED, name="Eyebrow")
text(s, 0.5, 0.6, 12.3, 1.3, [[("One recipe, two robots: ", {}), ("diffusion policy", {"color": DP}), (" students", {})],
                              [("from privileged RL teachers", {})]], size=40, font=HEAD, bold=True, name="Title")
labels = [("RL teacher · privileged state", "FFFFFF", TEACH, TEACH), ("teacher demos + DART noise", "FFFFFF", INK, LINE),
          ("diffusion policy · RGB + proprio", DP, "FFFFFF", None), ("frozen eval · held-out seeds", "FFFFFF", INK, LINE)]
cx = 0.5
for i, (lab, fill, fg, ln) in enumerate(labels):
    chip(s, cx, 2.12, 2.75, 0.46, lab, fill, fg, ln)
    if i < 3:
        text(s, cx + 2.78, 2.12, 0.4, 0.46, "→", size=18, color=MUTED, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    cx += 3.18
video(s, "pouring_hero", 0.5, 2.9, 5.6, 3.17)
video(s, "franka_dp_2x2", 6.3, 2.9, 4.23, 3.17)
text(s, 0.5, 6.15, 5.6, 0.5, [[("Dual-arm UR + Allegro · water pouring", {"bold": True, "color": INK}), ("  scene (top), policy camera (bottom)", {"color": MUTED})]], size=12)
text(s, 6.3, 6.15, 4.23, 0.5, [[("Franka Panda · cube lift", {"bold": True, "color": INK}), ("  cameras only", {"color": MUTED})]], size=12)
stat(s, 10.85, 2.9, 2.0, "63%", "pouring success\n(teacher 76%)", name="Stat pouring")
stat(s, 10.85, 4.02, 2.0, "76.4%", "Franka, fresh test\n(teacher 96%)", name="Stat Franka")
stat(s, 10.85, 5.14, 2.0, "2.3 mm", "cube position\nread from pixels", color=TEACH, name="Stat cube")
text(s, 0.5, 7.08, 9, 0.25, "Success over 200 episodes per held-out seed · same diffusion-policy code on both robots", size=9, color=MUTED, name="Footer")
text(s, 11.8, 7.08, 1.03, 0.25, f"1 / {TOTAL}", size=9, color=MUTED, align=PP_ALIGN.RIGHT, name="Slide number")
s.notes_slide.notes_text_frame.text = ("One distillation recipe on two robots: a privileged RL teacher (MAT for pouring, PPO for Franka) generates "
    "demonstrations, a transformer diffusion policy learns from camera images plus joint states, and it is evaluated on held-out seeds.")

# ---------------------------------------------------------------- 2. pouring
s = prs.slides.add_slide(BLANK)
frame(s, 2, "01 · Dual-arm water pouring · 2× UR + Allegro · 14-D joint actions", "One camera, two arms: 63% vs the teacher's 76%")
video(s, "pouring_hero", 0.5, 1.45, 6.9, 3.9)
text(s, 0.5, 5.42, 6.9, 0.35, [[("Diffusion policy", {"bold": True}), (" · two rollouts · scene (top) and what the policy camera sees (bottom)", {"color": MUTED})]], size=12)
hbar(s, 7.75, 1.35, 5.1, 4.4, [("Teacher · privileged", 76.0, 0, TEACH), ("DP · 1,200 demos", 63.0, 2.41, DP), ("DP · no-lag model", 62.5, 2.42, DP),
                                 ("TMAD-RGB (non-diffusion)", 89.2, 0, REF)],
     "Success rate · 50 DDIM sampling steps")
stat(s, 0.5, 5.95, 3.3, "+16.5 pts", "50 vs 5 DDIM sampling steps\n(46.5 → 63%)")
stat(s, 4.1, 5.95, 3.3, "1,200", "teacher demos with arm lag\nfitted to the real driver", color=TEACH)
text(s, 7.75, 5.95, 5.1, 0.9, "±1 SE in brackets · 200 episodes per held-out seed (201, 301) · TMAD-RGB is the thesis student (not diffusion), shown for reference",
     size=11, color=MUTED)
s.notes_slide.notes_text_frame.text = "Pouring: success = pour stage ≥ 9 of 10. Sampling with 50 instead of 5 DDIM steps added 16.5 points."

# ---------------------------------------------------------------- 3. Franka
s = prs.slides.add_slide(BLANK)
frame(s, 3, "02 · Franka Panda cube lift · 8-D joint + gripper actions", "Cameras only: 76.4% on a fresh test set vs the teacher's 96%")
video(s, "franka_teacher_2x2", 0.5, 1.4, 3.3, 2.475)
video(s, "franka_dp_2x2", 3.95, 1.4, 3.3, 2.475)
text(s, 0.5, 3.92, 3.3, 0.3, [[("Teacher", {"bold": True}), (" · privileged · 96.0%", {"color": MUTED})]], size=12)
text(s, 3.95, 3.92, 3.3, 0.3, [[("After", {"bold": True}), (" · + recovery demos · 72.5%", {"color": MUTED})]], size=12)
video(s, "franka_before_failures", 0.5, 4.35, 6.75, 1.69)
chip(s, 0.6, 4.43, 0.95, 0.3, "BEFORE", NEG, "FFFFFF")
text(s, 0.5, 6.12, 6.75, 0.6, [[("Before", {"bold": True, "color": NEG}), (" · no recovery behaviour: misses the cube, never re-grasps · 59.8%", {"color": MUTED})]], size=12)
hbar(s, 7.6, 1.3, 5.25, 4.2, [("Teacher · privileged", 96.0, 1.39, TEACH), ("DP · workspace cam", 42.0, 2.47, DP), ("DP · + wrist cam", 56.3, 2.48, DP),
                               ("DP · + recovery demos", 72.5, 2.23, DP), ("DP · + zoom · test", 75.2, 1.37, DP), ("DP · + robust DAgger · test", 76.4, 1.34, DP),
                               ("DAgger · PPO teacher", 48.0, 2.50, NEG)],
     "What moved Franka success")
stat(s, 7.6, 5.7, 2.5, "+14 pts", "wrist camera\n(42 → 56%)")
stat(s, 10.35, 5.7, 2.5, "48 / 23%", "DAgger with a PPO teacher\n(negative result)", color=NEG)
s.notes_slide.notes_text_frame.text = ("Franka: success = cube within 5 cm of a random goal after 5 s. 400 episodes per bar (teacher 200), ±1 SE in brackets. "
    "The wrist camera fixed most grasp misses; DAgger failed because the PPO teacher labels off-distribution states poorly.")

# ---------------------------------------------------------------- 4. where it looks
s = prs.slides.add_slide(BLANK)
frame(s, 4, "03 · Occlusion sensitivity · grey out a 16 px patch, measure the change in the predicted actions",
      "Where the policy looks: gripper and cube, then the wrist view")
for i, lab in enumerate(["Policy input · workspace | wrist", "Trained policy", "Random-weights baseline"]):
    text(s, 0.5 + i * 4.12, 1.32, 4.0, 0.3, lab, size=12, bold=True, color=[INK, DP, MUTED][i])
image(s, MEDIA / "occlusion_approach.png", 0.5, 1.65, w=12.33, name="Occlusion approach")
text(s, 0.5, 3.75, 12.3, 0.3, "Approach (0.5 s): the trained map peaks on the gripper and cube; the random baseline lights up the arm and background",
     size=12, color=MUTED)
image(s, MEDIA / "occl_grasp.png", 0.5, 4.3, w=3.5, name="Occlusion grasp")
image(s, MEDIA / "occl_lift.png", 4.2, 4.3, w=3.5, name="Occlusion lift")
text(s, 0.5, 6.1, 3.5, 0.3, [[("Grasp · 1.4 s", {"bold": True}), (" · cube between the fingers", {"color": MUTED})]], size=12)
text(s, 4.2, 6.1, 3.5, 0.3, [[("Lift · 2.8 s", {"bold": True}), (" · trained map", {"color": MUTED})]], size=12)
column(s, 8.1, 4.1, 4.75, 2.75, ["approach\n0.5 s", "grasp\n1.4 s", "lift\n2.8 s"], [27.5, 72.5, 52.9], [DPS, DP, DPS],
       "Wrist-camera share of the top-10% patches")
s.notes_slide.notes_text_frame.text = ("Occlusion maps are normalised per panel, so compare locations, not magnitudes. "
    "The grasp baseline is not shown because the random-weights model also reacts to the large cube in the wrist view there.")

# ---------------------------------------------------------------- 4b. live attention
s = prs.slides.add_slide(BLANK)
frame(s, 5, "03 · Live attention · the encoder's spatial softmax on the images the policy sees, final model, unseen seed",
      "Success: attention locks onto the cube in the wrist view")
VW, VH = 7.9, 7.9 * 320 / 1072
video(s, "attention_success", 0.5, 1.35, VW, VH)
video(s, "attention_failure", 0.5, 1.35 + VH + 0.42, VW, VH)
text(s, 0.5, 1.35 + VH + 0.05, VW, 0.3, [[("Success", {"bold": True, "color": OK}), (" · attention follows the cube between the fingers through grasp and lift", {"color": MUTED})]], size=12)
text(s, 0.5, 1.35 + 2 * VH + 0.47, VW, 0.3, [[("Failure", {"bold": True, "color": NEG}), (" · attention sits on the empty table; the gripper closes beside the cube", {"color": MUTED})]], size=12)
text(s, 8.65, 6.0, 4.2, 0.9, "Heatmap: spatial-softmax attention, weighted by how much the policy uses each keypoint · dots: the 12 most-used keypoints · videos at 0.5x speed",
     size=10, color=MUTED)
cd = CategoryChartData(); cd.categories = [f"{0.2 * i + 0.1:.1f}" for i in range(25)]
cd.add_series("Success (22)", [.452, .508, .561, .568, .587, .608, .617, .614, .609, .61, .615, .621, .619, .621, .62, .617, .616, .615, .616, .615, .615, .616, .616, .615, .615])
cd.add_series("Failure (10)", [.472, .518, .546, .499, .516, .483, .507, .505, .491, .472, .475, .485, .467, .48, .473, .491, .452, .477, .495, .445, .47, .463, .465, .464, .426])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(8.65), Inches(1.35), Inches(4.2), Inches(3.3), cd); gf.name = "Wrist attention share"; ch = gf.chart
ch.has_title = True; ch.chart_title.text_frame.text = "Attention on the wrist view (32 rollouts)"
r0 = ch.chart_title.text_frame.paragraphs[0].runs[0]; r0.font.size = Pt(12); r0.font.bold = True; r0.font.name = BODY; r0.font.color.rgb = rgb(INK)
from pptx.enum.chart import XL_LEGEND_POSITION
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(10)
for ser, c in zip(list(ch.plots[0].series), (OK, NEG)):
    ser.format.line.color.rgb = rgb(c); ser.format.line.width = Pt(2.25); ser.smooth = False
va = ch.value_axis; va.minimum_scale = 0.4; va.maximum_scale = 0.7; va.major_unit = 0.1; va.tick_labels.number_format = '0%'
va.tick_labels.number_format_is_linked = False; va.tick_labels.font.size = Pt(9); va.major_gridlines.format.line.color.rgb = rgb(LINE)
ca = ch.category_axis; ca.tick_labels.font.size = Pt(9); ca.tick_label_position  # seconds
ca.has_title = True; ca.axis_title.text_frame.text = "time (s)"; ca.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(9)
stat(s, 8.65, 4.8, 2.0, "62%", "success: wrist share\nafter the grasp starts", color=OK)
stat(s, 10.85, 4.8, 2.0, "47%", "failure: attention\nstays on the table", color=NEG)
s.notes_slide.notes_text_frame.text = ("The spatial softmax is the policy's only path from pixels to actions: 32 channels, each a softmax over the 4x8 "
    "feature grid of the [workspace | wrist] image; the policy receives each channel's expected (x, y). Heatmap = channel maps weighted by "
    "the norm of their projection weights. 32 rollouts of the final model on seeds 601-602 (22 successes, 10 failures). On the approach "
    "both groups look the same; from the grasp on, successes shift attention to the wrist view (62% vs 47%) and concentrate it.")

# ---------------------------------------------------------------- 5. denoising
s = prs.slides.add_slide(BLANK)
frame(s, 6, "04 · How the policy generates motion · DDIM, 10 steps, 16-step chunks", "From noise to motion in 10 denoising steps")
image(s, PLOTS / "denoising_sequence.png", 0.5, 1.3, w=12.33, name="Denoising sequence")
image(s, PLOTS / "sampling_consistency.png", 0.5, 4.65, h=2.3, name="Sampling consistency")
card(s, 8.1, 4.75, 4.75, 2.1)
text(s, 8.35, 4.9, 4.3, 1.9, [[("24 noise seeds, one motion", {"bold": True, "size": 16})],
                              [("Different noise always converges to the teacher's chunk. The teacher is deterministic, so a tight spread is the expected result.", {"color": MUTED, "size": 13})]],
     size=14)
s.notes_slide.notes_text_frame.text = "Top: actual DDIM samples x_k at k = 10, 7, 4, 1, 0 for three joints vs the teacher chunk (dark), held-out state."

# ---------------------------------------------------------------- 6. tracking + inference
s = prs.slides.add_slide(BLANK)
frame(s, 7, "05 · Teacher-forced, open-loop predictions on held-out episodes", "Tracking the teacher, chunk by chunk")
image(s, PLOTS / "receding_horizon.png", 0.5, 1.3, w=7.9, name="Receding horizon")
image(s, PLOTS / "cube_from_pixels.png", 8.75, 1.3, w=4.1, name="Cube from pixels")
stat(s, 8.75, 3.85, 4.1, "0.046 rad", "mean chunk error · 300 chunks,\n20 held-out episodes")
recovery_chart(s, 0.5, 4.45, 7.9, 2.5)
video(s, "franka_recovery_successes", 8.75, 4.75, 4.1, 1.025)
chip(s, 8.82, 4.8, 1.05, 0.26, "RECOVERY", OK, "FFFFFF")
text(s, 8.75, 5.85, 4.1, 1.0, [[("After recovery demos", {"bold": True, "color": OK})],
                               [("cube dropped or knocked away → re-grasped and lifted. Robust-teacher DAgger: drop re-grasp 12 → 38%; held-out perturbations ≈ unperturbed", {"color": MUTED, "size": 12})]], size=13)
s.notes_slide.notes_text_frame.text = ("Each chunk is predicted from the teacher's own observation at its start step (teacher-forced), so this is not a closed-loop rollout. "
    "Recovery test (seed 401, 100 episodes each): robust teacher 100/98/90/84 (none/missed grasp/drop/knock) and 100/100/90 on the held-out kinds (pushed in hand/arm shoved/late knock), which never appear in the training data. Earlier inference ablation: replanning 2x/4x more often or 50 DDIM steps gave no gain.")

# ---------------------------------------------------------------- 7. findings
s = prs.slides.add_slide(BLANK)
frame(s, 8, "06 · Lessons", "What mattered")
cards = [("+14 pts", "Wrist camera", "fixed most grasp misses (42 → 56%)", DP),
         ("+16.5 pts", "50 DDIM steps", "on pouring vs 5 in training; no gain on Franka", DP),
         ("+12.7 pts", "Recovery demos", "59.8 → 72.5%: the policy learns to re-grasp", DP),
         ("48 / 23%", "DAgger", "the PPO teacher mislabels states it never visited", NEG),
         ("76.4%", "Fresh test set", "1,000 episodes on never-used seeds; drop re-grasp 12 → 38%", DP)]
for i, (big, head, sub, col) in enumerate(cards):
    x = 0.5 + i * 2.5
    card(s, x, 1.55, 2.3, 3.0)
    text(s, x + 0.2, 1.75, 1.95, 0.7, big, size=24, font=HEAD, bold=True, color=col)
    text(s, x + 0.2, 2.6, 1.95, 1.85, [[(head, {"bold": True, "size": 16})], [(sub, {"color": MUTED, "size": 14})]], size=14)
text(s, 0.5, 4.95, 12.3, 1.6, [
    [("Protocol  ", {"bold": True}), ("success over 200 episodes per held-out seed (201, 301), ±1 SE · Franka: cube within 5 cm of a random goal at 5 s · pouring: stage ≥ 9 of 10", {"color": MUTED})],
    [("Model  ", {"bold": True}), ("transformer diffusion policy · ResNet18 + GroupNorm + spatial softmax · obs 2 · predict 16 · execute 8 (Franka) / 16 (pouring) · DDPM 100 / DDIM 10–50", {"color": MUTED})],
    [("Next  ", {"bold": True}), ("a noise-robust teacher so DAgger helps, or a small residual RL policy on top of the diffusion policy", {"color": MUTED})]], size=13)
s.notes_slide.notes_text_frame.text = "Findings with their numbers; the v3 literature recipe (about 45%) is omitted from the charts because it was renderer-sensitive."

out = OUT / "Diffusion_Policy_Across_Robots.pptx"
prs.save(out)
print("saved", out, out.stat().st_size // 1024, "KB")
