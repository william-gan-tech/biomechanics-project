# Phase 7: Inline Double Push Detection & Technique Feedback

> Plan drafted 10/18. Targets marked *proposed* are to be confirmed before
> testing. Main track: inline double push. Side track: controlled ice
> footage with a real fatigue measure.

## Formal Research Question

> To what extent can single-camera pose estimation, extended to the feet,
> **detect the inline double push** and describe how well it is executed,
> clearly enough to help skaters learn and improve their technique?

**Side question (ice):** with controlled side-on footage and a real fatigue
measure (lap times or heart rate), does an ice skater's form change
measurably as they tire?

## Status (10/27): ✅ PHASE 7 COMPLETE. 7a: foot tracking 98% on full-body footage. 7c: detection closed after three pre-set tests (77.8%, 72.2%, 60.9%; target 85% not met; best result about 3 in 4, above chance). 7d: automatic feedback failed its pre-set test; visual comparison tool (`doublepush_compare.py`) built and judged clear and useful. 7e: `PHASE7_SUMMARY.md` (reviewed). Side track (ice fatigue): not started, future work.

## Why This Phase

- The goal running through Phases 4-6 is technique analysis that helps
  other skaters. Double push is one of the most important and hardest
  techniques in inline speed skating, and it has never been measured in
  this project (inline was listed as Phase 5 Tier 3, 5n-5r).
- Double push is mainly a **foot** movement: the skate sets down on its
  outside edge, pulls inward under the body, then pushes out. The pipeline
  so far uses hips, knees, shoulders and ankles only, so the feet are new
  ground.
- Phases 5 and 6 showed what single-camera footage can and can't do: 2D
  angles depend on camera view, front-on knees are unreliable, estimated 3D
  pose fixes trunk lean but not knees. For double push the camera lesson
  flips: the sideways foot path is best seen **from the front or behind**,
  while knee bend and timing are best seen from the side.

## Lessons Carried Over From Phases 5-6
- Every target is fixed before testing; results are reported whether they
  succeed or not.
- Labels a result depends on are made manually (no pre-filled guesses).
- Camera view is recorded for every clip and controlled in every comparison.
- 3D trunk measures replace 2D trunk lean wherever views are mixed.
- Pack-race footage can use the Phase 6 tracker and pure pieces.

---

## Tier 1: Main Track (Double Push)

### 7a: Foot Tracking Feasibility
MediaPipe gives heel (29/30) and toe (31/32) points; check whether they
follow the skate reliably on inline footage, front and side views.
- Frame sheets with the foot points drawn; manual check of whether each
  point sits on the skate.
- *Proposed target:* foot points on the correct skate in ≥90% of checked
  frames in front/back views. If not met, the phase falls back to ankle
  and knee paths.

**Result (10/18):** check sheets `doublepush/foot_check_00-01.jpg` (60 frames
from the ten clip sections in `doublepush/clips.csv`, heel and toe points
drawn), checked manually:

| Footage type | Frames | Feet on the correct skate |
|---|---|---|
| Full body visible (Mantia, Swings, Thorup full-body/end) | 42 | **41 (98%)** |
| Feet or legs only (Thorup 4:27-5:14, Swings 1:02.9) | 13 | 0 (11 no pose found, 2 wrong) |
| Skipped (title card, talking head, blur) | 5 | n/a |

- **Target met for full-body footage**, not for feet-only close-ups. The pose
  model finds the person first, so it needs the whole body in frame.
- Mantia's front-view sections are the strongest data (every full-body frame
  tracked). Thorup's feet-only close-ups stay useful as visual reference for
  defining the technique, not for measurement.
- For any new footage: whole body in frame, filmed from in front or behind.

### 7b: Double Push Ground Truth

**Technique definition (10/18), from coaching sources (Mantia, Thorup and
others):**
1. **Set-down and underpush:** the recovering skate lands near the centre of
   gravity, almost flat, under the belly button; core, hips and glutes drive
   it inward on an outside edge (the underpush).
2. **Edge transition and compression:** as the skate passes the hip line,
   the wheels are compressed evenly (no stomping or dropping weight) and the
   edge rolls smoothly from outside to inside.
3. **Classic second push:** a powerful outward extension of the leg while
   weight transfers to the other foot; upper body quiet, low and compact (no
   bouncing or side-to-side throwing).
4. **Fall and catch:** after pushing out with one leg, the body's weight is
   over the other side; the centre of gravity is allowed to fall back, and
   the recovering skate is placed under the body to catch it. The falling
   weight pressing into the outside edge powers the underpush.
5. **S-curves, not zig-zags:** the common mistake is forcing the skate far
   out on a sharp outside edge (unstable, loses direction). The skate should
   land nearly flat under the body; the sideways fall puts it on a slight
   outside edge. The wheels trace a smooth S-curve.
6. **Compression timing (advanced):** steady pressure into the wheels during
   the underpush, released outward like a spring into the classic push, with
   knees driven forward and ankles flexed.

**What can be measured from the front (pose points):**

| Technique element | Measure |
|---|---|
| Set-down under the body | Foot's sideways landing position relative to the hip midline |
| Underpush | Foot moving toward and under the body after set-down |
| Hip-line crossing, classic push | Moment the foot crosses under the hips; outward movement until lift-off |
| Fall and catch | Hips shifting sideways toward the landing foot just before set-down |
| Quiet upper body | Up-down and side-to-side movement of head and shoulders |
| S-curve vs zig-zag | Smoothness of the foot's sideways path |

**Probably not measurable from this footage:** edge roll (heel and toe
points show foot direction, not skate tilt); wheel compression and knee
drive (front-view knee angles were unreliable in Phase 6). Reported as
limits, not measured.

**Labelling plan:** stretches of video labelled "double push" or "normal
stride" (for detection, like the corner/straight labels); technique measures
computed automatically and spot-checked manually on a handful of strides.

**Labels so far (`doublepush/style_labels.csv`, 10/18):**
- Thorup "styles" short (three different skaters, front view, full body):
  speedskater style and short-track style = normal stride; inline = double
  push.
- Mantia 3:03-3:09 and 3:15-3:23 = **flawed double push** (same skater as
  his good double push):
  - 3:03-3:09: a huge pressure spike into the ground, let go, then regained
    at the end of the stroke; loses the full value of the double push and
    creates wild movement.
  - 3:15-3:23: too much upper-body movement; loud skates (quiet is usually
    better and faster). Upper body mostly out of frame in this shot.

**First look at the foot path (10/18):**
- Thorup short: in the double-push section each foot crosses well under the
  body to the other side before pushing out (left to -1.5, right to +1.2
  torso lengths); in the short-track section the feet barely cross (≤0.7);
  the speedskater section is in between (0.9-1.6). Promising, but one short
  example per style from three different skaters.
- Mantia 2:14-4:34: the flawed clips cross under the body as much as the rest
  (0.78 vs 0.55 torso lengths), as expected for a double push done badly.
  Framing also changes (upper body out of frame), which makes the torso scale
  unreliable in those clips.

**Measures and first test (10/19), `doublepush_measures.py`:** scaled by
leg length (hip to ankle); detection = share of frames with a foot past the
hip midline; quality = foot-path roughness, hip up-down movement,
upper-body sway (only with the upper body in frame).

| Stretch | Label | Leg (px) | Feet crossing (share of frames) |
|---|---|---|---|
| Short-track style (short) | normal | 390 | 0.21 |
| Speedskater style (short) | normal | 302 | 0.32 |
| Junior 300 m TT | normal | 162 | 0.10 |
| Inline style (short) | double push | 399 | 0.54 |
| Mantia 3:03-3:09 | flawed double push | 281 | 0.50 |
| Mantia 3:15-3:23 | flawed double push | 374 | 0.60 |
| Men 300 m TT | normal | **97** | 0.68 ✗ |
| 200 m sprint short | normal | **65** | 0.78 ✗ |

- On close front-view footage, every double push separated from every
  normal stride (6 stretches).
- On small, diagonal footage the measure breaks down: foot points jitter by
  about as much as the movement measured, and a diagonal camera turns
  forward travel into apparent sideways movement. **Rule:** report "can't
  judge" unless the view is roughly front or back and the leg is at least
  ~150 px.
- Quality: no same-skater comparison yet (the only good double push is a
  different skater).

**Elite-reference approach (10/20):** elite skaters (Mantia, Thorup) set the
good example instead of large amounts of footage. Labels added: the whole
Mantia stroke clinic is normal single push (50 stretches, ~6 min, front view
from a follow camera); the timestamped tutorial sections (minus the flawed
clips) and the Swings video are double push. Long stretches are measured in
4-second windows (pose found in ≥80% of frames, leg ≥150 px).

**Detection test (set up 10/20, `validate_7c_detection.py`):** cut-off on
"feet crossing under the body" halfway between Mantia's normal (clinic) and
double-push (tutorial) medians, set on Mantia only; tested on other skaters
(Swings, the three skaters in Thorup's styles short, junior 300 m time
trial). **Target: ≥85% of test windows correct.**

**Result (10/22): 77.8% (14 of 18 test windows), target not met.**

| | Windows | Correct |
|---|---|---|
| Mantia normal (rule's own skater) | 82 | 80% |
| Mantia double push | 90 | 78% |
| Mantia flawed double push | 3 | 100% |
| **Other skaters (test)** | **18** | **77.8%** |
| Swings (double push) | 14 | 71% |
| Short-film skaters and junior 300 m | 4 | 100% |

- Cut-off 0.32 (Mantia normal median 0.14, double push 0.51). Separates the
  styles on average, not reliably window by window.
- All 14 Swings windows were checked by eye without seeing the rule's
  answers: all show front or back view striding, so the result stands.
- **The distinctive part of a double push is the leg's path after the push,
  on its way back (the recovery)**; feet crossing mixes the recovery with
  the rest of the stride. Next: a recovery-path measure (foot heights now
  stored to time lift-off and touchdown), with a new pre-set target and
  fresh test clips, since these test skaters have now been seen.

**Second detection measure (defined 10/23, before testing), `recovery_measure.py`:**
- **Main (pass/fail): where the recovering foot lands.** Touchdown = each
  foot's lowest point below the hips in a front view; measure = the foot's
  sideways position at touchdown relative to the hips, in leg lengths
  (positive = out on its own side, near zero or negative = under the body).
  Double push lands under the body; single push lands out to the side.
- **Secondary (reported only): foot twist**, the change in heel-to-toe
  direction in the 0.5 s after touchdown (the inside foot twists for the
  second push in double push, glides out in single push).
- Both cut-offs set on Mantia only. **Target: ≥85% of fresh test windows
  correct.**
- **Fresh test set** (`doublepush/fresh_test_labels.csv`, never used to set
  the rule): Thorup guide 7:17-7:23 and 8:06-8:29, Thorup "0-60 kph"
  0:04-end, Thorup "Skate Faster" (single push 1:05-1:10, 1:18-1:24,
  1:42-1:52, 3:01-3:10, 3:41-3:57, 4:14-4:26, 4:50-4:56, 6:03-6:22; double
  push 5:14-5:38, 6:27-6:41). About 133 s double push, 83 s single push,
  same skater.
- **Result: 72.2% (36 windows), target not met.** Cut-off +0.06 (Mantia
  single push +0.11, double push 0.00). Thorup: double push 89-100% except
  the 0-60 kph clip (43%); single push 71%. Foot twist: 52.8%, about chance.

**Third and final attempt (10/24), `detect_combined.py`:** both measures
combined (logistic regression), trained on 213 windows from 7 skaters
(Mantia, Thorup, Swings, the three skaters in the styles short, junior 300 m).
Leave-one-skater-out preview 78.4% (Swings 8%). **New test** on a skater never
used anywhere (Lars Scheenstra; `doublepush/new_test_labels.csv`): inline double
push in slow motion 59% (22 windows), ice single push 100% (only 1 usable
window). **60.9% (23 windows), target not met.** Weights: feet crossing +1.69,
landing position -0.76 (standardised).

**Detection: closed (10/24).**

| Attempt | Test | Result (target 85%) |
|---|---|---|
| Feet crossing under the body | other skaters | **77.8%** |
| Where the recovering foot lands | fresh Thorup clips | 72.2% |
| Both combined, 7 training skaters | new skater | 60.9% |

The best pose-only detector identified double push correctly about 3 times in
4 on skaters it wasn't set on, well above chance, but short of the target and
weaker on newer skaters. Single-camera pose captures the foot coming under the
body but not edge changes, wheel pressure or the twist for the second push.
Clips offered after a result are used for the feedback report, not to
re-score detection.

**Two layers, decided 10/18:**
1. **Detection** (double push vs normal stride): foot crossing under the
   body. Needs more normal-stride examples.
2. **Quality** (good vs flawed double push): foot-path smoothness (S-curve
   vs jerky), hip up-down movement (pressure dropped and regained),
   upper-body movement relative to the hips (only when the upper body is in
   frame). Optional idea: wheel noise from the audio as a pressure proxy
   (tutorial music and voiceover make it hard).
- Define the parts of a double-push stride (outside-edge set-down, inward
  pull under the body, outward push, recovery) and how each looks on video.
- Label strides in the elite clips: start and end of each part, per foot.
- Also label **non-double-push** strides (a classic push straight out), so
  detection has something to tell apart.

### 7c: Double Push Detection
- Measures: the foot's sideways path relative to the hip midline over a
  stride (double push shows an inward movement under the body before the
  outward push), foot angle, knee and hip angles, timing.
- Evaluated on skaters or clips it was not built on.
- *Proposed target:* ≥85% of labelled strides correctly called double push
  or not.

### 7d: Technique Feedback

**Coaching feedback written (10/20), `doublepush/feedback_text.json`:** an
explanation, cue and drill for six issues, each tied to the measure that
flags it: no underpush (feet crossing), skate landing too wide (landing
position, to be added), jerky foot path (path roughness), hip bounce (hip
up-down movement), too much upper-body movement (shoulder sway, upper body
in frame), no fall and catch (hip shift before set-down, to be added).
**First feedback report (10/26), `doublepush_report.py`: test failed.**
Six measures per window (feet crossing, landing position, foot-path
roughness, hip bounce, upper-body sway, and fall and catch: hip shift toward
the landing foot in the 0.3 s before touchdown) compared with the elite
5th-95th percentile range of Mantia's and Thorup's good double push (109
windows); an issue is listed when flagged in at least half of a clip's
windows; slow-motion clips skip timing-based measures. Reports in
`reports/doublepush/`.

| Test (fixed before running) | Expected | Result |
|---|---|---|
| Mantia 3:03-3:09 | jerky path or hip bounce | nothing flagged |
| Mantia 3:15-3:23 | upper body (or can't judge) | nothing flagged |
| Lars Scheenstra | each measure ≤10% of windows | landing too wide 27%, no underpush 11% |
| Mark Horsten | each measure ≤10% of windows | landing too wide 55% |

- Mantia's flawed clips measure like his good ones; the described flaws are
  too subtle for these measures.
- Elite skaters differ: Horsten and Scheenstra land wider than Mantia and
  Thorup, so a two-skater reference treats another elite style as a flaw.
- Upper-body sway is unreliable when the shoulders are out of frame.
- Next (decision pending): a visual comparison tool (the skater's measures
  beside an elite skater's, coaching text as guidance, the skater or coach
  judges) or close Phase 7 with the write-up.

**Visual comparison tool (10/27), `doublepush_compare.py`:** since the
automatic judge failed, the tool shows instead of judging: one sheet with a
skater's clip beside an elite example (Thorup), with a "how to read this
sheet" box giving the key differences in numbers, a plain-language caption on
each row (still frames at touchdown, feet sideways paths, where the recovering
foot lands, hips up-down), labels on the plots, and the coaching text as
guidance. Usage: `python -m doublepush_compare --video <clip> --start 1:05
--end 1:20 --name "..."` (no arguments = the Thorup single vs double demo).
- Thorup single push vs double push (same camera): recovering foot lands
  under the hips (within 0.15 leg lengths) 47% vs 68%, out to the side
  (beyond 0.30) 33% vs 10%, a foot crossed under the body 37% vs 47% of the
  time; hips up-down no different (0.09 vs 0.10).
- Mantia double push (6:17-6:40) vs Thorup: 60% vs 68% under the hips, 3% vs
  10% out to the side, the same pattern.
- First version showed the data without saying what to look for; after
  review, captions, numbers and labels were added, and the sheet was judged
  clear and useful.

**Same-skater side-by-side example (10/26):** Thorup "Classic vs.
Doublepush test" 3:35-3:45 double push, 3:45-3:51 classic
(`doublepush/thorup_cvd_labels.csv`).

**Original plan items:**
- An elite double-push reference (3+ elite skaters) and per-stride measures
  a coach would use: how far the foot pulls under the body, when the
  outward push starts, knee bend through the stride.
- A report (or app mode) that compares a skater's strides against the
  reference and explains the differences in plain terms.

### 7e: Phase 7 Evaluation & Write-Up
Summary in the same format as Phases 5 and 6, including what didn't work.

---

## Side Track: Controlled Ice Footage & Fatigue (7s)
- A fixed side-on camera filming hard laps, with lap times (and heart rate
  if available) as the fatigue measure.
- Analysed with 3D trunk measures and side-on knee angles, early vs late,
  as in 5f.
- Depends on footage being filmed; anyone filmed besides the researcher
  agrees to it first.

## Future Work (not this phase)
- Multi-camera 3D.
- Inline corner/crossover mechanics and pack-race analysis beyond what the
  double-push work needs.

---

## Data Sources

**Downloading 10/18 (elite double-push technique videos, 1080p):**
- Joey Mantia, "Double Push Tutorial, Episode 1": https://www.youtube.com/watch?v=55YXOkf6n8Q
  (`data/doublepush_mantia_tutorial.mp4`). Useful: 0:00-1:20, 2:14-4:34,
  4:50-5:41, 6:17-8:30, all front view.
- Bart Swings, "Double Push skate technique" (Powerslide): https://www.youtube.com/watch?v=xTHvlyW6aBo
  (`data/doublepush_swings_powerslide.mp4`). Useful: nearly the whole video.
- Viktor Hald Thorup, "Learn to DOUBLEPUSH, Full guide": https://www.youtube.com/watch?v=b7YoP_JtFAs
  (`data/doublepush_thorup_guide.mp4`). Useful: 2:53-3:30 and 4:26-4:50
  (feet only), 4:51-5:19 (legs), 7:23 (full body), 8:06-8:29.

**Known limits of tutorial footage:** edited (cuts, close-ups of feet only,
possibly slow motion), so timing measures need checking against frame rate
and playback speed; a few skaters only.

**Downloaded 10/19 (normal strides):**
- Short, 200 m sprint (Skater Nikhil): https://www.youtube.com/shorts/uNLZGT4_Qk8 (`data/normal_nikhil_short.mp4`). Useful: 0:12-0:16.
- 2013 Championships VLog01, Junior Men 300 m TT (Powerslide): https://www.youtube.com/watch?v=1Mk-_2ABe_o (`data/normal_powerslide_junior300tt.mp4`). Useful: 0:24-0:26.
- World Skate 500 m Senior Men qualification (Oliver Grob): https://www.youtube.com/watch?v=PDTBZdUKoJs (section 0:30-0:40 only, `data/normal_worldskate_500m_0030-0040.mp4`). Useful: 0:34-0:36, three skaters.
- 2013 Championships VLog02, Men 300 m TT (Powerslide): https://www.youtube.com/watch?v=5fuKYhiGrYU (section 0:32-0:41 only, `data/normal_powerslide_men300tt_0032-0041.mp4`). Useful: 0:36-0:37.
- World Games 2017, Men 15,000 m Elimination final: https://www.youtube.com/watch?v=WwcyaBYl18o (`data/normal_worldgames2017_15000m_elim.mp4`). Normal strides, many skaters, some corners.
- Joey Mantia, "Skate Stroke Coordination & Timing, Inline Skating Casual Clinic": https://www.youtube.com/watch?v=CJTQWpfeIqQ (`data/normal_mantia_stroke_clinic.mp4`). Normal single push throughout, skated a bit upright (not full speed).
- Viktor Hald Thorup, "Double-Push Skating, 0-60 kph in 30 seconds": https://www.youtube.com/watch?v=WpTlBT-OjiI (`data/doublepush_thorup_0to60.mp4`, 1080p60). Fresh test, 0:04 to the end.
- Viktor Hald Thorup, "Change this and you will SKATE FASTER!": https://www.youtube.com/watch?v=XJ__iyVP-Ts (`data/thorup_skate_faster.mp4`). Fresh test, single and double push sections.
- Lars Scheenstra, "Techniek skeeleren double push, Slowmotion" (Catch Your Moment): https://www.youtube.com/watch?v=QwtzobwXFdk (`data/newtest_scheenstra_doublepush_slowmo.mp4`). Final detection test, 0:01-1:53.
- Lars Scheenstra and Wim Dillen (ice): https://www.youtube.com/watch?v=KZk9jJ7uQak (`data/newtest_scheenstra_ice_singlepush.mp4`, 480p). Final detection test, 0:16-0:49 (orange suit).
- Viktor Hald Thorup, "Classic vs. Doublepush test": https://www.youtube.com/watch?v=Lo8wya4pwaw (`data/thorup_classic_vs_doublepush.mp4`). For the feedback report; 6 candidate stretches (196 s) to be labelled.
- Mark Horsten, "Inline Skate Double Push Technique Slowmotion" (Catch Your Moment): https://www.youtube.com/watch?v=EmMf91AVUXc (`data/doublepush_slowmo_extra.mp4`). For the feedback report.
- Also 10/18: Viktor Thorup, styles short: https://www.youtube.com/shorts/ByLdwq7IGy8 (`data/doublepush_thorup_styles_short.mp4`).

## Real Bugs & Limitations Found This Phase (Running Log)

| Date | Finding | Status |
|---|---|---|
| 10/18 | yt-dlp left video and audio as separate files (ffmpeg not installed) | Video-only files renamed and used (audio not needed); `data/*.m4a` added to `.gitignore` |
| 10/18 | Pose not found (or wrong) on feet-only and legs-only close-ups | Documented: measurement needs full-body footage; close-ups kept as visual reference only |
| 10/18 | Foot sideways position scaled by per-frame hip width blew up (hip points nearly overlap in some frames) | Scaled by median torso length per section instead |
| 10/18 | Mantia's "wrong" clips are flawed double pushes, not normal strides; foot crossing can't separate them | Relabelled; detection and quality treated as separate layers |
| 10/19 | Small (leg under ~100 px), diagonal race footage: foot points jitter as much as the movement, and diagonal views turn forward travel into apparent sideways movement; normal strides read as double push | "Can't judge" rule: front or back view and leg at least ~150 px |
| 10/19 | Pose model marked shoulders visible when mostly out of frame | Upper-body sway needs an in-frame check, not the visibility score alone |
| 10/19 | Long videos downloaded whole for a few seconds of use | ffmpeg installed; only the needed sections are downloaded |
| 10/20 | View-angle check (hip width / leg length, sideways drift) did not separate diagonal from front views | Not adopted; skater size rule (leg ≥150 px) kept; camera angle checked by eye |
| 10/22 | Long measurement run stopped part-way with no error (one-off crash; the stretch ran fine alone) | Resume option added (`DP_RESUME=1` reuses saved per-frame data) |
| 10/22 | Detection by feet crossing under the body: 77.8% on other skaters (target 85%) | Not met; recovery-path measure next |
| 10/23 | In a front view the foot furthest forward appears lowest, so "lifted = higher in the picture" fails | Touchdown timed at each foot's lowest point below the hips |
| 10/23 | First fresh test set had only double push (a rule calling everything double push would pass) | Single-push sections from the same skater added before running |
| 10/24 | Second measure (landing position) 72.2% on fresh clips; foot twist at chance | Not met |
| 10/24 | Combined measure 60.9% on a new skater; preview showed one elite skater (Swings) reads as single push on both measures | Not met; detection closed and documented |
| 10/24 | A clip was offered after a result | Not used to re-score detection; used for the feedback report |
| 10/26 | Feedback report: Mantia's flawed double pushes measure like his good ones | Not flagged; test failed |
| 10/26 | Feedback report: other elite skaters (Horsten, Scheenstra) land wider than the Mantia/Thorup reference and get flagged | Two-skater reference treats style differences as flaws |
| 10/26 | Automatic scan of Thorup's comparison video found only sitting/talking or ice clips | Skating section taken from the researcher's timestamps, labelled on 5-frames-per-second zoomed sheets |
| 10/27 | Thorup's comparison section (3:35-3:51) too far from the camera to measure | Same-skater example taken from his "Skate Faster" video instead |
| 10/27 | First comparison sheet showed data without saying what to look for | Added a how-to-read box with numbers, row captions and plot labels |
