# Phase 9: Skating Phase Model & Pacing Database

> Plan drafted 11/9. A long-running phase built in rounds, not a single test.
> Targets marked *proposed* are to be confirmed before the first test. Field:
> long track ice, using the race broadcasts already downloaded in Phases 5-8
> plus new races added over time.

## Formal Research Questions

> **9A:** To what extent can one model, trained on hand labels gathered across
> Phases 4-9, label every frame of a long track race broadcast as start,
> straightaway or corner, and find each stroke set-down, accurately enough on
> races it has never seen to replace hand labelling?

> **9B:** How do elite long track skaters pace their races across distances,
> skaters and seasons, measured from lap times read from race broadcasts?

## Status (11/22): Pacing database has all 12 skaters of the Beijing 2022 5000 m (154 laps, 2 finishes missing, all checked). Label rules and model targets fixed. Pair 10 marked: Swings (start and 15 corners usable) and van der Poel (whole race). Identity confirmed per stretch (2,209 training frames). Bloemen (pair 09, round 1 test race) marked. Old labels audited (89.8% of frames agree). Round 1 run once: 98.8% of frames (target 80% met, but 92% of test frames were corners; straights 88.9%); boundaries too little data. Round 2 run once: 89.8% (average of corner and straight scores, target 85% met) but only 16 straight frames; boundaries too little data. Round 3 run once on Ghiotto (Calgary): pose model 89.3% (target met), straights 80.7%, corners 97.8%, boundaries 66.7% (target 85% not met); Teachable Machine 83.3%. Straight errors mostly in the second half of straights.

## Why This Phase
- **The same problem kept returning.** Automatic corner/straight detection
  reached 77% in Phase 5g and 81.7% in Phase 6d, and in Phase 8 corner
  crossovers inside stretches labelled "straight" were the main source of
  wrong stroke rates. A model built for this one job, with frame-precise
  labels, addresses all three.
- **Broadcast lap times were the most reliable result of Phase 8.** All 78 lap
  times checked out, and they showed pacing differences clearly. Many more
  races are available.
- **Both grow with time.** Each labelled race makes 9A better and gives it a
  new test; each new race adds to 9B. Neither has a natural end point.
- The idea for 9A comes from the Phase 8 Tier 2 note (stroke phase classifier,
  Teachable Machine).

## What It Combines
| From | What it brings |
|---|---|
| Phase 4 | start labels (start rest, start acceleration) |
| Phase 5 | 176 corner and straightaway segments on 8 races (`labeled_segments.csv`), camera shot labels |
| Phase 6 | piece labels with phase and shot on 5 races, camera-cut detection, two-skater tracking |
| Phase 8 | 133 race stretches labelled, 37 frame-precise straight marks, 21 clips of hand-marked set-downs, 78 lap times, the marking player, the lap-time reader |

Older labels mark segments by eye, not exact frames, so they are used for
training only after a check, and frame-precise labels are made for testing.

## How Rounds Work (9A)
1. **Label a new race** (or a skater in one) with frame-precise boundaries:
   corner entry, corner exit, start, finish, and set-downs on chosen stretches.
2. **Test first:** the current model labels that race before it has seen any of
   it. The score is recorded against the target fixed beforehand.
3. **Then train on it:** the race joins the training set for the next round.

Every race is a fresh test once, so results are never scored on data the model
was trained on, and the running log shows whether the model improves as data
grows.

## Tier 1: Core

### 9a: Labelling Tool and Label Audit
- A full-race mode of the marking player: mark each corner entry and exit, the
  start and the finish, with the lap shown, so a whole race can be labelled in
  one pass.
- Audit the Phase 5, 6 and 8 labels: which are usable for training as they are,
  and how far their boundaries are from frame-precise marks on a sample.
- **Progress (11/12):** `mark_phases.py` built (start, entry, exit, finish keys;
  upper-case for boundaries not seen on screen; race clock and lap on screen;
  streams the video and keeps the last 10 s). Marks in `phase9/phase_marks.csv`.
  Corners lasted 6.3-8.3 s and straights 6.7-8.1 s. (These marks were first
  saved as van der Poel; on 11/13 they were found to follow Swings and
  relabelled.)
- 11/13: Swings marked through lap 10; marks after lap 8 (frame 186826)
  flagged unclear (Swings mostly off camera), so the start and 15 corners are
  usable. Each pair's race cut to its own file (`phase9/clips/`, not in git)
  so the player opens in seconds; review spreadsheet and frame check sheets
  (`phase_check_sheet.py`) used to confirm marks.
- **Whose pose the model sees:** the Phase 8 race pose follows whichever
  skater the pose model picks, often the one nearer the camera. The model
  needs the labelled skater's pose, so pose will be taken per skater from the
  Phase 6 tracks, with suit colour telling the two skaters apart.
- 11/14: van der Poel marked for the whole race (start, 25 corners, finish;
  marked finish 0.19 s from the official time). `skater_pose.py` built; after
  one fix about 6 of 55 sampled boxes are on officials (skater small or
  hidden); researcher's check pending.
- **Old-label audit (11/14, `label_audit.py`):** of 29 Phase 6 pieces for pair
  10, 23 agree exactly with the marks and 89.8% of frames agree; errors are at
  corner boundaries (10-37 frames) and two pieces are badly wrong. Decision:
  train on frame-precise marks only for now; keep a sample of old pieces to
  check by eye later; old labels never used for testing; trimmed old labels
  may be tried in a later round and kept only if the unseen-race score rises.
- **Size rule (decided 11/15):** the phase model keeps the 150 px rule
  ("normal size" in the targets), so it learns from close-ups, where pose and
  identity are reliable. Identity was poor mostly in wide shots.
- **Identity by stretch (11/15):** colour alone took the lap-counter official
  for Swings and, with the boards in the box, others for van der Poel. Each
  stretch of the same tracked person in the usable frames is confirmed by the
  researcher from one picture (`skater_pose.py --runs`). Pair 10: 14 of 77
  stretches wrong (13 of 59 van der Poel, 1 of 18 Swings); 2,209 frames kept
  (Swings 397, van der Poel 1,812).
- **Bloemen marked (11/15):** whole race, start, 25 corners and finish, all
  seen; marked finish 0.15 s from the official time; corners lengthen from
  6.5-8.0 s to 8.6-9.1 s in laps 10-12 as his laps slow.
- **Corner rule, decided 11/13:** a corner runs from the first crossover
  (entry) to the last crossover (exit). On the first corner after the start,
  skaters take some straight strokes between crossovers (researcher's
  observation); these are marked inside the corner (`in_straight` where
  straight strokes begin, `in_cross` where crossovers begin again), so the
  model can learn them as their own label (option B, chosen over counting
  them as corner).
- **Round rules:** the van der Poel race is training data. Round 1 tests on a
  pair with no training labels; later rounds test on other broadcasts
  (Inzell, Calgary, Stavanger), which the model has never seen at all.

### 9b: Phase Model
- Input: pose over a short window of frames (feet, hips, leg length, as in
  Phases 6-8), plus camera-cut and shot information. A simple model first;
  something larger only if it beats the simple one on unseen races.
- Output per frame: start, straight, corner, or "can't tell" (skater too small,
  not live, not the skater).
- **Targets, fixed 11/13 before any model was built:**
  - Round 1: at least **80%** of frames correct on an unseen race, normal-size
    skaters only. The frame target may be raised (for example to 85%) before a
    later round's test, never after seeing that round's result.
  - At least **85%** of corner entries and exits found within **0.5 s** (15
    frames) of the hand mark, so boundaries are timed right, not just most
    frames.
  - **Head-to-head (comparison, not pass/fail):** the Phase 6d detector run on
    the same test race and scored the same way. Its 81.7% was scored on whole
    pieces, so it is not directly comparable until rerun.

**Round 1 setup (written 11/16, before any model was trained):**
- **Training data:** pair 10 (Swings laps 1-8, van der Poel whole race),
  only frames inside the skater's marked race, race clock on screen, leg at
  least 150 px, a steady track (15+ frames), and in a stretch the researcher
  confirmed as the right skater: 2,209 frames.
- **Test data:** pair 09, Bloemen, the same frame rules and the same
  stretch-by-stretch identity check by the researcher, done before the model
  is run on it. The model is run on it once; nothing is changed afterwards
  and rerun.
- **Labels:** from the marks, corner (entry to exit, straight strokes inside a
  corner counted as corner) or straight.
- **Inputs, per frame, from the skater's own pose, over a window of 31 frames
  (about 1 s, centred):** each foot's sideways position from the hips and
  height, in leg lengths; the gap between the feet; which foot is further
  left (feet crossing); the hips' movement from frame to frame in leg lengths.
  Summarised over the window as mean, spread, minimum, maximum, and the share
  of frames with the feet crossed. A frame is used only if at least 10 frames
  of its window have pose.
- **Model:** a gradient-boosted tree classifier (scikit-learn
  HistGradientBoostingClassifier, default settings, classes weighted
  equally), then a 15-frame majority filter within each stretch. Any choice
  between settings is made by training on one pair 10 skater and testing on
  the other, never on pair 09.
- **Scoring:**
  - Frames: share of test frames labelled correctly; target 80% (11/13).
  - Boundaries: a marked entry or exit is scorable if the test frames cover
    at least 80% of the 15 frames before and after it; it is found if the
    predicted label changes the right way within 15 frames (0.5 s) of the
    mark. Target 85% (11/13). With fewer than 10 scorable boundaries the
    result is reported as too little data, not as pass or fail.
  - Comparison: the Phase 6d detector, scored on the same frames.
- **Pair 10 swap test (11/16, before round 1; nothing changed after it):**
  trained on van der Poel, 82.9% of Swings' frames correct; trained on Swings
  (397 frames), 69.5% of van der Poel's. Only 2-3 boundaries per skater were
  scorable, so the boundary target may report too little data in round 1.
- **Torso colour tested against the researcher's stretch check (11/16):** it
  rejects 11 of the 14 wrong stretches (frames on the wrong person: 301 down
  to 98) and keeps 55 of the 63 right ones. Better, but not clean, so the
  researcher's stretch check stays for every race.

**Round 1 result (11/17, run once):** trained on pair 10 (2,042 frames),
tested on Bloemen (1,124 frames, identity confirmed on all 51 stretches).
- Frames: **98.8%** correct, target 80% **met**. The test was lopsided: 92%
  of frames were corners (close-ups are mostly in corners), so always
  answering corner would score 92%. Corner frames 99.6% (1,030 of 1,034),
  straight frames 88.9% (80 of 90, from 8 straights).
- Boundaries: **0 scorable**, reported as **too little data**. No entry or
  exit had close-ups on both sides.
- Phase 6d comparison: not yet run (it scores pieces, to be adapted).

**Round 2 rules (fixed 11/17, before round 2):**
- Frames are scored as the **average of the corner and straight scores**
  (balanced), so always answering one label scores 50%. Target **85%**
  (raised from 80% before this round, as allowed).
- A boundary is also scorable with **only one side covered** (at least 80% of
  the 15 frames on that side); it is found if no frame on the covered side
  more than 15 frames from the mark has the wrong label. Two-sided boundaries
  are scored as in round 1. Target unchanged (85%), and fewer than 10
  scorable boundaries is still reported as too little data.
- Training: pair 10 and Bloemen (pair 09). Test: pair 06 (Bergsma, orange;
  Ichinohe, white), marked and identity-checked before the model is run.
- **Labels between marks that do not fit together (11/18):** an exit after an
  exit (entry off camera, not marked) or an entry after an entry leaves the
  frames between them unlabelled instead of guessed. Only Ichinohe's labels
  changed; the training skaters' labels are identical, so round 1 stands.
- **Open before round 2 (11/18):** pair 06 has few usable test frames (Bergsma
  168, Ichinohe 62) because Bergsma's identity flickers near the colour
  cutoff and every miss restarts the 15-frame steady-track count. Allowing
  gaps of up to 5 frames would give Bergsma 697 and Bloemen 1,635 (from
  1,174). To be decided by the researcher before round 2 is run.
- **Decided 11/19, before round 2:** a stretch continues through misses of up
  to 5 frames (1/6 s) when the same tracked person is found again; the missed
  frames are not used, and a stretch needs 15 frames where the skater was
  actually found. Stretch sheets for pair 06 and pair 09 are remade under this
  rule and checked again by the researcher; round 1's stretches are kept as
  its record.

**Round 2 result (11/19, run once):** trained on pair 10 and Bloemen (3,552
frames), tested on Bergsma (657 frames; all 33 stretches confirmed). Ichinohe
had no test data: all 9 of his stretches had the box on Bergsma.
- Frames: average of corner and straight scores **89.8%**, target 85% **met**.
  Corners 98.3% (630 of 641), straights 81.2% (13 of **16**). The straight half
  rests on about half a second of one straight, so straight detection is
  not really tested.
- Boundaries: 3 scorable (all one-sided), all found; **too little data**.
- **Across rounds 1 and 2:** corners 98-99% on two unseen skaters. Close-ups
  on this broadcast are almost all in corners (1,034 vs 90 and 641 vs 16
  frames), so straights and boundaries cannot be tested well at the 150 px
  size rule.
- **Round 3 direction (decided 11/19):** check whether another broadcast films
  straights closer before changing the size rule or accepting the limit.

### 9c: Set-Down Detection Inside the Model
- Re-test the Phase 8 counter only inside frames the model calls straight.
- *Proposed target:* the 8c target (85% of set-downs within 0.1 s), now on race
  stretches, not just clean clips.

### 9d: Pacing Database
- Read lap times for both skaters in each pair, race by race, using the Phase 8
  reader adapted to each broadcast's graphic. Each race checked against the
  final time and cumulative splits, with a spot-check by the researcher.
- One table: race, distance, season, skater, lap, lap time, final time, place.
- Start with the races already downloaded (Beijing 2022 5000 m opponents, Inzell
  2026 5000 m, Calgary and Stavanger, Tomaszow 2026), then add races by distance.
- **Progress:** `pacing/lap_times.csv`. 11/10: Bart Swings (Beijing 2022, pair
  10), 13 laps, every lap agrees with the cumulative times, checked against the
  sheet by the researcher. The Phase 8 lap sheets show both skaters, so the
  other Beijing opponents need only the second row read.
- 11/12: the other five opponents read and checked. The finish graphic was
  missed by the Phase 8 detector (its rank box moves left); `finish_graphics.py`
  recovered three finishes. Thorup's and Giovannini's finishes were not shown on
  the broadcast. All six original skaters' finishes on it matched Phase 8.
- First look (description only): in 5 of 6 pairs the faster finisher faded less
  (median laps 9-12 minus laps 2-5); Bloemen faded most of all 12 (+2.18 s) but
  held the lead over Zakharov (+1.34 s).

### 9e: Pacing Questions (fixed before the analysis)
1. How does the shape of a race (even, fade, negative split) differ between
   3000 m, 5000 m and 10000 m?
2. Do faster finishers pace more evenly than slower ones in the same race?
3. Where in a race does a fade usually start, and how large is it?
- Bars and how many races each question needs are set before the analysis
  starts, as in Phase 8.

### 9f: Combining Both
- Once 9b and 9c pass, stroke rate per lap from the model joins the lap times,
  which reopens the Phase 8 questions on a stronger base.

### Round 3 focus: straights (11/20)
- **Test race:** Calgary 2025 10,000 m, Ghiotto (blue, Italy) vs Eitrem (red,
  Norway). Chosen from scouting sheets of four downloaded races: 62% of
  sampled frames were close-ups, and the researcher judged about half of
  those to be straights (Beijing close-ups were almost all corners).
  Calgary is 25 fps and 1080p, so frame counts (0.5 s, window, filter) and
  the size rule (150 px at 720p = 225 px at 1080p) must be converted in the
  round 3 rules.
- **Researcher's observations for telling straights from corners:** on
  straights the feet do not cross, and the more one foot crosses the other,
  the more likely it is a corner; crossovers come with much more lean; on
  straights a lean comes with the other leg pushing out wide, while in
  corners both legs stay close during the lean.
- **Measures from these, to try before round 3:** how often the feet swap
  sides in the window; body lean (hips to shoulders); lean together with
  the gap between the feet. Shoulder and knee points are added to the
  existing pose files with `augment_pose.py` (same boxes, so the identity
  checks stand). Whether to keep them is decided only on the Beijing data
  (swap tests and rounds 1-2 data), never on Calgary.
- **Cue test (11/21), four Beijing skaters, each left out in turn:** straights
  79.8% without the cues and 78.0% with them, corners 96.1% and 95.1%;
  mixed per skater. **Not used in round 3.** Likely reason: lean and crossing
  in a 2D picture depend on the camera angle.
- **Calgary frame rules (11/21):** its own live check (the blue race-time box,
  99.8% of Ghiotto's race) instead of Beijing's clock graphic, which read 8%;
  size rule scaled by video height (225 px at 1080p). Ghiotto: all 164
  stretches confirmed, 6,748 frames (4,161 corner, 2,379 straight).
- **Teachable Machine pictures (11/21, `label_tm.py`):** 1,822-picture pool
  from the four Beijing skaters; 270 labelled by the researcher (113 corner,
  93 straight, 64 skipped). The researcher's eye agrees with the marks on
  92.7% of them.

**Round 3 rules (fixed 11/22, before anything is run on Calgary):**
- **Test:** Ghiotto, Calgary 2025 10,000 m, all 6,748 confirmed frames (marked
  race, Calgary live check, leg at least 225 px at 1080p, stretches through
  misses of up to 5 frames, all 164 stretches confirmed). Not hand-picked.
  Each model is run on it once; nothing is changed afterwards and rerun.
- **Model A, pose model:** as in round 2, trained on the four Beijing
  skaters (Swings, van der Poel, Bloemen, Bergsma; confirmed frames, misses of
  up to 5 frames bridged), without the researcher's cue measures (they did
  not help on Beijing, 11/21).
- **Model B, Teachable Machine:** the researcher's model, trained once in the
  browser with default settings on 258 corner and 267 straight pictures that
  the researcher chose and labelled from the four Beijing skaters
  (`label_tm.py`). Run on a picture of each test frame cut out the same way
  (the skater's box with 15% padding; Teachable Machine then takes a centred
  square at 224 x 224).
- **Same for both:** labels from the marks (straight strokes inside a corner
  count as corner); a majority filter of 0.5 s within each stretch; both
  scored on exactly the same frames.
- **25 fps conversions:** times are kept in seconds. Pose window 1 s (25
  frames, at least 8 with pose), majority filter 0.5 s (13 frames), boundary
  tolerance 0.5 s (13 frames); per-frame hip movement is scaled to the 30 fps
  rate the pose model was trained on.
- **Scoring:** as round 2. Frames: average of the corner and straight scores,
  target 85%; the straight score is reported on its own as the main question
  of this round. Boundaries: two-sided and one-sided, target 85%, fewer than
  10 scorable reported as too little data.

**Round 3 result (11/22, run once):** Ghiotto, 6,327 frames (4,071 corner,
2,256 straight).

| | Pose model | Teachable Machine |
|---|---|---|
| Corners | 97.8% | 91.8% |
| Straights | 80.7% | 74.8% |
| Average (target 85%) | **89.3%, met** | 83.3%, not met |
| Boundaries within 0.5 s (target 85%) | 20 of 30 (66.7%), not met | 18 of 30 (60.0%), not met |

- First test with real straights; the pose model carries over to a new
  broadcast (arena, cameras, 25 fps).
- Straight errors by time since the corner exit: 91% (0-1 s), 95% (1-2 s),
  91% (2-3 s), **74% (3+ s)**; 52% within 0.5 s of a boundary. The model calls
  about 1 in 4 late-straight frames a corner. The researcher's view: late
  straight strokes look similar to a corner with much less lean; camera
  changes are not the main cause.
- **Next test on a new race:** changes made after seeing this result are not
  tested on Calgary again; round 4 uses a race not yet used, and Calgary can
  join training.
- **Feet visibility (11/22, analysis of the same model):** late straights with
  the feet barely visible (below 0.3) were right 50% of the time (378
  frames), against about 83% with the feet visible; early straights 81%
  against 95%. The researcher noticed the camera zooms in on the upper body or
  cuts to officials late in straights. Proposed for round 4 (to decide before
  it runs): such frames reported as "can't judge", not scored.
- **Round 4 test race (set up 11/22):** Inzell 2026 5000 m, Eitrem (red,
  Norway) vs Jilek (white, Czech Republic); Phase 6 tracks reused; own live
  check (frames 310-9,886); Eitrem not yet marked.

### 9g: Teachable Machine for Skater Identity (added 11/20)
- **Why:** identity by suit colour was the biggest practical problem in rounds
  1 and 2 (officials in yellow, staff in orange, Bergsma taken as Ichinohe);
  every round needed the researcher's stretch-by-stretch check.
- **What:** an image classifier made in Google Teachable Machine that tells
  whether a tracked box is a racing skater or someone else (official, coach,
  staff, spectator). Training images come from the stretches the researcher
  already checked (ok / wrong), cut out of the video into folders; the
  researcher trains it in the browser and exports the model, which then
  filters boxes before the stretch check.
- **Test (fixed before use):** trained on boxes from some races, tested on a
  race it has not seen, against the researcher's stretch check there. It is
  only adopted if it removes wrong boxes without dropping many right ones; the
  stretch check stays until it has passed.
- The researcher's original Teachable Machine idea for corner vs straight
  (Tier 2) remains as a comparison with the pose model after round 3.

## Tier 2: If Time Allows
- **Image-based comparison:** a Teachable Machine style model on frames instead
  of pose, tested on the same unseen races.
- **Finer labels:** crossover strokes in corners, start phases, push vs glide.
- **Other footage:** inline races, if the model is useful there.
- **A pacing sheet for any race** from the database, in the 8e style.

## Considerations
- **Labelling is the main cost.** A full 5000 m is about 12 laps, so about 50
  corner boundaries per skater; labelling should be planned in small pieces.
- **Each broadcast's split graphic differs,** so the lap-time reader may need a
  small change per broadcaster.
- **Skater identity** in pairs needs care (Phase 6 tracking, Phase 8 lessons:
  skaters in similar suits).

## Risks
- Broadcast camera limits remain: small skaters, cuts, replays.
- Older labels may be too rough to help training.
- Pacing data shows what skaters did, not why.
- Official lap times may exist for some races; if found, they are used to check
  broadcast readings, not mixed in unmarked.

## Real Bugs & Limitations Found This Phase (Running Log)

| Date | Issue | Fix |
|---|---|---|
| 11/12 | Phase 8 lap-sheet detector missed the finish graphic (rank box at the left edge) | `finish_graphics.py` scans for it after each pair's last split |
| 11/12 | Lehman's first name written as Ethan in planning; the graphic shows Emery | Corrected in the database |
| 11/12 | Phase player's undo removed the latest frame, not the last mark pressed | Undo follows key order; fixed before use |
| 11/13 | Marks saved under van der Poel followed Swings (the skater near the camera) | Frame sheet showed it; researcher confirmed; relabelled as Swings |
| 11/13 | A correction to the marks file was overwritten when the open player saved on quit | Marks file edited only with the player closed |
| 11/13 | Undo pressed before any new mark removed an earlier session's mark | Undo limited to the current session |
| 11/13 | Player took minutes to open a late pair (reads the broadcast from frame 0) | Each pair's race cut to its own file |
| 11/13 | Swings' marks from lap 8 irregular (corners of 1.8 and 2.7 s) because the camera followed van der Poel | Flagged unclear, not used |
| 11/14 | Per-skater identity missed Swings (bent forward, box includes ice) and picked up officials in blue and yellow | Upper half of the box, lower cutoff, largest box of the colour; ~6 of 55 sampled boxes still on officials |
| 11/14 | Phase 6 corner/straight labels wrong at boundaries on ~10% of frames, two pieces badly wrong | Not used for testing; marks only for training for now |
| 11/15 | Identity sheet: 7 of 43 sampled frames on the wrong person, nearly all wide shots | 150 px size rule kept for the model |
| 11/15 | Lap-counter official (yellow jacket, black trousers) taken as Swings; boards behind skaters made others look like van der Poel's blue | Identity confirmed per stretch by the researcher; 14 of 77 stretches dropped |
| 11/15 | Planned crouched-vs-upright check could not tell skaters from officials | Dropped |
| 11/16 | Pair 09 tracking stopped when the session ended, before saving | Long jobs run as separate processes |
| 11/16 | Colour read from the box's upper half included the blue boards | Colour read only inside the pose's torso (shoulders to hips) |
| 11/18 | Bergsma corner entry keyed as a start (s next to e) | Changed to entry |
| 11/18 | Ichinohe exits with no entry (entry off camera) would have made a corner count as straight | Frames between marks that do not fit together left unlabelled |
| 11/18 | Two Ichinohe entries marked when he came back into view mid-corner | Flagged unclear, not used |
| 11/18 | Bergsma's orange reads as hue 3-4 on this broadcast, outside the orange range | Range widened to 0-17; Bergsma found in 25% of frames, up from 13% |
| 11/19 | Bergsma's identity flickered near the colour cutoff, so most stretches never reached 15 frames in a row | Stretches continue through misses of up to 5 frames |
| 11/19 | Ichinohe's white colour check picked up Bergsma in all 9 stretches | Ichinohe excluded; no test data |
| 11/19 | Close-ups almost all in corners, so straights and boundaries barely appear in tests | Look for broadcasts that film straights closer |
| 11/20 | Tools fixed to the Beijing video (windows, race clock, 30 fps) | Race registry: video, window, fps and race name per race |
| 11/20 | Ghiotto: two entries in a row (5727, 6107) | Frames between left unlabelled |
| 11/21 | Beijing race-clock check read only 8% of Calgary as live | Calgary check from its own race-time box; set per race |
| 11/21 | Size rule in pixels set on 720p video | Scaled by video height (225 px at 1080p) |
| 11/21 | The researcher's straight cues lowered straights on Beijing (79.8% to 78.0%) | Not used; may suit a picture-based model better |
| 11/22 | Teachable Machine model could not run without input-scaling metadata | `ai-edge-litert` runtime installed (with permission), scaling done in code |
| 11/22 | Straights in the second half of a straight called corner about 1 in 4 times | Teachable Machine pictures to include late straights; a new race for round 4 |

## Rounds (Running Log)

| Round | Date | Test race | Frames correct | Boundaries within 0.5 s | Training races after |
|---|---|---|---|---|---|
