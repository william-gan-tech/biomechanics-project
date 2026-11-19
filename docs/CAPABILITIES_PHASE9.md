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

## Status (11/18): Pacing database has all 12 skaters of the Beijing 2022 5000 m (154 laps, 2 finishes missing, all checked). Label rules and model targets fixed. Pair 10 marked: Swings (start and 15 corners usable) and van der Poel (whole race). Identity confirmed per stretch (2,209 training frames). Bloemen (pair 09, round 1 test race) marked. Old labels audited (89.8% of frames agree). Round 1 run once: 98.8% of frames (target 80% met, but 92% of test frames were corners; straights 88.9%); boundaries too little data. Round 2 rules fixed; test pair 06 marked (Bergsma whole race, Ichinohe to about lap 9), identity check pending.

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

## Rounds (Running Log)

| Round | Date | Test race | Frames correct | Boundaries within 0.5 s | Training races after |
|---|---|---|---|---|---|
