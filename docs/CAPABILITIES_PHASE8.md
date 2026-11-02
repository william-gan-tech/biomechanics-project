# Phase 8: Stroke Rate & Pacing in Long Track Races

> Plan drafted 10/28. Targets marked *proposed* are to be confirmed before
> testing. Field: long track ice, using the whole-race broadcasts already
> downloaded and scanned in Phases 5-6.

## Formal Research Question

> To what extent can single-camera pose estimation count a long track
> skater's strokes reliably enough to measure **stroke rate and stroke
> length lap by lap**, and how do they change across a race and between
> faster and slower skaters?

## Status (11/1): 8b hand marks done (21 clips). 8c stroke counter tested once: 84.2% of set-downs found within 0.1 s on normal-size test clips (target 85% not met, by one set-down; extras 10.5%, within the limit); small skaters 50%. 8a lap times read from the broadcast for six skaters (78 laps, checked). 8d done: all 133 race stretches labelled; no general claim (3 of 6 skaters slower late, 2 faster); Bloemen past the 10% bar; corner-exit crossovers found inside "straight" stretches. Next: 8e pacing sheets.

## Why This Phase

- Inline double push data was limited (few skaters, edited technique
  videos). Ice race broadcasts are plentiful, and six whole 5000 m and
  10000 m races are already downloaded, scanned and mapped to pairs
  (`race_scan_*`).
- Stroke rate is a **timing** measure. Phase 7 found that timing and
  position measures hold up on single-camera footage, while joint angles
  and fine detail do not.
- **Speed comes from official lap times**, which are public for every
  World Cup and championship race. Stroke length = speed / stroke rate, so
  pose only has to count strokes; it never has to measure distance.
- It approaches the fatigue question from Phases 1-6 from a new direction:
  instead of "does form look different late in the race", it asks "does
  stroke rate or stroke length fall first when lap times slow".
- Coaches already use strokes per straightaway, so the result is easy for
  skaters to use.

## How This Differs From 5d (Stride Rhythm)

Phase 5d measured stride rhythm (how even strides are) from knee-angle
troughs and found it weak on broadcast footage: segments were short and
broken by camera cuts, so evenness was noisy. Phase 8 is different:
- It **counts** strokes (a number per straightaway or per lap), which is
  much less sensitive to short segments than an evenness score.
- It uses the Phase 7 touchdown signal (foot lowest point below the hips)
  and the side-to-side hip and foot movement, not knee angle, with knee
  angle only as a fallback.
- It uses Phase 6 pure pieces (camera cuts and overlaps handled) and the
  6d corner/straight detection to place each count on a known part of the
  lap.
- Every automatic count is checked against counts made by eye.

## Lessons Carried Over
- Every target is fixed before testing; results are reported whether they
  succeed or not.
- Ground truth (stroke counts by eye) is made before the automatic counts
  are seen, on clips chosen in advance.
- Camera view is recorded for every clip; small or distant skaters (leg
  under ~150 px) are reported as "can't count", not guessed.
- Clips offered after a result are not used to re-score it.

---

## Tier 1: Core

### 8a: Data and Lap Times
- List the six downloaded races with their pairs (`race_scan_*/pairs_map.md`)
  and collect the official lap times (every 400 m) for each skater from the
  published results.
- Match each lap in the video to its official lap (lap 1, 2, ...), using
  the broadcast lap counter or crossing the finish line.
- Output: `stroke_rate/lap_times.csv` (race, skater, lap, lap time).

**Result (10/30):** the Beijing broadcast shows a split graphic every 400 m
(cumulative time and lap time per skater). `lap_graphics.py` finds the
graphic along the bottom of the frame and builds one sheet per pair
(`stroke_rate/lap_sheets/`); all 78 lap times for six skaters (van der
Poel, Roest, Bergsma, Kramer, Bloemen, Rijhnen) were read from them.
- Check: every lap time equals the cumulative time minus the previous
  split (78 of 78).
- Three splits were not shown for the skater (Roest lap 7, Kramer laps 1
  and 10); each is derived exactly from the next split (cumulative minus
  lap time). Final laps come from the official finishing times.
- Spot-check by the researcher against the Bergsma sheet (four splits):
  all matched.
- Pacing differs widely: van der Poel even to the finish (29.0 s laps,
  28.97 last lap); Bloemen fast early (28.8) and 31.8 by lap 12; Kramer
  a steady fade (29.3 to 30.8).

### 8b: Stroke Counting Ground Truth
- Pick test straightaways in advance (several skaters, early and late laps,
  front, side and back views), before any automatic counting is run.
- The researcher counts strokes by eye on each, using frame sheets or
  slowed clips: number of pushes, and the frame of each push.
- Output: `stroke_rate/hand_counts.csv`.

**Set up (10/28):** lap times will be read from the broadcast later (the
race clock is on screen); stroke counting goes first since it does not need
them. Test clips: the 21 Beijing 2022 5000 m straightaways already called by
hand in Phase 5 (`labeled_segments.csv`) that are at least 2.4 s long,
split by skater before any counting:
- tune (11 clips): Kramer, Roest. The counter may be adjusted on these.
- test (10 clips): Rijhnen, Bergsma, Bloemen. Never used to adjust it.

`stroke_clips.py` makes a 4x slowed clip of each (`stroke_rate/clips/`,
not committed, regenerated by the script) with the video frame number
burned in. For each clip the researcher writes down the frame of every
skate set-down (the new skate first touching the ice).

**Change before testing:** broadcast segments show 2.4-5.2 s of a
straightaway, never a whole one, so "strokes per straightaway" can't be
measured. The target is set on set-down timing instead (below).

### 8c: Automatic Stroke Counter
- Count pushes on each pure piece from the touchdown signal; place each on
  a corner or straight with 6d.
- Report stroke rate (strokes per minute) on straights and in corners
  separately, and "can't count" where the view is too small or too broken.
- *Target (set 10/28, before testing):* on the 10 test clips, the counter
  finds **at least 85%** of the hand-marked set-downs within 3 frames
  (0.1 s), with extra (unmatched) set-downs no more than 15% of the hand
  count.
- **Pose tracking (10/29), before any counting:** `stroke_track.py` ran the
  Phase 7 tracker on all 21 clips; pose was found in nearly every frame
  (all but tune_14: 68 of 91). Five clips fall under the Phase 7 size rule
  (leg under 150 px): tune_03, tune_10, test_08, test_15, test_19. Decided
  before counting: the target is judged on the 7 test clips at or above
  150 px; the 3 small test clips are scored and reported separately, not
  counted toward the target.

**Hand marks (10/29):** all 21 clips marked with `mark_setdowns.py` (a
player that records the frame on a key press). A first pass was redone
from scratch after a check of the gaps between marks found double presses.
Most straightaways end on a set-down (the step into the corner), which
was confirmed on rewatching.

**Counter (10/29), `stroke_counter.py`:** compared on the tune clips only:
foot height (the Phase 7 touchdown signal), feet closest together,
foot-to-foot distance, and feet furthest apart sideways. The furthest-apart
moment (end of the push) matched best, with the set-down about 7 frames
(0.23 s) later, and was stable across settings. Foot height, the Phase 7
signal, found only about 40% (broadcast cameras are mostly side-on).
Scoring rule fixed before the test: only hand marks the counter can see
both sides of are scored (from 10 frames after the clip start to 3 before
its end). Tune clips: 81.2% found, 12.5% extra (normal size).

**Result (10/29), one run on the test clips:**

| Test clips | Set-downs found within 0.1 s | Extra |
|---|---|---|
| Normal size (7 clips, Rijhnen, Bergsma, Bloemen) | **16 of 19 = 84.2%** | 10.5% |
| Small, leg under 150 px (3 clips, reported separately) | 5 of 10 = 50.0% | 50.0% |

**Target (85%) not met**, by one set-down. Of the 3 misses, two were found
4-5 frames from the hand mark (just outside the 3-frame window) and one was
not detected. The stroke **count** matched the hand count on 6 of the 7
normal-size clips (not the target; recorded as a description). Small
skaters are not countable reliably, as expected from the size rule.

### 8d: Stroke Rate and Length Across a Race
- For each skater with enough countable laps: stroke rate and stroke length
  (lap speed / stroke rate) lap by lap.
- Questions, fixed before looking:
  1. Does stroke rate rise, fall or stay level across a 5000 m / 10000 m?
  2. When lap times slow late in the race, is it mostly a drop in stroke
     rate or in stroke length?
  3. Within a pair, does the faster skater differ in stroke rate, stroke
     length, or both?
- *Proposed bar:* a change is only reported as real if it is larger than
  the counter's own error from 8c and shows in most skaters, not one.

**Rules fixed 10/31, before any race stroke rates were computed:**
- Stroke rate of a stretch = 60 x fps / median gap between the counter's
  set-downs (strokes per minute); needs at least 3 set-downs (2 gaps).
  The median keeps one missed set-down from halving the rate.
- Counter error on stroke rate, from the 8c clips (normal size, hand vs
  counter, per clip): median 4.5% on test clips (one clip 50% off, from a
  missed set-down), 9.4% on tune clips. The bar is set at **10%**.
- A skater's change is real only if their median stroke rate in late laps
  (9-12) differs from early laps (2-5) by more than 10%. A general claim
  ("skaters slow their stroke rate late") needs the same direction in at
  least 4 of the 6 skaters.
- Stroke length per lap = lap speed (400 m / lap time) / (stroke rate / 60),
  using the lap's straightaway stroke rate; reported per lap group, with
  the same 10% bar.
- Only stretches confirmed by the researcher as one named skater on a
  straightaway are used; laps are assigned from the race clock.

**Result (11/1).** All 133 stretches labelled. Five stretches with rates
above 100/min were hand-checked after the result was seen
(`stroke_rate/race_checks.csv`); hand marks follow the same 3-set-down
minimum. Both versions are reported (`stroke_rate/skater_summary.csv`).

| Skater | Stretches | Early → late, counter (spm) | Change | Hand-checked change | Past 10%? |
|---|---|---|---|---|---|
| van der Poel | 11 | 81.8 → 78.2 | -4.5% | -11.0% (late 72.8) | hand-checked only |
| Roest | 5 | 66.1 → 71.4 | +7.9% | same | no |
| Kramer | 8 | 64.3 → 66.7 | +3.7% | same | no |
| Rijhnen | 4 | 76.6 → 72.0 (1 stretch each) | -6.0% | same | no |
| Bergsma | 2 | no early stretch | - | - | can't measure |
| Bloemen | 7 (6 checked) | 101.9 → 70.6 | -30.7% | -15.7% (early: 1 stretch) | yes, weak |

- **Question 1: no general claim.** Stroke rate fell late for 3 skaters
  and rose for 2, short of 4 of 6, in both versions.
- **What the hand checks showed.** Two stretches were the counter
  double-counting (Bloemen 09-03, van der Poel 10-17). Three (van der Poel
  10-03 and 10-09, Rijhnen 02-03) begin with corner-exit crossovers
  (gaps 10-18 frames) before the straight rhythm (23-29 frames): six
  thumbnails per stretch cannot show where the corner ends. This error was
  not in the 8c test, which used pure straightaway clips, and it pushes
  rates up. Fix for later: mark the exact corner-exit frame per stretch.
- van der Poel's hand-checked -11% rests on checks added after the result
  and on stretches that still include corner strokes: not settled.
- Stroke rate and length differ more between skaters than within a race:
  van der Poel about 80/min at about 10-11 m per stroke; Roest, Kramer and
  Bergsma about 55-75/min at 11-15 m.

### 8e: Pacing Sheet
- One visual sheet per skater, in the Phase 7 style: lap times, stroke rate
  and stroke length lap by lap, with a "how to read this sheet" box and
  comparison to the other skater in the pair.

### 8f: Evaluation & Write-Up
- `PHASE8_SUMMARY.md`: what was tested, what held up, what didn't.

## Tier 2: If Time Allows
- **Shorter distances** (1000 m, 1500 m): more stroke-rate change per lap,
  but fewer laps; would need new downloads.
- **Team pursuit**: whether the three skaters stroke in step, and whether
  that changes at lead swaps or late in the race.
- **Corners**: crossover rate in corners vs straights, and whether it
  changes as skaters tire.
- **Stroke phase classifier**: label stroke phases (push, glide, recovery,
  crossover) frame by frame with the marking player, then train a small
  classifier on the pose measures already computed (foot positions, feet
  apart, hip movement) and test it on skaters it never saw, with the
  target fixed before testing. Inspired by Google Teachable Machine (see
  Considerations below).

## Considerations

**Google Teachable Machine (10/30).** Teachable Machine trains an image or
pose classifier in the browser from labelled examples, and could in
principle label stroke phases. Not used directly, for three reasons:
- It classifies whole frames, so on broadcast footage it would partly learn
  the background, camera angle and suit colours rather than the skating,
  and would likely fail on a new race.
- It has no simple way to test on held-out skaters with a target fixed
  before testing, so its accuracy could not be trusted on the same terms
  as the rest of the project.
- The hard part of stroke detection is exact timing (the counter's misses
  were 4-5 frames off), and frame-by-frame classifiers tend to flicker at
  exactly those transitions.

The same idea in a testable form is the Tier 2 stroke phase classifier:
trained on pose measures instead of raw images, with ground truth from the
marking player (`docs/MARKING_PLAYER.md`), and tested on skaters it never
saw. Phase 6d did something similar for corners vs straights (81.7%).

## Risks
- Broadcast cameras switch views and cut often, so some laps may not be
  countable at all. Plan: report how many laps per skater are countable,
  and only use skaters with enough of them.
- Lap times are per 400 m, but strokes are counted only where the skater is
  on screen. Plan: stroke length uses stroke rate on the visible part of
  the lap and the speed of the whole lap, and this is stated as a limit.
- Slow-motion replays must be excluded (they would make stroke rate look
  low).

---

## Real Bugs & Limitations Found This Phase (Running Log)

| Date | Issue | Fix |
|---|---|---|
| 10/29 | First pass of hand marks had double presses (marks 1-4 frames apart) and marks that looked accidental on the last frame | All marks redone; player shows "END OF CLIP"; last-frame set-downs confirmed real on rewatching |
| 10/29 | Phase 7 foot-height touchdown signal found only ~40% of set-downs on side-on broadcast footage | Feet furthest apart sideways used instead (chosen on tune clips only) |
| 10/29 | Counter can't see a set-down right at a clip edge (its signal comes 7 frames earlier) | Scoring window fixed before the test: 10 frames after start to 3 before end |
| 10/31 | One of Bloemen's two early stretches (pair09-03) gave 120 strokes/min, set-downs every 15 frames, faster than any hand-marked straightaway | Hand check of that stretch set up (added after seeing the result; both versions to be reported) |
| 11/1 | Hand checks: the counter double-counted on 2 of 5 high-rate stretches | Hand-checked version reported beside the counter version |
| 11/1 | Stretches labelled straight from six thumbnails can start with corner-exit crossovers, which are faster than straight strokes and push the rate up; not covered by the 8c error | Recorded as a limitation; fix is marking the corner-exit frame per stretch |
