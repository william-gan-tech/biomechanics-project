# Phase 6: Dual-Skater Tracking & Automatic Phase Detection

> Plan set 10/8. Tracker approach: `ultralytics` YOLO + ByteTrack; 6a
> target: identity correct on ≥95% of checked frames.

## Formal Research Question

> To what extent can a dedicated multi-object tracking approach reliably
> follow **both** skaters of a long-track pair through broadcast footage
> without identity swaps, and can corner and straightaway phases be detected
> **automatically** well enough that the Phase 5 form analysis runs on an
> unlabelled race video?

## Status (10/14): 6a/6b done: "pure pieces" confirmed on 3 held-out clips and adopted, with camera-cut detection added after a real-footage failure. 6c: Inzell pieces labelled; 5 Beijing races recovered (81 sheets), pairs 4 and 10 reviewed; crop-based pose checked (usable on side shots); recovered segments connected to 5f (side-on only), but Wenger and van der Poel are both too thin; recovered footage can't add 5f skaters on its own (late laps mostly front-on). Next: use it for the elite reference and 6d. 6d at 81.7%.

## Why This Phase

Phase 5 showed the analysis methods work, but broadcast footage limited how
much of it could be used and required hand-labelling everything:

- **Roughly half of all candidate stretches were unusable** because both
  skaters were in frame (Inzell, Beijing pairs 3, 4, 7, 8; much of every
  race). The single-target tracker can't tell them apart reliably.
- **Every segment had to be hand-labelled** corner/straightaway (and camera
  shot type). The 5g classifier test reached 77% (target 85%), leaning on a
  camera-dependent feature.
- Dual tracking was attempted on 9/22 (three iterations) and failed on
  identity swaps. Diagnosed causes: ~21% duplicate MediaPipe detections, and
  similar-looking uniforms in different technique phases at the same moment.
  The diagnosis was that this needs a proper multi-object tracker, not more
  patches.

## What "Complete" Means Here

A defensible v1: both skaters tracked through a race with a **measured**
identity-swap rate against manually checked ground truth, and automatic
phase detection with a **measured** accuracy on skaters it never trained on.
Each result is reported whether it succeeds or not, with the threshold for
"good enough" fixed before testing.

---

## Tier 1: Core

### 6a: Tracking Ground Truth & Benchmark
Before building anything, define how success is measured.
- Pick test clips with both skaters in frame (Beijing pairs, Inzell).
- Mark which skater is which on sampled frames (frame sheets
  with numbered boxes) as the identity ground truth.
- Metrics: identity-swap count per minute, fraction of frames where each
  skater is correctly tracked, fraction of frames lost.
- Baseline: the current single-target tracker and the 9/22 dual attempts.
- **Target (fixed 10/8, before testing):** identity correct on ≥95% of
  checked frames.

**Baseline result (10/9):** all 6 test clips were checked
(which track number was on which skater in each sampled frame), saved as
`dual_tracking/identity_ground_truth.csv`; `score_dual_tracking.py` scores
any tracker version against it.
- **Scoring rule:** each track number belongs to the skater it covers most;
  frames where it is on the other skater count as wrong (the track mixes two
  people), as do one box around both skaters and two boxes on one skater. A
  lenient rule counting only the frame of each swap gives 95.5% but hides
  mixed tracks, so it is not the headline.

| Clip | Identity correct | Errors |
|---|---|---|
| Rijhnen / Lehman | 100% | None |
| Bloemen / Zakharov (b) | 100% | None (new IDs when a skater leaves frame) |
| Eitrem / Jílek | 100% | None |
| Ghiotto / Eitrem (start) | 94% | 3 frames with two boxes on one skater |
| Bloemen / Zakharov (a) | 83% | 1 swap: ID passed on when a skater lost their box |
| Bergsma / Ichinohe (close-up) | 71% | one box around both while overlapping, 2 swaps |
| **Overall (286 skater-frames)** | **89.9%** | **target 95% not met** |

- Reliable when the skaters are apart (a clear improvement on 9/22); all
  identity errors come from overlapping skaters or a skater briefly losing
  their box.
- **Next fix:** there are always exactly two skaters in different suits, so
  assign every box to skater A/B by suit appearance (learned from clean
  frames) instead of trusting track numbers, and drop frames where one box
  covers both skaters. Re-score against the same ground truth.

**Fix attempts (10/9):** `relabel_dual.py`
and `score_dual_tracking.py --labels / --purity`:

| Approach | On the right skater (of frames kept) | Coverage of checked skater-frames |
|---|---|---|
| Tracker IDs only (baseline) | 89.9% | 100% |
| Appearance, each frame matched independently | 72.6% | ~79% |
| Appearance, per track piece (median over frames) | 74.8% → 80.7% → 83.2% (three revisions) | ~89% |
| **Pure pieces:** tracker IDs, cut where a box jumps further than a skater can move, frames near any overlap of two skater-sized boxes dropped | **100%** | **42.6%** |

- **Appearance (suit-colour histograms) did not beat the tracker's own IDs.**
  It fixed the swap in Bloemen/Zakharov "a" (83% → 95%) and helped
  Ghiotto/Eitrem (94% → 98%), but similar suits (Rijhnen/Lehman, both mostly
  black) and officials in the skater's colours (Dutch coaches in orange in the
  Bergsma clip) made it worse elsewhere. Colour histograms aren't distinctive
  enough on broadcast footage.
- **Caveat:** the appearance method was revised three times against the same
  six clips it was scored on, so those numbers are optimistic.
- **Pure pieces** reframes the goal: for recovering data (6c), the tracker
  only needs to guarantee each piece is ONE person, since skaters are
  already identified manually when reviewing frame sheets. Two general rules (a box
  can't teleport; overlapping skaters are unsafe), set once and not iterated,
  gave **100% purity** with 42.6% coverage. Phase 5 recovered 0% of these
  both-in-frame stretches, and contaminated data is worse than less data.
- **Next:** confirm on 2-3 new, unchecked clips (fair
  held-out test), then use pure pieces for 6c. **Held-out clips prepared
  (10/9):** `dual_tracking/heldout_clips.csv`: Beijing Engebraaten/Trofimov
  (42306-42420), Wenger/Cepuran (61578-61752) and van der Poel/Swings
  (187398-187479, starts with the skaters overlapping), all pairs not used
  in the 6 test clips; review sheets ready for the identity
  check. Pure-pieces rules stay fixed as they are for this test.

**Held-out test result (10/11), pure pieces confirmed:** the 3 new
clips were checked (added to `identity_ground_truth.csv`; one
unreadable Trofimov number was identified from the tracker output by
position, and a "16" read as 22 for the same reason).

| | Development clips (6) | Held-out clips (3) |
|---|---|---|
| Tracker alone, identity correct (strict) | 89.9% | 98.3% |
| **Pure pieces, kept frames on the right skater** | **100%** | **100%** |
| Pure pieces, footage kept | 42.6% | 60.8% |

- On clips it was never developed on, every kept piece contained only one
  skater. **Pure pieces is adopted for 6c.**
- The tracker alone also scored well on the held-out clips (98.3%), since they had
  less close-up overlap than Bergsma/Ichinohe. Its raw score depends heavily
  on how much the skaters overlap (71-100% across the 9 clips); pure pieces
  stayed at 100% throughout.
- All 9 clips together: tracker alone 92.3%; pure pieces 100% purity, 47.9%
  coverage. Possible later improvement in
  coverage: a learned person re-identification model (designed for telling
  people apart) instead of colour histograms.

### 6b: Multi-Object Tracker
- A person detector that gives one clean box per skater (fixes the 9/22
  duplicate-detection problem), plus a tracker that uses motion (Kalman
  prediction) and appearance (suit colours) to keep identities.
- Options: an established detector + tracker (e.g. YOLO + ByteTrack via the
  `ultralytics` package; needs installing, small model download), or a
  custom Kalman + Hungarian-assignment tracker on top of MediaPipe (no new
  dependencies, but closer to what failed on 9/22).
- MediaPipe pose then runs separately on each tracked skater's box.

**First run (10/8):** installed `ultralytics` (8.4.174, YOLO11n). Built
`dual_track.py` (YOLO person detection + ByteTrack; boxes under 8% of frame
height dropped to skip distant crowd/officials) and ran it on 6 test clips
with both skaters in frame (`dual_tracking/test_clips.csv`): Beijing
Bergsma/Ichinohe, Bloemen/Zakharov (x2), Rijhnen/Lehman; Inzell
Eitrem/Jílek; Calgary Ghiotto/Eitrem start. Review sheets with each
track's ID drawn are in `dual_tracking/*_review.jpg` for the
identity check.
- **Preliminary look (not ground truth):** Rijhnen/Lehman tracked
  cleanly (IDs 1 and 2 throughout). Bergsma/Ichinohe close-up: when the two
  skaters overlap, the detector draws **one box around both** (e.g. 126312,
  126336, 126366), and after they separate the IDs can come back **swapped**
  (ID 93 on Bergsma at 126348, on Ichinohe by 126372). This pins the 9/22
  failure to a specific cause: overlapping skaters merge into one detection.
- Track counts per clip are high (8-40) because officials, coaches, crowd
  and camera cuts create extra and new IDs; what matters is whether the two
  skaters keep their own IDs, which the identity check measures.

### 6c: Two-Skater Extraction for Phase 5 Data
- Re-run the race scans with dual tracking to recover stretches previously
  skipped because both skaters were in frame.
- Samples are checked manually (as in Phase 5) before any are logged.
- Re-run 5f with the extra data: the most direct test of whether the
  trunk-asymmetry lead holds with more data.

### 6d: Automatic Corner/Straight Detection
- Start from the 5g classifier (77%) and the 163 labelled
  segments as training data.
- Improvements to test: longer time windows (a corner lasts ~10 s), the
  skater's movement across the frame and lean direction over time, and
  removing camera-dependent features that let it recognise shots instead of
  track sections.
- Evaluate leave-one-skater-out, same as 5g. **Target: ≥85% segment
  accuracy.**
- If reached: connect to `form_report.py` so a report can run on an
  unlabelled clip (completing what 5g couldn't).

**First improvements (10/8),** leave-one-skater-out, `phase_classifier.py`
with `PHASE_CLF_ANGLES_ONLY` / `PHASE_CLF_WIN`:

| Variant | Segment accuracy | Baseline |
|---|---|---|
| Original (5g) | 77.3% | 65.6% |
| Angles only, 1 s windows | 77.9% | 65.6% |
| Angles only, 2 s windows | 81.7% | 55.9%* |
| Whole segment, all signals | 80.2% | 64.7% |
| Whole segment, angles only | 79.0% | 64.7% |

\*2-second windows can't be cut from segments shorter than 2 s, so this is
tested on fewer segments, not directly comparable.
- Dropping the camera-dependent signals (projected hip width, pelvic tilt)
  did not hurt accuracy, and the most useful features became lean angle and
  its variation, which is physically what distinguishes a corner. A better basis,
  still short of the 85% target.
- Next to try: features describing a whole corner (~10 s) rather than short
  slices, e.g. sustained lean direction over time.

### 6e: Phase 6 Evaluation & Write-Up
Summary in the same format as Phase 5, including what didn't work.

---

## Tier 2: If Time Allows

### 6f: Single-Camera 3D Pose Estimates
MediaPipe can output estimated 3D joint positions ("world landmarks") from
one camera; the pipeline doesn't capture them yet. Test whether 3D-estimated
angles are less affected by camera angle than 2D angles: e.g. whether the
5f trunk-asymmetry result holds up better within and across shot types.
Not true multi-camera 3D (see Future Work), but uses existing footage.

---

## Future Work (not this phase)
- **Controlled fixed-camera footage with a real fatigue measure** (lap times,
  heart rate): the most direct fix for Phase 5's camera limitation.
- **True multi-camera 3D:** needs synchronized cameras; broadcast footage
  switches between cameras rather than showing one moment from several, and
  no public synchronized speed-skating dataset is known.
- **Inline skating** (Phase 5 Tier 3).

## Real Bugs & Limitations Found This Phase (Running Log)

| Date | Finding | Status |
|---|---|---|
| 10/9 | First scoring rule (count only the frame of each swap) gave 95.5% and would have "met" the target while hiding tracks that mix two skaters | Replaced by the stricter majority-owner rule (89.9%); both printed |
| 10/9 | Overlapping skaters merge into one detection box; IDs can come back swapped after they separate | Addressed by "pure pieces": frames near overlaps dropped (100% purity, 42.6% coverage on the test clips) |
| 10/9 | When one skater briefly loses their box, their track number can pass to the other skater | Addressed by "pure pieces": tracks cut where a box jumps further than a skater can move |
| 10/9 | Suit-colour appearance relabelling did worse than the tracker's own IDs (similar suits; officials in skater colours) | Documented; not adopted |
| 10/9 | Appearance method was revised three times against the same clips it was scored on | Documented as optimistic; held-out clips needed for a fair test |
| 10/11 | **Pure pieces failed on real footage:** in the full Inzell race, a piece followed Jílek, then Eitrem after a camera cut that placed Eitrem where Jílek had been (no jump in box position or size, so no rule fired). Found in the manual review; the 9 scored clips had no such case | **Fixed:** `detect_cuts.py` (pixel-difference cut detection; caught the cut at frame 303); pieces never continue across a detected cut. Re-checked: no piece spans a cut; scored clips still 100% purity, coverage unchanged. Missed cuts remain possible, so the manual review still checks each piece |
| 10/11 | Recovered pieces included people in the stands, too-small wide shots, and pieces across zoom changes | Size filter (box ≥15% of frame height), cut on sudden box-size change; people in the stands are marked "not a skater" |
| 10/11 | Review sheets were too dense (10 pieces × 8 frames) to read clearly | Changed to 4 pieces × 5 larger frames per sheet |
| 10/11 | All 70 Inzell pieces labelled: Eitrem 150 s, Jílek 52 s, 9 not-skater pieces (22 s). Frame-range labels were ambiguous where two pieces overlap in time | Overlaps resolved by suit colour in the box; P1 and P5 marked for re-check; labels in `inzell_full_race_piece_labels.csv` |
| 10/12 | Inzell Jílek pieces are all in the first ~3 min, so he has no late-race footage for 5f | Jílek goes to the elite reference only; 5 Beijing races tracked instead (pairs 3, 4, 7, 8, 10; 38-74 s of late-race footage each); 81 review sheets; review started (12 pieces) |
| 10/12 | Phase 5 pose ran on the whole frame and kept the first person found, so with two skaters it could measure the wrong one | `piece_pose.py` measures the boxed skater on a crop. Checked on 8 verified segments: 6/6 side-on agree (knees ~4°, lean ~5°); 1 of 2 front-on fails (head-on close-up). Usable on side shots, care on front; lite model kept (full model agreed less) |
| 10/12 | Pre-filling corner/straight with the 81.7% classifier could bias the manual calls | Not done; labelling from blank (10/1 rule) |
| 10/13 | Recovered pieces could not be fed into 5f (pose CSV lacked Phase 5's columns; 5f read whole-frame features only) | `piece_pose.py` saves Phase 5 columns; `get_segment_frames` reads piece pose via `pose_source`; `recovered_segments.py` (laps from race clock) and `PHASE5F_RECOVERED=1`. Phase 5 results reproduced exactly without it |
| 10/13 | **Front-on recovered segments ~30° more upright / straighter-kneed than side-on**; Wenger's late laps mostly front-on, so the raw 5f showed a false "upright late" change | 5f uses side-on recovered segments only (`PHASE5F_RECOVERED_SHOTS`, default `side`) |
| 10/13 | Pair 4 sheet 05: boxes on people at the boards while Wenger was in the shot; labels described the scene | Set to not-a-skater after checking the sheet; all 35 Wenger/Cepuran labels checked by suit colour in the box (all agree). Review rule: label who is in the box |
| 10/13 | Side-on, Wenger has only 2 late-race segments (laps 11, 12) | Not added to 5f; next: van der Poel (pair 10) early laps |
| 10/14 | Pair 10 labelled (38/54): van der Poel has 1 early (lap 4) and 1 late (lap 11) side-on corner; side-on corners in laps 5 and 8 fall in the middle third | Not added to 5f; thirds not widened after seeing the data |
| 10/14 | Front-on pose unreliable again (P48: 14-40° lean, 170-179° knees while visibly crouched) | Front-on stays excluded; P48 re-checked and confirmed front |
| 10/14 | **Recovered footage can't add 5f skaters on its own:** late-race both-skaters footage is mostly front-on | Pairs 3, 7, 8 not labelled for 5f; recovered side-on pieces to be used for the elite reference and 6d classifier training instead |
