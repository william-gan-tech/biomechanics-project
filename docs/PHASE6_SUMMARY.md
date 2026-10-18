# Phase 6 Summary & Conclusion

> **Reviewed 10/16; 6f added 10/17. Phase 6 complete.** Numbers are taken
> from the analysis outputs and `docs/CAPABILITIES_PHASE6.md`.

## The Question
To what extent can a dedicated multi-object tracking approach reliably
follow **both** skaters of a long-track pair through broadcast footage
without identity swaps, and can corner and straightaway phases be detected
**automatically** well enough that the Phase 5 form analysis runs on an
unlabelled race video?

Phase 5 had to skip roughly half of all candidate stretches because both
skaters were in frame, and every segment had to be labelled corner or
straightaway by hand. This phase tested whether both limits could be
removed.

## What Was Actually Tested
- **6a (tracking ground truth):** which track number was on which skater,
  checked frame by frame on 6 development clips and 3 held-out clips from
  pairs never used during development. The target (≥95% identity correct)
  was fixed before testing.
- **6b (multi-object tracker):** YOLO person detection + ByteTrack
  (`ultralytics`), then two ways of making its output usable: suit-colour
  relabelling, and "pure pieces" (cutting tracks wherever identity could be
  lost).
- **6c (recovering both-skaters footage):** the tracker run over six whole
  races (Inzell 2026 5000m and five Beijing 2022 pairs), pure pieces
  reviewed and labelled with skater, phase and camera shot, pose measured
  on each piece's box, and the result fed into the Phase 5 fatigue
  comparison (5f).
- **6d (automatic corner/straight detection):** the 5g classifier
  improved and re-tested leave-one-skater-out (never tested on a skater it
  trained on), with and without the recovered footage. Target ≥85%,
  fixed beforehand.
- **6f (single-camera 3D pose estimates):** MediaPipe's estimated 3D
  ("world") landmarks, tested on whether they remove the front-on vs
  side-on difference (target: at least half removed, side-on values within
  ~5° of 2D), then used to re-run the 5f fatigue comparison and the 6d
  classifier.

## The Result

**6a/6b: the tracker alone did not meet the target; pure pieces did what
was needed.**

| | Development clips (6) | Held-out clips (3) |
|---|---|---|
| Tracker alone, identity correct (strict) | 89.9% | 98.3% |
| Pure pieces, kept frames on the right skater | **100%** | **100%** |
| Pure pieces, footage kept | 42.6% | 60.8% |

- The tracker is reliable when the skaters are apart. Every identity error
  came from overlapping skaters (one box around both, swapped IDs after
  they separate) or a skater briefly losing their box.
- Suit-colour relabelling did not beat the tracker's own IDs (best 83.2%)
  and was not adopted.
- Pure pieces only has to guarantee that each piece is one person, since
  skaters are identified when the pieces are reviewed. Two fixed rules (a
  box can't teleport; overlapping skaters are unsafe) gave 100% purity on
  clips it was never developed on. A third rule was added after a real
  failure: on a full race, a camera cut put one skater exactly where the
  other had been, so a piece switched skater with no jump in the box.
  Camera-cut detection (`detect_cuts.py`) now ends every piece at a cut.

**6c: footage was recovered, but it cannot add skaters to 5f on its own.**
- Six races gave 50-80 pieces each (133-224 s per race). Two Beijing pairs
  (Wenger/Cepuran, van der Poel/Swings) and the Inzell race were fully
  reviewed for the early and late laps.
- Pose measured on a crop around each piece's box agrees with Phase 5 on
  side-on footage: on 6 of 6 verified side-on segments, average knee angles
  were within ~4° and trunk lean within ~5°.
- **Front-on footage is badly biased.** Front-on pieces read about 30°
  more upright and straighter-legged than side-on pieces of the same
  skater in the same race, and pose sometimes failed outright on head-on
  close-ups (near-straight knees while the skater was visibly crouched).
  A first 5f run on Wenger showed a false "upright late in the race"
  change for exactly this reason: his late laps were mostly front-on. 5f
  now uses side-on recovered footage only.
- With side-on footage only, Wenger has 2 late-race segments and van der
  Poel 1 early and 1 late, too few for a result. Jílek (Inzell) has no
  late-race pieces at all. When both skaters are together late in a race,
  broadcasts mostly film them from the front.

**6d: automatic corner/straight detection reached 81.7%, short of 85%.**

| Version (leave-one-skater-out) | Segment accuracy |
|---|---|
| Original (5g) | 77.3% |
| Angles only, 2-second windows (best) | **81.7%** |
| Best version + recovered footage | 76.3% overall |
| ... on recovered side-on pieces | 77-80% |
| ... on recovered front-on pieces | 60-74% |

Dropping camera-dependent features did not hurt accuracy and made lean
angle and its variation the most useful features, which is what physically
distinguishes a corner. More broadcast footage did not help: accuracy on
the original skaters was unchanged, and it fell on front-on footage for the
same reason as in 6c.

**6f: 3D pose estimates remove most of the camera effect on trunk lean,
and with it most of Phase 5's fatigue lead.**

| 3D vs 2D | Trunk lean | Knee angle |
|---|---|---|
| Front-on vs side-on gap removed | **68%** | 54% |
| Side-on agreement with 2D (median difference) | 1.7° | 3.8° |

- Front-on trunk lean moved from 30° (2D) to 60° (3D), next to 63-65°
  side-on; knees stayed ~16° straighter front-on, so only trunk measures
  become camera-independent.
- 3D does not simply flatten everything: on side-on footage the spread
  between segments and corner-vs-straightaway differences are unchanged.
- **5f on 3D angles, all shots, per skater:** late-race trunk-lean
  left/right difference +0.7° (7 of 9 skaters, p = 0.038), against +3.3° in
  2D. This meets the pre-set bar (at least 7 of 9, p < 0.05), but it is
  about 6 mm at the shoulder, well inside frame-to-frame scatter, and a
  visual check of the skater with the largest 2D increase (Bergsma) showed
  no visible change. **Judged not practically meaningful.**
- 3D angles did not improve corner/straight detection (73-74% vs 77% in 2D
  on the same segments).

## Why This Result Is Trustworthy
- **Targets were fixed before testing** (95% identity, 85% phase
  accuracy), and both shortfalls are reported as shortfalls.
- **The stricter scoring rule is the headline.** A lenient rule (counting
  only the frame of each swap) gave 95.5% and would have "met" the target
  while hiding tracks that mix two skaters.
- **Held-out test:** pure pieces was confirmed on 3 clips from pairs not
  used during development, with its rules left unchanged. The suit-colour
  method, revised three times on the same clips it was scored on, is
  flagged as optimistic.
- **Review caught what the scored clips didn't** (the camera-cut failure),
  and the fix was re-checked: no piece spans a detected cut, and the
  scored clips stayed at 100% purity.
- **Identity labels were cross-checked** by suit colour inside each box
  (all 35 Wenger/Cepuran labels agree) and on enlarged frames where colour
  was unreliable.
- **Phase 5 results were reproduced exactly** with the new code paths
  turned off, so nothing in Phase 5 changed silently.
- **The 3D test checked itself:** a fixed target, a check that 3D wasn't
  flattening real differences, a match between the new 2D values and Phase
  5's (r = 0.96), and a visual check of the key case before interpreting
  the result.
- **No after-the-fact changes:** two good side-on van der Poel corners fell
  in the middle third of the race and were left out rather than widening
  the early/late windows.

## Honest Limitations
- Single-camera 2D broadcast video remains the dominant limitation, now
  shown in two more ways: front-on pose bias and its effect on phase
  detection.
- The held-out tracking test is 3 clips; the full-race reviews cover 3 of
  6 races (Beijing pairs 3, 7 and 8 were not reviewed).
- The camera-cut detector can miss cuts, so every piece still needs a
  look before use.
- Suit colour is only a rough identity check (the blue rink boards look
  like a blue suit), and identity, phase and shot calls are one person's
  judgement.
- Pure pieces keeps 43-61% of both-skaters footage; the rest (mainly
  overlaps) is still lost.

## What It Actually Means
For the first half of the question: **two skaters can be kept apart
reliably enough to recover usable footage**, not by trusting the tracker's
identities but by cutting tracks into pieces that are each guaranteed to be
one person. For recovering *useful* footage, though, the camera matters
more than the tracker: the both-skaters stretches it unlocks are mostly
filmed front-on late in the race, the one view where 2D pose is unreliable,
so they don't strengthen the fatigue comparison. For the second half:
**automatic phase detection improves to 81.7% but levels off on broadcast
footage**, for the same underlying reason. 6f showed that estimated 3D
pose can remove most of the camera effect on trunk lean, and that once it
does, Phase 5's main fatigue lead (late-race trunk asymmetry) shrinks to
under 1°, too small to matter. So fatigue-linked form change is **not
detectable from broadcast footage with these measures**. Every result in
Phases 5 and 6 points the same way: the remaining limits come from
single-camera broadcast footage rather than from the methods.

## What This Sets Up Next
- **Controlled fixed-camera footage** (side-on, with a real fatigue
  measure such as lap times or heart rate) is now the clearest next step
  for 5f and for automatic phase detection.
- **3D trunk measures** should replace 2D trunk lean in any future analysis
  that mixes camera views.
- Recovered side-on pieces can still be added to the elite reference,
  where early and late laps don't matter.
