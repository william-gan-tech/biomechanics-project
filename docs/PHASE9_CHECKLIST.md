# Phase 9 Checklist: What to Improve or Train Next

> Made 11/26, after round 4 and the first upper-body check. Tick items off as
> they are done; the evidence for each is in `docs/CAPABILITIES_PHASE9.md`.

## Where things stand

### What works
| | Evidence |
|---|---|
| Corners with the feet visible | 97.8-100% on three skaters never seen in training (rounds 3 and 4), on two broadcasts |
| Straights with the feet visible | 92% on a new race (round 4), up from 81% (round 3) |
| Working on new broadcasts | Beijing, Calgary and Inzell (different arenas, cameras and frame rates), all above the 85% target |
| Knowing when it cannot tell | the "can't judge" rule removes the frames where it was guessing (about 15-22% of a race) |
| The researcher's Teachable Machine models | full-body model passed round 4 (90.8%, best on straights at 94.5%); upper-body model 95.8% on Eitrem's can't-judge frames (11/26 check) |

### What needs work, and why
| | Why |
|---|---|
| Corner timing (boundaries) | Never properly tested: 66.7% in round 3 (target 85%), too few to score in rounds 1, 2 and 4. Broadcasts rarely keep a close-up through a corner entry or exit. |
| The can't-judge frames | About 15-22% of a race still has no answer. The Teachable Machine models look promising there but have not had a proper test. |
| Upper-body corners specifically | 92% against 99% for straights in the 11/26 check, so corners are the weaker side of the upper-body model. |
| Finding the right skater | Colour still mixes up officials and team staff, so every race needs the researcher's stretch check (the Teachable Machine identity idea, 9g, is not built yet). |
| Amount of evidence | Each round tests one skater on one race. More races would make the numbers more trustworthy. |

## 1. Training
- [ ] **Train the upper-body model again** with the reviewed set (460 corner,
      470 straight) and export it (kept as `phase9/tm_model_upper_v2`).
- [ ] **More upper-body pictures of Swings, Bergsma and Bloemen** (most are
      Ghiotto's now), and more upper-body corners, the weaker side (92% against
      99% for straights in the 11/26 check).
- [ ] **Full-body Teachable Machine corners:** they dropped to 87% after the
      late-straight pictures were added. Add corner pictures, especially near
      corner entries and exits, so it does not lean towards "straight".

## 2. Decide before the next round (written down first)
- [ ] **Which model judges the "can't judge" frames:** the upper-body model or
      the full-body Teachable Machine (95.8% against 98.1% on 279 of Eitrem's
      frames, too few to separate them).
- [ ] **Whether to combine models** (the pose model is near-perfect on corners,
      Teachable Machine stronger on straights), and how.

## 3. Testing
- [ ] **Round 5 on a fresh race** (Tomaszow 2026 or Stavanger): mark the
      skater, check identity by stretch, write the rules, run once. System: pose
      model where the feet are visible, a Teachable Machine model elsewhere.
- [ ] **More test skaters per round,** so one skater's quirks do not decide a
      result (for example both skaters of a pair when the camera allows it).
- [ ] **Corner boundary timing:** find a way to test it (races that keep
      close-ups through corner entries and exits, or scoring boundaries on
      smaller skaters only near the boundary), or report it as a limit of
      broadcast footage.
- [ ] **The Phase 6 detector comparison,** pending since round 1: the
      shoulder and knee points it needs have now been added to the pose files.

## 4. Skater identity
- [ ] **Teachable Machine for skater identity (9g):** racing skater against
      officials, coaches and staff, trained on the stretches already checked,
      to make the stretch check by eye shorter.

## 5. Labels
- [ ] **Spot-check a sample of the old Phase 5 and 6 labels** by eye (decided
      11/14, not yet done).
- [ ] Check marks wherever picture labels and marks disagree in a cluster
      (as with Ghiotto around frame 14850, which turned out right).

## 6. Using the model
- [ ] **Stroke rate without corner strokes:** use the model to keep only
      straight strokes, the problem that limited Phase 8.
- [ ] **Pacing database (9B):** add races beyond Beijing 2022 (paused since
      11/12).
