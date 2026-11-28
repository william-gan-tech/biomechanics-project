# Phase 9 Checklist: What to Improve or Train Next

> Made 11/26, after round 4 and the first upper-body check; updated 11/27.
> Tick items off as they are done; the evidence for each is in
> `docs/CAPABILITIES_PHASE9.md`. Improvement results marked "six-skater test"
> come from leaving each marked skater out in turn: they are optimistic, so a
> fresh race stays the final check.

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

## Approach (decided 11/27)
- Improve the **model as a whole** (corners, straights, timing, identity,
  frames judged), measured over all marked races together, not tuned to one
  skater, race or moment.
- **Main measure:** leave one broadcast out (Beijing, Calgary, Inzell; more as
  races are marked), with leaving one skater out as a second view. A race kept
  completely untouched is the final check.

## Decided
- [x] **Feet cutoff for "can't judge":** 0.3 (six-skater test 11/26).
- [x] **Pose model settings (11/27):** 1.0 s window, 0.8 s smoothing, smaller
      skaters (down to 100 px) allowed near corner entries and exits for
      tracked people already confirmed. Six-skater test: 96.0% average, 79% of
      boundaries within 0.5 s (up from 94.1% and 59%).
- [x] **Upper-body model retrained** (`phase9/tm_model_upper_v2`, 11/26).
- [x] **Identity with Teachable Machine tested** (11/27): 97-98% of non-racers
      caught on Calgary and Inzell, 74% on Beijing (each group unseen).

## Where the whole model is weakest (all marked races, 11/27)
| What | Now | Aim |
|---|---|---|
| Corner timing (boundaries within 0.5 s) | 79% | 85% |
| Identity: non-racers caught on a broadcast not trained on | 74-98% | 95%+ everywhere |
| Full-body Teachable Machine corners | 87% (round 4) | 95% |
| Frames judged (rest "can't judge") | 88% | as high as possible |
| Identity: real racers kept | 90-95% | 98% |
| Straights overall | 96% | keep, while raising the rest |

## Checklist
**Data (the biggest lever for the whole model)**
- [ ] **Mark more races,** ideally both skaters, from broadcasts not used yet;
      every race adds training variety and one more leave-one-out test.
- [ ] **Keep one race untouched** for the final check.

**Training**
- [ ] **Final identity model trained on all three groups** (810 pictures),
      then used to filter boxes before the stretch check by eye.
- [ ] **Full-body Teachable Machine corners:** add corner pictures (especially
      near entries and exits) so it stops leaning towards "straight".
- [ ] **More upper-body pictures** across skaters (most are Ghiotto's now).
- [ ] **Which model judges the "can't judge" frames:** upper-body or full-body
      Teachable Machine, decided on the leave-one-broadcast-out test.
- [ ] **Combine models:** the pose model and Teachable Machine together (for
      example, pose for corners, Teachable Machine for straights, or their
      confidence averaged), designed on the marked races, judged on all of them.

**Timing**
- [ ] **Corner timing to 85%,** or report the gap as a limit of broadcast
      footage (most misses come from Calgary's camera cutting at corners).

**Checks and comparisons**
- [ ] **Phase 6 detector comparison** (pending since round 1).
- [ ] **Spot-check a sample of old Phase 5 and 6 labels** (decided 11/14).

**Final check**
- [ ] **One fresh race,** more than one skater if possible: mark it, filter
      boxes with the identity model, stretch check by eye, rules written
      first, run once.

**Using the model**
- [ ] **Stroke rate without corner strokes** (the Phase 8 problem).
- [ ] **Pacing database (9B):** more races.
