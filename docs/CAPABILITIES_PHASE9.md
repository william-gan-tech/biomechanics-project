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

## Status (11/9): Planned. No labelling, models or lap times yet.

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

### 9b: Phase Model
- Input: pose over a short window of frames (feet, hips, leg length, as in
  Phases 6-8), plus camera-cut and shot information. A simple model first;
  something larger only if it beats the simple one on unseen races.
- Output per frame: start, straight, corner, or "can't tell" (skater too small,
  not live, not the skater).
- *Proposed targets* (fixed before the first test):
  - At least 90% of frames correct on an unseen race, normal-size skaters only.
  - Corner entry and exit found within 0.5 s for at least 85% of boundaries.
  - Compared with the Phase 6d detector (81.7%) on the same race.

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

## Rounds (Running Log)

| Round | Date | Test race | Frames correct | Boundaries within 0.5 s | Training races after |
|---|---|---|---|---|---|
