# Phase 8 Summary & Conclusion

> **Finished 11/5. Phase 8 complete.** Numbers are taken from the analysis
> outputs and `docs/CAPABILITIES_PHASE8.md`. Opponents' lap times were not
> read and stay future work.

## The Question
To what extent can single-camera pose estimation count a long track skater's
strokes reliably enough to measure **stroke rate and stroke length lap by
lap**, and how do they change across a race and between faster and slower
skaters?

Three questions were fixed before looking at results:
1. Does stroke rate rise, fall or stay level across a race?
2. When lap times slow late in a race, is it mostly a drop in stroke rate or
   in stroke length?
3. Does the faster skater differ in stroke rate, stroke length, or both?

## What Was Actually Tested
- **8a (lap times):** 78 lap times for six skaters in the Beijing 2022 men's
  5000 m, read from the broadcast split graphic and checked against the
  cumulative times. Official lap times could not be found online.
- **8b (ground truth):** every skate set-down marked by hand on 21 straightaway
  clips, 10 of them held out by skater.
- **8c (counter):** an automatic stroke counter, set on the tune clips and
  tested once, with an 85% target fixed before testing.
- **8d (race analysis):** all 133 usable race stretches labelled by hand;
  stroke rate compared between early laps (2-5) and late laps (9-12) under
  rules fixed before any race result: a 10% bar from the counter's own error,
  and a general claim needing the same direction in at least 4 of 6 skaters.
- **Follow-up checks (added after seeing the result):** seven stretches with
  high rates hand-checked, and a second analysis on straight strokes only,
  with its rules fixed before any trimmed rate was computed and the marking
  done with the lap hidden.
- **8e (pacing sheets):** one sheet per skater in `reports/pacing/`.

## The Result

**8a: lap times show clear pacing differences.** van der Poel stayed almost
flat (median 29.06 s early, 29.15 s late). Bloemen faded most (28.90 s to
31.08 s).

**8c: the counter just missed its target on clean clips.** 84.2% of set-downs
found within 0.1 s on normal-size test clips (target 85% not met, by one
set-down), extras 10.5%, stroke counts matched on 6 of 7 clips. Small skaters
reached 50%.

**8d: the counter was not reliable enough on race footage.** Seven stretches
with high rates were hand-checked. In four the counter over-counted, and in
three the stretch included corner crossovers, which are faster than
straightaway strokes. Six thumbnails per stretch could not show where the
corner ended. Neither problem was covered by the 8c error, so the 10% bar
was too generous for race footage.

| Skater | Counter | Hand-checked | Straight-only (strokes/min) | Straight-only change |
|---|---|---|---|---|
| Kramer | +3.7% | +3.7% | 61.0 → 66.7 | +9.3% |
| Roest | +7.9% | +7.9% | 65.5 → 71.4 | +8.9% |
| van der Poel | -4.5% | -13.7% | 70.6 → 70.6 | 0.0% |
| Bloemen | -30.7% | -15.7% | no early stretch | can't measure |
| Rijhnen | -6.0% | -6.0% | no early or late stretch | can't measure |
| Bergsma | no early stretch | no early stretch | no early stretch | can't measure |

The counter and hand-checked columns are the rules as first fixed. The
straight-only column is the cleanest data but the second analysis.

**Question 1: no general claim.** The counter version has 3 skaters falling
and 2 rising, short of 4 of 6. Straight-only leaves only 3 skaters, none past
the 10% bar. The large fall first seen for Bloemen came mostly from counter
and corner errors, and whether any real fall remains can't be tested.

**Question 2: not answered.** Stroke rate did not fall for the skaters who
could be measured, so any late slowing would have to come from stroke length.
But stroke length here is worked out from lap speed and stroke rate, not
measured separately, and Bloemen, who slowed most, has no early rate.

**Question 3: not answered.** Straight-only medians run from about 55 to 71
strokes/min, with no clear link to finishing time: the two fastest skaters
(van der Poel 71, Roest 69) are close, and Rijhnen's and Bergsma's values rest
on one and two stretches. An earlier contrast (van der Poel about 80/min,
Roest about 66-71) did not survive the corrections.

## Why This Result Is Trustworthy
- **Targets and rules were fixed before results,** and the shortfalls are
  reported as shortfalls (8c by one set-down; no general claim in 8d).
- **Results were not replaced when they changed.** The first version stays on
  record beside each later one, and every check added after seeing a result
  is labelled as such.
- **Hand marks follow the same rule as the counter,** including the 3-set-down
  minimum, even when that removed a stretch.
- **The straight-only marking was blind to early and late:** the lap was
  hidden and the clips shuffled.
- **Skating knowledge found the main error.** The corner crossover pattern was
  spotted by the researcher while marking, not by any automatic check.

## Honest Limitations
- One race, six skaters, one broadcast. Broadcast cameras decide which laps
  can be measured, and most usable straight pieces are only a few seconds long.
- Hand checks went only to high rates, so corrections went one way. Lower rates
  were never checked and may hold counter errors, including missed set-downs.
- Early vs late rests on 2-3 stretches per skater even where it can be measured.
- Rates still swing from lap to lap (van der Poel 61 to 82 strokes/min) while
  that skater's lap times are almost flat, so some of what remains is probably
  measurement noise.
- The second analysis was added after seeing the first result.
- Stroke length uses the whole lap's speed with a straightaway stroke rate, and
  lap speeds are close, so it mostly mirrors stroke rate.
- Corner strokes were not measured at all, though corners are about half of
  each lap.

## What It Actually Means
Pose estimation can count strokes well on clean, large, pure straightaway
clips, but it cannot yet produce trustworthy lap-by-lap stroke rates from a
race broadcast without a person checking it. The obstacles are specific: the
counter over-counts on some race stretches, it cannot tell crossovers from
straight strokes, and broadcasts rarely show a skater long enough. The clearest
result is about method: each hand check moved the numbers, and early
conclusions (Bloemen's fall, van der Poel's high rate) disappeared once the data
was cleaned. Lap times from the broadcast, on the other hand, proved reliable
and already show pacing clearly.

## What This Sets Up Next
- **Telling corner from straight automatically,** which the Tier 2 stroke phase
  classifier could test, would remove the need for hand trimming.
- **Opponents' lap times:** the split graphic shows both skaters in a pair, so
  the other skater's laps could be read too, for pair comparisons and more
  skaters.
- **Footage with a fixed view of the straightaway** would give longer stretches
  and early laps for every skater, which is what this analysis lacked most.
