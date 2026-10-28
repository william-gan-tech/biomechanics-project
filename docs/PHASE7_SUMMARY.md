# Phase 7 Summary & Conclusion

> **Reviewed 10/27. Phase 7 complete.** Numbers are taken from the analysis
> outputs and `docs/CAPABILITIES_PHASE7.md`. The side track (controlled ice
> footage with a real fatigue measure) was not started and stays future work.

## The Question
To what extent can single-camera pose estimation, extended to the feet,
**detect the inline double push** and describe how well it is executed,
clearly enough to help skaters learn and improve their technique?

Double push is one of the most important and hardest techniques in inline
speed skating, and it had never been measured in this project. The goal,
running through Phases 4-7, is technique analysis that helps other skaters.

## What Was Actually Tested
- **7a (foot tracking):** whether MediaPipe's heel and toe points stay on the
  skates, checked by eye on 60 frames from elite technique videos.
- **7b (ground truth):** the technique written down in measurable terms (set
  down under the body, underpush on an outside edge, edge roll, classic
  second push, fall and catch, S-curves not zig-zags) and labelled examples
  of single push, double push and flawed double push from public videos of
  elite skaters (Joey Mantia, Viktor Hald Thorup, Bart Swings, Lars
  Scheenstra, Mark Horsten, and others).
- **7c (detection):** three attempts to tell double push from single push,
  each with an 85% target fixed before testing and a test on skaters or clips
  never used to set the rule.
- **7d (feedback):** an automatic report comparing a skater with the elite
  range and giving coaching advice (tested against rules fixed beforehand),
  then a visual comparison tool where the skater or coach judges.

## The Result

**7a: feet can be tracked, if the whole body is in frame.** Heel and toe
points were on the correct skate in 41 of 42 full-body frames (98%, target
90% met), and in none of the 13 feet-only close-ups: the pose model finds the
person first, so it needs the whole body.

**7c: detection works about 3 times in 4, short of the target.**

| Attempt | Test | Result (target 85%) |
|---|---|---|
| A foot crossing under the body | other skaters | **77.8%** |
| Where the recovering foot lands | fresh clips (Thorup) | 72.2% |
| Both combined, trained on 7 skaters | a new skater (Scheenstra) | 60.9% |

The best detector was right on about 3 in 4 windows from skaters it was
never set on, well above chance (50%). It did not reach the target, and it
was weaker on newer skaters. A secondary measure (the inside foot's twist for
the second push) was at chance.

**7d: an automatic judge failed; a visual comparison works.**
- The automatic report missed both of Mantia's flawed double pushes (they
  measured like his good ones) and flagged two other elite skaters' landing
  position as a flaw (they land wider than Mantia and Thorup by style).
- The visual comparison tool (`doublepush_compare.py`) shows a skater's
  measured movement beside an elite example, with plain-language captions,
  the differences in numbers, and the coaching text as guidance. On Thorup's
  own single push vs double push it shows the difference clearly: the
  recovering foot lands under the hips in 68% of double-push strides vs 47%
  of single-push strides, and out to the side in 10% vs 33%. It was judged
  clear and useful once it explained what to look for.

## Why This Result Is Trustworthy
- **Every target was fixed before testing,** and every shortfall is reported
  as one (three detection tests, the automatic report).
- **Tests used footage never used to set the rule,** and test clips were
  chosen and labelled before results were seen. Clips offered after a result
  were not used to re-score it.
- **Labels came from skating expertise,** including the
  observations that shaped the measures (the recovering leg is what sets
  double push apart; the foot twists for the second push).
- **Checks against fooling ourselves:** a foot-tracking check by eye, a check
  of test windows without seeing the rule's answers, and a size rule so small
  or distant skaters are reported as "can't judge" instead of guessed.

## Honest Limitations
- **Single-camera 2D video is still the limit.** The defining parts of a
  double push (edge changes, wheel pressure, the twist for the second push)
  are not visible in pose points from one camera.
- Footage is public technique and race video: few skaters, edited, partly in
  slow motion, mostly elite. No footage of developing skaters was tested.
- The pose model needs the whole body in frame and the skater reasonably
  large (leg at least ~150 px); diagonal race footage breaks the measures.
- Elite skaters differ from each other, so a reference from one or two
  skaters can treat style as a flaw.
- The visual tool's usefulness was judged by one skater, not tested with
  other skaters or coaches.

## What It Actually Means
Pose estimation can **measure** parts of the double push reliably on good
footage (where the recovering foot lands, how far the foot comes under the
body) and **show** them clearly, but it cannot yet **judge** a double push on
its own: detection reaches about 3 in 4, and automatic feedback mistakes
style for flaws. The practical result is a tool that puts a skater's
measured movement next to an elite example and explains what to look for,
leaving the judgement to the skater or coach. That matches what Phases 5 and
6 found: the methods work, and the limits come from single-camera footage.

## What This Sets Up Next
- **New footage:** filming developing skaters from in
  front, with the whole body in frame, would test whether the tool helps
  someone who is still learning.
- **Controlled ice footage with a fatigue measure** (the Phase 7 side track,
  not started) remains the clearest way to settle the fatigue question from
  Phases 5 and 6.
- Better views of the feet (a second camera at skate level, or higher
  resolution) would be needed to measure edges and the second-push twist.
