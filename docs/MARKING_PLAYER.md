# Marking Player (`mark_setdowns.py`)

> Built 10/29 for Phase 8b. A tool for marking events in video by hand,
> frame by frame, straight into the file the automatic measure is tested
> against.

## Why It Exists

Every measure in this project is checked against labels made by eye. For
Phase 8 the labels are the exact frames where a skate sets down on the ice,
so the stroke counter can be scored on timing (within 3 frames, 0.1 s).

The first way of doing this was watching a slowed clip with the frame
number burned in and writing each number down. That was slow, easy to
misread, and the numbers then had to be typed into a file by hand. The
player removes those steps: a key press records the frame.

## How It Works

- **Loads the clip into memory.** Every frame of the clip is read once and
  kept, so stepping forward and back is instant and always lands on the
  same frame (seeking in downloaded videos is unreliable).
- **Links each frame to the race video.** A clip frame is converted to its
  frame in the original broadcast (clip frame + the clip's start frame), so
  a mark lines up directly with the pose data and with the counter.
- **Controls playback.** Speed goes from 0.1x to 2x of real time. The usual
  way to mark: play at a comfortable speed, pause near a set-down, step
  frame by frame to the exact moment, then mark.
- **Shows its state.** The bottom bar shows the clip name, play/pause,
  speed, and how many marks have been made. "MARKED" appears on any
  marked frame, and "END OF CLIP" on the last frame.
- **Saves as it goes.** Marks are written into
  `stroke_rate/hand_counts.csv` (`setdown_frames` column) after every clip,
  so quitting partway loses nothing, and the next run starts from the
  first clip with no marks.
- **Keeps the labels independent.** The player never shows the automatic
  counter's answers, so the hand marks can't be influenced by the result
  they are used to test.

## Keys

| Key | Action |
|---|---|
| Space | play / pause |
| + / - | faster / slower (0.1x, 0.25x, 0.5x, 1x, 2x; starts at 0.5x) |
| D / A | one frame forward / back |
| M | mark a set-down at the current frame |
| U | undo the last mark |
| R | replay the clip from the start (marks kept) |
| N | save this clip's marks, go to the next clip |
| C | "can't tell" for this clip, go to the next |
| Q | save and quit |

## Usage

```
python -m mark_setdowns                 # every clip without marks yet
python -m mark_setdowns --set test      # only the test clips
python -m mark_setdowns --clip test_05  # one clip (to redo it)
```

Clips are made first by `stroke_clips.py` (one slowed clip per segment,
frame number burned in).

## Marking Rules (Phase 8)

- Mark the frame where the new skate **first touches the ice**; every
  set-down counts, either foot.
- Only mark set-downs seen happening: a stroke already under way when the
  clip starts is not marked, nor one about to land when it ends.
- If the skater can't be seen well enough, use "can't tell" instead of
  guessing.

## Checks on the Marks

After a pass, the gaps between marks are checked: strokes on a 5000 m
straightaway are usually 23-32 frames apart, so marks under about 15
frames apart are rechecked. On 10/29 this found double presses in the
first pass, and every clip was marked again from scratch. The same check
showed that most straightaways end on a set-down (the step into the
corner), which was confirmed on rewatching.

## Lessons From First Use (10/29)

- Marks on the last frame of a clip could be real or a key pressed after
  the clip stopped; the "END OF CLIP" message was added so the two can be
  told apart.
- Clip frames run at the broadcast's 30 fps, so one frame is 0.033 s; the
  3-frame window used for scoring is 0.1 s.

## Possible Uses Beyond Phase 8

The player records "this event happened at this frame". With a different
key list it could label other events the same way, for example the start
and end of each crossover in a corner, or push and glide phases, as ground
truth for any later measure.
