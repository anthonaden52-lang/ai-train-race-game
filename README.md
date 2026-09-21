# AI Track Racer

2D Pygame racer on `track.jpg`, with a training environment attached.

## Run

    py -m pip install -r requirements.txt
    py race_game.py

Sanity check without a window:

    py test_env.py

## Controls

| key | action |
|---|---|
| W / Up | accelerate |
| S / Down | brake, then reverse |
| A / Left, D / Right | steer |
| T | toggle the demo AI driver |
| C | show / hide the racing line |
| M | show / hide the drivable-area mask |
| R | reset to the start line |
| P | pause |
| Esc | quit |

## What was wrong

The game printed `CRASHED - press R` on the first frame, before any key
was pressed.

`CENTERLINE_BASE` in the old `race_env.py` was hand-written and did not
line up with the track image. Its first point, `(210, 555)`, is the
start position, and that pixel is sand run-off, not tarmac — so
`_on_road` returned False on frame 1 and the game was over immediately.
The yellow line in your screenshot shows the same problem: it cuts
across grass and buildings all the way round.

Fixes:

* The centerline is now derived from `track.jpg` itself — grey road
  pixels, largest connected blob, medial axis, resampled to 240 points
  ~10 px apart. All 240 points sit on tarmac. `build_centerline.py`
  regenerates it if you swap the track.
* Collision uses `track_mask.png`, a 1-bit drivable-area map built by
  the same script, instead of re-thresholding colours each frame. The
  car's whole footprint is sampled, and it only counts as off-track when
  more than half of it has left the road.
* Progress was `nearest_index / len(centerline)`, which jumped around
  wherever two parts of the circuit run close together. It's now arc
  length accumulated from a windowed nearest-point search, so laps,
  reward and the percentage readout are all monotonic.
* Steering at full lock gave a ~78 px turn radius; the hairpins here are
  about 40 px. The turn rate and its speed scaling were retuned so the
  track is actually drivable.
* `requirements.txt` had two shell commands pasted above the dependency,
  so `pip install -r requirements.txt` failed.
* Start line moved to the bottom straight, and the game now keeps lap
  and best-lap times.

The demo AI (`T`) laps in about 14.0-14.6 s. That's your baseline.

## Training API

```python
env = RacingEnv(track_surface)
obs = env.reset()
obs, reward, done, info = env.step(steer, throttle)
```

`steer` and `throttle` are both in `[-1, 1]`; negative throttle brakes.

Observation, 8 floats, all roughly in `[-1, 1]`:

    [lateral_error, heading_error, speed, c1, c2, c3, c4, c5]

`lateral_error` is signed distance from the centerline over the half
road width, `heading_error` is the angle to the track over pi, `speed`
is normalised, and `c1..c5` are the track's curvature 1/3/5/7/9
waypoints ahead. The curvature terms are what let an agent brake before
a corner instead of after it.

Reward per step:

    +0.05 per pixel of forward progress
    -0.002 per pixel of distance from the centerline
    -0.01 flat, so faster laps score higher
    -20   off track
    +100  lap complete

Episodes end on a crash, on a completed lap, after `MAX_STEPS`, or after
`STALL_STEPS` frames without progress. Set `env.terminate_on_lap = False`
to keep circulating — `race_game.py` does exactly that.

Car state lives on the env: `env.x`, `env.y`, `env.heading`, `env.speed`,
plus `env.laps`, `env.distance` and `env.progress`.

The old `env.get_state(x, y, heading, speed)` still works if you had
anything built against it.

## Swapping the track

    py -m pip install numpy opencv-python scikit-image pillow
    py build_centerline.py my_track.png

Writes `track_mask.png`, `centerline.txt` (paste over `CENTERLINE` in
`race_env.py`) and `centerline_preview.png` so you can check the line
before trusting it. It assumes grey tarmac on a non-grey background; if
your track uses different colours, adjust the threshold in `road_mask`.
