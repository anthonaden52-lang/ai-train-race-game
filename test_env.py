"""
test_env.py - run the demo driver with no window, as a sanity check.

    py test_env.py

Expect three clean laps of roughly 14-15 seconds each. If this fails,
the centerline or the mask no longer matches track.jpg.
"""

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pygame  # noqa: E402

from race_env import RacingEnv, TRACK_SIZE  # noqa: E402

pygame.init()
pygame.display.set_mode((1, 1))

track = pygame.transform.scale(pygame.image.load("track.jpg").convert(), TRACK_SIZE)
env = RacingEnv(track)
env.terminate_on_lap = False
env.MAX_STEPS = 20000
env.STALL_STEPS = 600

print(f"lap length      {env.lap_length:.0f} px")
print(f"start           ({env.start_x:.0f}, {env.start_y:.0f})")
print(f"start on road   {env.map.car_on_road(env.x, env.y, env.heading)}")
off = [p for p in env.centerline if not env.map.at(*p)]
print(f"line off road   {len(off)}  (should be 0)")

env.reset()
laps, step, last = [], 0, 0
while len(laps) < 3 and step < 6000:
    idx = env.index
    look = 5 + int(4 * env.speed / env.MAX_SPEED)
    tx, ty = env.centerline[(idx + look) % env.n]
    desired = math.atan2(ty - env.y, tx - env.x)
    err = math.atan2(math.sin(desired - env.heading), math.cos(desired - env.heading))
    lateral = env.lateral_error(env.x, env.y, idx)
    steer = max(-1.0, min(1.0, err * 3.2 - lateral * 0.012))

    curve = max(abs(env.curvature_ahead(idx, s)) for s in (1, 3, 5, 7, 9, 11))
    throttle = 1.0 if env.speed < env.MAX_SPEED * (1.0 - 0.55 * min(1.0, curve)) else -0.7

    _obs, _reward, done, info = env.step(steer, throttle)
    step += 1

    if info["lap_complete"]:
        laps.append((step - last) / 60.0)
        last = step
    if info["crashed"]:
        print(f"CRASHED at {info['progress'] * 100:.1f}% "
              f"near ({env.x:.0f}, {env.y:.0f})")
        break

for i, t in enumerate(laps, 1):
    print(f"lap {i}          {t:.3f} s")
print("OK" if len(laps) == 3 else "FAILED")
pygame.quit()
