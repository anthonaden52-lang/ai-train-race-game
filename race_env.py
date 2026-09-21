"""
race_env.py - environment for the AI Track Racer.

What changed vs. the first version
----------------------------------
The old CENTERLINE_BASE was hand-guessed and did not match track.jpg.
Its very first point, (210, 555), sits on the sand run-off, so the car
was "off road" on frame 1 and the game showed CRASHED before you could
touch a key.

The centerline below is generated from track.jpg itself (see
build_centerline.py): the grey road pixels are thresholded, the biggest
connected blob is kept, and its medial axis is traced and resampled at a
constant arc length. Index 0 is the start/finish line on the bottom
straight, and the direction of travel is left-to-right from there.

Public API
----------
    env = RacingEnv(track_surface)
    obs = env.reset()
    obs, reward, done, info = env.step(steer, throttle)

    steer    in [-1, 1]   (-1 = full left)
    throttle in [-1, 1]   (negative = brake / reverse)

Observation (all roughly in [-1, 1]):
    [ lateral_error, heading_error, speed, c1, c2, c3, c4, c5 ]

    lateral_error  signed distance from the centerline / HALF_WIDTH
    heading_error  angle between the car and the track / pi
    speed          speed / MAX_SPEED
    c1..c5         curvature of the track 15/30/45/60/75 px ahead

Car state lives on the env (env.x, env.y, env.heading, env.speed) so a
trainer only has to touch reset() and step().
"""

import math
import os

import pygame

# --------------------------------------------------------------------
# Track geometry
# --------------------------------------------------------------------

TRACK_SIZE = (965, 680)          # the centerline is in these coordinates
HALF_WIDTH = 26.0               # approx. half the road width, in px
MASK_FILE = "track_mask.png"     # 1-bit drivable-area map (white = road)

# Generated from track.jpg. 240 points, ~10 px apart, closed loop.
CENTERLINE = [
    (298.9, 534.0), (308.8, 534.0), (318.8, 534.0), (328.8, 534.0), (338.7, 534.0), (348.7, 534.0),
    (358.7, 534.0), (368.7, 534.0), (378.6, 534.0), (388.6, 534.0), (398.6, 534.0), (408.5, 534.0),
    (418.5, 534.0), (428.5, 534.0), (438.5, 534.0), (448.4, 534.0), (458.4, 534.0), (468.4, 534.0),
    (478.3, 533.6), (488.2, 532.5), (498.0, 530.6), (507.7, 528.4), (517.4, 525.8), (526.8, 522.7),
    (536.3, 519.7), (546.1, 517.6), (556.0, 517.0), (566.0, 516.4), (575.9, 515.2), (585.8, 514.1),
    (595.5, 512.1), (602.9, 505.7), (607.2, 496.7), (611.2, 487.6), (615.7, 478.7), (620.5, 470.0),
    (627.9, 463.4), (637.2, 460.0), (647.0, 458.2), (657.0, 457.4), (666.9, 457.0), (676.9, 457.0),
    (686.9, 457.0), (696.8, 457.0), (706.8, 457.0), (716.8, 457.0), (726.7, 457.0), (736.7, 457.0),
    (746.7, 457.0), (756.7, 457.0), (766.6, 457.0), (776.6, 457.0), (786.6, 457.0), (796.5, 456.9),
    (806.4, 455.7), (816.3, 454.5), (826.0, 452.2), (835.3, 448.6), (842.7, 442.0), (847.2, 433.2),
    (848.9, 423.4), (848.8, 413.4), (847.5, 403.5), (845.6, 393.8), (842.8, 384.2), (836.0, 377.2),
    (826.5, 374.3), (816.9, 371.8), (807.2, 369.4), (797.6, 366.7), (788.0, 364.2), (778.4, 361.6),
    (768.7, 359.3), (759.0, 356.8), (749.4, 354.5), (739.7, 352.1), (730.0, 349.6), (720.4, 347.3),
    (710.7, 344.7), (701.0, 342.6), (691.8, 338.8), (684.1, 332.6), (677.8, 324.9), (672.5, 316.4),
    (669.1, 307.2), (671.3, 297.5), (676.8, 289.3), (683.9, 282.3), (692.7, 277.6), (702.2, 274.9),
    (712.0, 272.9), (721.8, 271.3), (731.5, 268.7), (740.7, 265.1), (749.9, 261.2), (758.7, 256.6),
    (767.2, 251.5), (773.9, 244.2), (778.4, 235.3), (781.3, 225.8), (783.2, 216.0), (784.0, 206.1),
    (784.0, 196.1), (782.5, 186.3), (779.7, 176.7), (774.7, 168.1), (768.7, 160.2), (761.6, 153.2),
    (753.3, 147.7), (744.0, 144.2), (734.2, 142.7), (724.4, 140.9), (714.5, 139.7), (704.5, 139.0),
    (694.6, 139.0), (684.6, 139.0), (674.6, 139.0), (664.7, 139.7), (654.8, 140.9), (644.9, 142.2),
    (635.0, 143.1), (625.1, 143.9), (615.1, 144.0), (605.1, 144.0), (595.1, 144.0), (585.2, 144.0),
    (575.2, 144.0), (565.2, 144.1), (555.4, 145.8), (545.6, 147.7), (535.8, 149.0), (525.8, 149.0),
    (515.8, 149.0), (505.9, 149.0), (495.9, 149.0), (486.0, 149.6), (476.2, 151.6), (466.4, 153.5),
    (456.4, 154.0), (446.5, 154.0), (436.5, 154.0), (426.5, 154.0), (416.6, 154.0), (406.6, 154.0),
    (396.7, 155.5), (387.0, 157.4), (377.1, 159.0), (367.2, 159.0), (357.2, 159.0), (347.2, 159.0),
    (337.2, 159.0), (327.3, 159.4), (317.5, 161.0), (307.7, 163.0), (297.9, 164.9), (288.0, 166.1),
    (278.3, 168.2), (270.5, 174.3), (266.4, 183.3), (264.3, 193.0), (263.8, 203.0), (264.4, 212.9),
    (266.1, 222.7), (269.3, 232.2), (273.9, 241.0), (281.2, 247.6), (289.9, 252.5), (299.0, 256.5),
    (308.2, 260.4), (317.7, 263.5), (327.1, 266.7), (336.3, 270.5), (345.9, 273.1), (355.5, 275.7),
    (365.2, 278.1), (374.6, 281.1), (384.1, 284.2), (391.8, 290.4), (397.1, 298.8), (400.5, 308.2),
    (403.9, 317.6), (404.4, 327.5), (402.0, 337.1), (397.4, 345.8), (389.2, 351.2), (379.3, 352.0),
    (369.3, 352.0), (359.4, 352.0), (349.4, 352.0), (339.4, 352.0), (329.5, 352.0), (319.5, 352.0),
    (309.5, 352.0), (299.5, 352.0), (289.6, 352.0), (279.6, 352.0), (269.6, 352.0), (259.7, 352.2),
    (249.7, 353.0), (239.8, 353.8), (229.8, 354.0), (219.9, 354.0), (210.0, 354.9), (200.6, 358.2),
    (192.5, 364.0), (186.6, 371.9), (184.4, 381.6), (183.7, 391.5), (181.9, 401.3), (178.2, 410.5),
    (172.2, 418.4), (164.3, 424.4), (155.4, 429.0), (146.1, 432.6), (137.4, 437.4), (129.2, 443.0),
    (122.7, 450.5), (118.3, 459.4), (115.5, 469.0), (114.2, 478.9), (115.3, 488.8), (117.1, 498.6),
    (123.0, 506.4), (132.3, 509.9), (142.1, 511.4), (151.9, 512.9), (161.8, 514.2), (171.6, 516.1),
    (181.1, 519.0), (190.6, 522.2), (200.0, 525.4), (209.7, 527.9), (219.3, 530.3), (229.2, 531.8),
    (239.1, 533.1), (249.0, 533.9), (259.0, 534.0), (268.9, 534.0), (278.9, 534.0), (288.9, 534.0),
]

LAP_LENGTH = 0.0  # filled in below


def _seg_lengths(points):
    out = []
    for i in range(len(points)):
        ax, ay = points[i]
        bx, by = points[(i + 1) % len(points)]
        out.append(math.hypot(bx - ax, by - ay))
    return out


SEG_LENGTHS = _seg_lengths(CENTERLINE)
LAP_LENGTH = sum(SEG_LENGTHS)

# Cumulative arc length at each waypoint.
CUM_LENGTH = [0.0]
for _s in SEG_LENGTHS[:-1]:
    CUM_LENGTH.append(CUM_LENGTH[-1] + _s)


def angle_diff(a, b):
    """Shortest signed angle from b to a, in [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


# --------------------------------------------------------------------
# Drivable area
# --------------------------------------------------------------------

class TrackMap:
    """
    Answers "is this pixel on the road?".

    Prefers track_mask.png, which build_centerline.py writes next to the
    track image. If that file is missing it falls back to sampling the
    track colours directly, which works but is noisier at the kerbs.
    """

    def __init__(self, track_surface, mask_path=MASK_FILE):
        self.width = track_surface.get_width()
        self.height = track_surface.get_height()
        self.surface = track_surface
        self.mask = None

        here = os.path.dirname(os.path.abspath(__file__))
        path = mask_path if os.path.isabs(mask_path) else os.path.join(here, mask_path)
        if os.path.exists(path):
            img = pygame.image.load(path)
            try:
                img = img.convert()
            except pygame.error:
                pass  # no display yet; the raw surface works fine
            if img.get_size() != (self.width, self.height):
                img = pygame.transform.scale(img, (self.width, self.height))
            self.mask = pygame.mask.from_threshold(
                img, (255, 255, 255), (100, 100, 100, 255)
            )

    def _colour_is_road(self, px, py):
        r, g, b = self.surface.get_at((px, py))[:3]
        lo, hi = min(r, g, b), max(r, g, b)
        if hi - lo > 45:          # coloured: grass, sand, trees
            return False
        if not (45 <= lo and hi <= 140):
            return False
        if r > g + 12:            # dark red tyre barriers
            return False
        return True

    def at(self, x, y):
        px, py = int(x), int(y)
        if not (0 <= px < self.width and 0 <= py < self.height):
            return False
        if self.mask is not None:
            return bool(self.mask.get_at((px, py)))
        return self._colour_is_road(px, py)

    def car_on_road(self, x, y, heading, half_len=17.0, half_wid=9.0):
        """True while at least half of the car's footprint is on tarmac."""
        ch, sh = math.cos(heading), math.sin(heading)
        hits = 0
        total = 0
        for fwd in (half_len, 0.0, -half_len):
            for side in (half_wid, 0.0, -half_wid):
                px = x + ch * fwd - sh * side
                py = y + sh * fwd + ch * side
                total += 1
                if self.at(px, py):
                    hits += 1
        return hits * 2 >= total


# --------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------

class RacingEnv:
    # Physics
    MAX_SPEED = 7.0          # in "speed units"; 1 unit = 30 px/s
    PIXELS_PER_UNIT = 30.0
    ACCELERATION = 8.0
    BRAKE = 12.0
    DRAG = 2.0
    TURN_RATE = 3.6          # rad/s at full lock

    # Episode
    MAX_STEPS = 6000
    STALL_STEPS = 240        # give up if no progress for this many steps

    # Training usually wants an episode to end at the flag; the playable
    # game sets this to False so you can keep circulating.
    terminate_on_lap = True

    def __init__(self, track_surface, centerline=None):
        self.track = track_surface
        self.width = track_surface.get_width()
        self.height = track_surface.get_height()

        self.centerline = list(centerline) if centerline else list(CENTERLINE)
        self.n = len(self.centerline)
        self.seg = _seg_lengths(self.centerline)
        self.lap_length = sum(self.seg)
        self.cum = [0.0]
        for s in self.seg[:-1]:
            self.cum.append(self.cum[-1] + s)

        self.map = TrackMap(track_surface)

        self.start_x, self.start_y = self.centerline[0]
        self.start_heading = self.path_angle(0)

        self.reset()

    # ---------------- geometry helpers ----------------

    def path_angle(self, i):
        ax, ay = self.centerline[i]
        bx, by = self.centerline[(i + 1) % self.n]
        return math.atan2(by - ay, bx - ax)

    def nearest_index(self, x, y, around=None, window=25):
        """
        Nearest waypoint. Searching a window around the last known index
        stops the car from "teleporting" onto a different part of the
        track where two sections run close together.
        """
        if around is None:
            rng = range(self.n)
        else:
            rng = (i % self.n for i in range(around - window // 3, around + window))

        best_i, best_d2 = 0, float("inf")
        for i in rng:
            px, py = self.centerline[i]
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return best_i, math.sqrt(best_d2)

    def lateral_error(self, x, y, idx):
        """Signed distance to the centerline. Positive = right of the line."""
        cx, cy = self.centerline[idx]
        nx, ny = self.centerline[(idx + 1) % self.n]
        tx, ty = nx - cx, ny - cy
        length = max(1e-6, math.hypot(tx, ty))
        tx, ty = tx / length, ty / length
        return tx * (y - cy) - ty * (x - cx)

    def curvature_ahead(self, idx, steps):
        a = self.path_angle((idx + steps) % self.n)
        b = self.path_angle((idx + steps + 3) % self.n)
        return angle_diff(b, a) / (math.pi / 2)

    # ---------------- gym-style API ----------------

    def reset(self):
        self.x, self.y = self.start_x, self.start_y
        self.heading = self.start_heading
        self.speed = 0.0

        self.index = 0
        self.distance = 0.0        # arc length travelled, in px
        self.laps = 0
        self.steps = 0
        self.stalled = 0
        self.best_distance = 0.0
        self.crashed = False
        self.lap_complete = False
        return self.observe()

    @property
    def progress(self):
        """Fraction of the current lap completed, 0..1."""
        return (self.distance % self.lap_length) / self.lap_length

    def observe(self):
        idx = self.index
        lat = self.lateral_error(self.x, self.y, idx)
        head_err = angle_diff(self.heading, self.path_angle(idx))
        return [
            max(-2.0, min(2.0, lat / HALF_WIDTH)),
            head_err / math.pi,
            self.speed / self.MAX_SPEED,
            self.curvature_ahead(idx, 1),
            self.curvature_ahead(idx, 3),
            self.curvature_ahead(idx, 5),
            self.curvature_ahead(idx, 7),
            self.curvature_ahead(idx, 9),
        ]

    def step(self, steer, throttle, dt=1 / 60.0):
        steer = max(-1.0, min(1.0, float(steer)))
        throttle = max(-1.0, min(1.0, float(throttle)))
        dt = max(1e-4, min(0.05, dt))

        # --- physics ---
        if throttle >= 0.0:
            self.speed += throttle * self.ACCELERATION * dt
        else:
            self.speed += throttle * self.BRAKE * dt
        self.speed -= self.DRAG * dt
        self.speed = max(0.0, min(self.MAX_SPEED, self.speed))

        grip = 0.55 + 0.45 * (self.speed / self.MAX_SPEED)
        self.heading += steer * self.TURN_RATE * grip * dt

        self.x += math.cos(self.heading) * self.speed * self.PIXELS_PER_UNIT * dt
        self.y += math.sin(self.heading) * self.speed * self.PIXELS_PER_UNIT * dt

        # --- progress along the centerline ---
        new_idx, dist_to_line = self.nearest_index(self.x, self.y, around=self.index)
        delta_i = (new_idx - self.index) % self.n
        if delta_i > self.n // 2:
            delta_i -= self.n          # went backwards

        advance = 0.0
        i = self.index
        for _ in range(abs(delta_i)):
            if delta_i > 0:
                advance += self.seg[i % self.n]
                i += 1
            else:
                i -= 1
                advance -= self.seg[i % self.n]
        self.index = new_idx
        self.distance += advance

        self.lap_complete = self.distance >= self.lap_length * (self.laps + 1)
        if self.lap_complete:
            self.laps += 1

        # --- termination ---
        self.crashed = not self.map.car_on_road(self.x, self.y, self.heading)
        self.steps += 1

        if self.distance > self.best_distance + 1.0:
            self.best_distance = self.distance
            self.stalled = 0
        else:
            self.stalled += 1

        timeout = self.steps >= self.MAX_STEPS or self.stalled >= self.STALL_STEPS
        done = self.crashed or timeout or (self.lap_complete and self.terminate_on_lap)

        # --- reward ---
        reward = advance * 0.05                 # ~1.0 per 20 px of track
        reward -= abs(dist_to_line) * 0.002     # stay near the middle
        reward -= 0.01                          # small cost per step: be quick
        if self.crashed:
            reward -= 20.0
        if self.lap_complete:
            reward += 100.0

        info = {
            "crashed": self.crashed,
            "lap_complete": self.lap_complete,
            "laps": self.laps,
            "nearest_index": self.index,
            "distance_from_center": dist_to_line,
            "progress": self.progress,
            "timeout": timeout,
        }
        return self.observe(), reward, done, info

    # ---------------- compatibility shim ----------------

    def get_state(self, x, y, heading, speed):
        """Old signature, kept so existing scripts keep running."""
        idx, _ = self.nearest_index(x, y, around=self.index)
        return [
            self.lateral_error(x, y, idx),
            angle_diff(heading, self.path_angle(idx)),
            speed,
            self.progress,
        ]
