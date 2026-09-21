"""
build_centerline.py - regenerate the racing line and the drivable-area
mask straight from a track image.

You only need this if you swap in a different track. For track.jpg the
results are already baked into race_env.py and track_mask.png.

    py -m pip install numpy opencv-python scikit-image pillow
    py build_centerline.py track.jpg

It writes:
    track_mask.png    1-bit drivable area, loaded by race_env.TrackMap
    centerline.txt    paste this over CENTERLINE in race_env.py

How it works
    1. Keep pixels that are a neutral dark grey (the tarmac).
    2. Close/open to tidy kerbs and painted lines, keep the largest blob.
    3. Fill small holes (painted lines, pit walls) but not the infield.
    4. Skeletonise to the medial axis, prune every dead-end branch until
       only the closed loop is left.
    5. Walk the loop, smooth it, resample at a constant arc length.
"""

import math
import sys

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

WIDTH, HEIGHT = 965, 680
N_POINTS = 240
HOLE_MAX_AREA = 50_000     # anything bigger is treated as the infield


def road_mask(path):
    img = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.NEAREST)
    a = np.array(img).astype(int)
    hi, lo = a.max(2), a.min(2)

    mask = ((hi - lo < 45) & (lo >= 45) & (hi <= 140)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    biggest = int(np.argsort(-stats[1:, 4])[0]) + 1
    road = (lab == biggest).astype(np.uint8)

    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(1 - road, 4)
    for i in range(1, n2):
        if stats2[i, 4] < HOLE_MAX_AREA:
            road[lab2 == i] = 1
    return road


def loop_skeleton(road):
    sk = skeletonize(road > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(1000):
        neighbours = cv2.filter2D(sk, -1, kernel, borderType=cv2.BORDER_CONSTANT) - sk
        ends = (sk == 1) & (neighbours <= 1)
        if not ends.any():
            break
        sk[ends] = 0
    return sk


def order_loop(sk):
    pts = set(map(tuple, np.argwhere(sk > 0)))       # (y, x)
    n8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    cur = min(pts, key=lambda p: (p[0], p[1]))
    path, seen = [cur], {cur}
    while True:
        nxt = [(cur[0] + dy, cur[1] + dx) for dy, dx in n8]
        nxt = [p for p in nxt if p in pts and p not in seen]
        if not nxt:
            break
        nxt.sort(key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
        cur = nxt[0]
        seen.add(cur)
        path.append(cur)
    return [(float(x), float(y)) for y, x in path]


def smooth(points, k=12):
    n = len(points)
    out = []
    for i in range(n):
        sx = sy = 0.0
        for j in range(-k, k + 1):
            px, py = points[(i + j) % n]
            sx += px
            sy += py
        out.append((sx / (2 * k + 1), sy / (2 * k + 1)))
    return out


def resample(points, count):
    cum = [0.0]
    for i in range(len(points)):
        ax, ay = points[i]
        bx, by = points[(i + 1) % len(points)]
        cum.append(cum[-1] + math.hypot(bx - ax, by - ay))
    total = cum[-1]

    out, j = [], 0
    for i in range(count):
        target = total * i / count
        while cum[j + 1] < target:
            j += 1
        span = cum[j + 1] - cum[j]
        f = 0.0 if span == 0 else (target - cum[j]) / span
        ax, ay = points[j]
        bx, by = points[(j + 1) % len(points)]
        out.append((round(ax + (bx - ax) * f, 1), round(ay + (by - ay) * f, 1)))
    return out, total


def rotate_to_start(points, target=(300, 545)):
    """Put index 0 on the bottom straight so it reads as a start line."""
    i = min(
        range(len(points)),
        key=lambda k: (points[k][0] - target[0]) ** 2 + (points[k][1] - target[1]) ** 2,
    )
    return points[i:] + points[:i]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "track.jpg"

    road = road_mask(src)
    Image.fromarray((road * 255).astype(np.uint8)).convert("1").save("track_mask.png")

    centerline, length = resample(smooth(order_loop(loop_skeleton(road))), N_POINTS)
    centerline = rotate_to_start(centerline)

    off = sum(1 for x, y in centerline if not road[int(y), int(x)])
    print(f"road pixels      {int(road.sum())}")
    print(f"lap length       {length:.0f} px")
    print(f"points off road  {off}  (should be 0)")

    with open("centerline.txt", "w") as fh:
        fh.write("CENTERLINE = [\n")
        for i in range(0, len(centerline), 6):
            row = " ".join(f"({x}, {y})," for x, y in centerline[i:i + 6])
            fh.write(f"    {row}\n")
        fh.write("]\n")

    preview = np.array(
        Image.open(src).convert("RGB").resize((WIDTH, HEIGHT), Image.NEAREST)
    )
    for x, y in centerline:
        cv2.circle(preview, (int(x), int(y)), 3, (255, 0, 0), -1)
    Image.fromarray(preview).save("centerline_preview.png")
    print("wrote track_mask.png, centerline.txt, centerline_preview.png")


if __name__ == "__main__":
    main()
