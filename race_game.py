"""
race_game.py - playable front end for the AI Track Racer.

Controls
    W / Up      accelerate
    S / Down    brake and reverse
    A / Left    steer left
    D / Right   steer right
    T           toggle the demo AI driver
    C           show / hide the racing line
    M           show / hide the drivable-area mask (debug)
    R           reset to the start line
    P           pause
    Esc         quit
"""

import math
import os

import pygame

from race_env import RacingEnv, TRACK_SIZE

os.chdir(os.path.dirname(os.path.abspath(__file__)))

pygame.init()

WIDTH, HEIGHT = TRACK_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Track Racer")
clock = pygame.time.Clock()

track_img = pygame.transform.scale(
    pygame.image.load("track.jpg").convert(), (WIDTH, HEIGHT)
)

env = RacingEnv(track_img)
env.terminate_on_lap = False      # keep driving after the flag
env.MAX_STEPS = 10 ** 9           # no episode limit while playing
env.STALL_STEPS = 10 ** 9

font = pygame.font.SysFont("Consolas", 20)
big_font = pygame.font.SysFont("Consolas", 40, bold=True)

ai_mode = False
paused = False
show_line = True
show_mask = False
crashed = False

lap_timer = 0.0
last_lap = None
best_lap = None

mask_overlay = None


def build_mask_overlay():
    """Blue tint over everything the env considers drivable. Built once,
    the first time you press M, because it walks the whole bitmap."""
    surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for py in range(0, HEIGHT, 2):
        for px in range(0, WIDTH, 2):
            if env.map.at(px, py):
                surf.fill((0, 160, 255, 70), (px, py, 2, 2))
    return surf


def reset():
    global crashed, lap_timer
    env.reset()
    crashed = False
    lap_timer = 0.0


def ai_controls():
    """
    Waypoint-following demo driver. It aims a few points down the
    centerline, nudges itself back towards the middle, and lifts off
    before corners. Good for about a 14.5 s lap; beatable by hand and a
    fair baseline for a trained agent.
    """
    idx = env.index
    look = 5 + int(4 * env.speed / env.MAX_SPEED)
    tx, ty = env.centerline[(idx + look) % env.n]

    desired = math.atan2(ty - env.y, tx - env.x)
    err = math.atan2(math.sin(desired - env.heading), math.cos(desired - env.heading))
    lateral = env.lateral_error(env.x, env.y, idx)
    steer = max(-1.0, min(1.0, err * 3.2 - lateral * 0.012))

    curve = max(abs(env.curvature_ahead(idx, s)) for s in (1, 3, 5, 7, 9, 11))
    target_speed = env.MAX_SPEED * (1.0 - 0.55 * min(1.0, curve))
    throttle = 1.0 if env.speed < target_speed else -0.7
    return steer, throttle


def human_controls():
    keys = pygame.key.get_pressed()
    steer = 0.0
    throttle = 0.0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer -= 1.0
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer += 1.0
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        throttle += 1.0
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        throttle -= 1.0
    return steer, throttle


def draw_car(surface, px, py, angle, color=(40, 110, 230)):
    car = pygame.Surface((34, 18), pygame.SRCALPHA)
    pygame.draw.rect(car, color, (0, 0, 34, 18), border_radius=5)
    pygame.draw.rect(car, (25, 25, 30), (20, 3, 9, 12), border_radius=2)
    pygame.draw.rect(car, (245, 245, 245), (1, 5, 4, 8), border_radius=1)
    rotated = pygame.transform.rotate(car, -math.degrees(angle))
    surface.blit(rotated, rotated.get_rect(center=(px, py)))


def draw_racing_line(surface):
    pts = [(int(x), int(y)) for x, y in env.centerline]
    pygame.draw.lines(surface, (255, 220, 50), True, pts, 2)
    sx, sy = env.centerline[0]
    a = env.path_angle(0)
    nx, ny = -math.sin(a) * 30, math.cos(a) * 30
    pygame.draw.line(
        surface, (255, 255, 255),
        (sx - nx, sy - ny), (sx + nx, sy + ny), 4,
    )


def fmt(t):
    return "--.---" if t is None else f"{t:6.3f}"


reset()
running = True
while running:
    dt = min(clock.tick(60) / 1000.0, 0.05)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                reset()
            elif event.key == pygame.K_t:
                ai_mode = not ai_mode
                reset()
            elif event.key == pygame.K_p:
                paused = not paused
            elif event.key == pygame.K_c:
                show_line = not show_line
            elif event.key == pygame.K_m:
                show_mask = not show_mask
                if show_mask and mask_overlay is None:
                    mask_overlay = build_mask_overlay()

    if not paused and not crashed:
        steer, throttle = ai_controls() if ai_mode else human_controls()
        _obs, _reward, _done, info = env.step(steer, throttle, dt)

        lap_timer += dt
        crashed = info["crashed"]

        if info["lap_complete"]:
            last_lap = lap_timer
            if best_lap is None or last_lap < best_lap:
                best_lap = last_lap
            lap_timer = 0.0

    screen.blit(track_img, (0, 0))
    if show_mask and mask_overlay is not None:
        screen.blit(mask_overlay, (0, 0))
    if show_line:
        draw_racing_line(screen)
    draw_car(screen, env.x, env.y, env.heading)

    hud = [
        f"{'AI DEMO' if ai_mode else 'MANUAL':<8} speed {env.speed:4.1f}",
        f"lap {env.laps + 1}   {env.progress * 100:5.1f}%   {lap_timer:6.3f}",
        f"last {fmt(last_lap)}   best {fmt(best_lap)}",
        "W/S throttle  A/D steer  T ai  C line  M mask  R reset  P pause",
    ]
    panel = pygame.Surface((560, 26 * len(hud) + 10), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 190))
    screen.blit(panel, (8, 8))
    for i, line in enumerate(hud):
        screen.blit(font.render(line, True, (20, 20, 20)), (16, 14 + i * 26))

    if crashed:
        txt = big_font.render("OFF TRACK - press R", True, (210, 30, 30))
        rect = txt.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        back = pygame.Surface((rect.width + 24, rect.height + 16), pygame.SRCALPHA)
        back.fill((255, 255, 255, 210))
        screen.blit(back, (rect.x - 12, rect.y - 8))
        screen.blit(txt, rect)

    if paused:
        txt = big_font.render("PAUSED", True, (40, 40, 40))
        screen.blit(txt, txt.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60)))

    pygame.display.flip()

pygame.quit()
