"""Pygame front end: webcam feed on the left, the 'escape room console' on the right."""
import math
import time

import cv2
import pygame

import config
from game.engine import GameEngine, State

# palette
BG = (14, 17, 26)
PANEL = (22, 27, 40)
CARD = (31, 38, 56)
TEXT = (232, 236, 245)
MUTED = (140, 150, 172)
ACCENT = (255, 184, 64)     # amber "escape room" light
GOOD = (72, 214, 140)
BAD = (240, 84, 96)
LOCKED = (70, 80, 102)


class App:
    def __init__(self, source, use_yolo=True, seed=None, demo=False, levels=config.NUM_LEVELS):
        pygame.init()
        pygame.display.set_caption("CV Escape Room")
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.SysFont("segoeui,arial,helvetica", 54, bold=True)
        self.f_big = pygame.font.SysFont("segoeui,arial,helvetica", 34, bold=True)
        self.f_med = pygame.font.SysFont("segoeui,arial,helvetica", 22, bold=True)
        self.f_txt = pygame.font.SysFont("segoeui,arial,helvetica", 19)
        self.f_small = pygame.font.SysFont("segoeui,arial,helvetica", 15)

        self._splash("Starting camera and loading AI models...")
        self.cap = _open_camera(source)
        from vision.perception import Perception  # imported here so the splash shows first
        self.perception = Perception(use_yolo=use_yolo)
        self.engine = GameEngine(n_levels=levels, seed=seed, demo=demo, use_objects=use_yolo)
        self.show_overlay = True
        self.cam_rect = self._camera_rect()
        self.fps = 0.0

    # ------------------------------------------------------------------ main loop
    def run(self):
        running = True
        while running:
            now = time.monotonic()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._on_key(event.key, now)

            ok, frame = self.cap.read()
            if ok:
                if config.MIRROR:
                    frame = cv2.flip(frame, 1)
                obs, annotated = self.perception.process(frame, draw=self.show_overlay,
                                                         **self.engine.requirements())
                self.engine.update(obs, now)
            else:
                obs, annotated = None, None

            self._draw(annotated, obs, now)
            pygame.display.flip()
            self.clock.tick(config.FPS)
            self.fps = self.clock.get_fps()

        self.cap.release()
        self.perception.close()
        pygame.quit()

    def _on_key(self, key, now) -> bool:
        st = self.engine.state
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_SPACE and st in (State.MENU, State.GAME_OVER, State.VICTORY):
            self.engine.start(now)
        elif key == pygame.K_h:
            self.engine.use_hint(now)
        elif key == pygame.K_s:
            self.engine.skip(now)
        elif key == pygame.K_d:
            self.show_overlay = not self.show_overlay
        return True

    # ------------------------------------------------------------------ drawing
    def _camera_rect(self):
        area_w = config.WINDOW_WIDTH - config.SIDEBAR_WIDTH
        area_h = config.WINDOW_HEIGHT
        w = area_w - 40
        h = int(w * 3 / 4)
        if h > area_h - 40:
            h = area_h - 40
            w = int(h * 4 / 3)
        return pygame.Rect((area_w - w) // 2, (area_h - h) // 2, w, h)

    def _draw(self, frame, obs, now):
        self.screen.fill(BG)
        r = self.cam_rect
        if frame is not None:
            rgb = cv2.cvtColor(cv2.resize(frame, (r.w, r.h)), cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(rgb.tobytes(), (r.w, r.h), "RGB")
            self.screen.blit(surf, r.topleft)
        else:
            pygame.draw.rect(self.screen, CARD, r)
            self._center_text("Camera not available - check config.CAMERA_INDEX", self.f_med, BAD, r.center)
        border = GOOD if self.engine.state == State.LEVEL_CLEAR else ACCENT
        pygame.draw.rect(self.screen, border, r.inflate(6, 6), 3, border_radius=6)

        self._sidebar(obs, now)

        st = self.engine.state
        if st == State.MENU:
            self._menu()
        elif st == State.LEVEL_CLEAR:
            self._level_clear(now)
        elif st in (State.GAME_OVER, State.VICTORY):
            self._end_screen()

        msg = self.engine.current_message(now)
        if msg and st == State.PLAYING:
            self._toast(msg)
        self.screen.blit(self.f_small.render(f"{self.fps:4.1f} fps", True, MUTED), (r.x + 8, r.bottom - 22))

    def _sidebar(self, obs, now):
        e = self.engine
        x0 = config.WINDOW_WIDTH - config.SIDEBAR_WIDTH
        pygame.draw.rect(self.screen, PANEL, (x0, 0, config.SIDEBAR_WIDTH, config.WINDOW_HEIGHT))
        x, y, w = x0 + 22, 20, config.SIDEBAR_WIDTH - 44

        self.screen.blit(self.f_small.render("AI ESCAPE ROOM", True, ACCENT), (x, y))
        y += 22
        total = len(e.levels) or e.n_levels
        lvl = min(e.level + 1, total)
        self.screen.blit(self.f_big.render(f"Level {lvl} / {total}", True, TEXT), (x, y))
        y += 50
        # padlocks: one per level
        for i in range(total):
            if i < e.level or (i == e.level and e.state in (State.LEVEL_CLEAR, State.VICTORY)):
                col, open_ = GOOD, True
            elif i == e.level and e.state == State.PLAYING:
                col, open_ = ACCENT, False
            else:
                col, open_ = LOCKED, False
            _padlock(self.screen, x + i * 44, y, col, open_)
        y += 50

        c = e.challenge
        if c is not None and e.state in (State.PLAYING, State.LEVEL_CLEAR):
            y = self._card_title("YOUR TASK", x, y)
            y = self._wrap(c.description, self.f_med, TEXT, x, y, w) + 6
            for text, state in c.steps():
                col = {"done": GOOD, "active": ACCENT, "todo": MUTED}[state]
                mark = {"done": "[x]", "active": "[>]", "todo": "[ ]"}[state]
                y = self._wrap(f"{mark} {text}", self.f_small, col, x, y, w) + 2
            if hasattr(c, "time_left_for_step") and c.index > 0 and e.state == State.PLAYING:
                left = c.time_left_for_step(now)
                y = self._wrap(f"Next step within {left:0.0f}s", self.f_small, BAD if left < 4 else MUTED, x, y, w)
            y += 10

            # verification bar
            label = "Verified!" if e.state == State.LEVEL_CLEAR else "AI verifying..."
            self.screen.blit(self.f_small.render(label, True, MUTED), (x, y))
            y += 20
            _bar(self.screen, x, y, w, 14, 1.0 if e.state == State.LEVEL_CLEAR else c.progress_frac, GOOD)
            y += 30

            # timer (frozen while the celebration screen is up)
            t = e.time_left(now) if e.state == State.PLAYING else e.cleared_with
            tcol = BAD if t < 10 else TEXT
            self.screen.blit(self.f_big.render(f"{int(t // 60)}:{int(t % 60):02d}", True, tcol), (x, y))
            _bar(self.screen, x + 110, y + 18, w - 110, 8, t / e.level_time, tcol)
            y += 50

        # score + lives
        self.screen.blit(self.f_med.render(f"Score {e.score}", True, TEXT), (x, y))
        for i in range(e.max_lives):
            _heart(self.screen, x + w - 20 - i * 28, y + 14, BAD if i < e.lives else LOCKED)
        y += 30
        self.screen.blit(self.f_small.render(f"Best {e.highscore}", True, MUTED), (x, y))
        y += 30

        if e.show_hint and c is not None and e.state == State.PLAYING:
            y = self._card_title("HINT", x, y)
            y = self._wrap(c.hint, self.f_small, ACCENT, x, y, w) + 12

        # what the AI currently sees: great for explaining the system during a demo
        y = self._card_title("AI SEES", x, y)
        for line in (obs.summary() if obs else ["(no camera)"])[:5]:
            y = self._wrap(line, self.f_small, TEXT, x, y, w)

        foot = "SPACE start   H hint   S skip   D overlay   ESC quit"
        self.screen.blit(self.f_small.render(foot, True, MUTED), (x, config.WINDOW_HEIGHT - 30))

    def _menu(self):
        r = self.cam_rect
        _dim(self.screen, r, 190)
        cx = r.centerx
        self._center_text("AI ESCAPE ROOM", self.f_title, ACCENT, (cx, r.y + 110))
        self._center_text("The camera is your game master.", self.f_med, TEXT, (cx, r.y + 170))
        lines = ["Solve each puzzle in the real world: show objects, colours and hand gestures.",
                 "Hold the answer steady until the verification bar fills up.",
                 f"{self.engine.n_levels} locked levels, {config.LEVEL_TIME}s each, {config.LIVES} lives.",
                 "Every game generates a new set of puzzles."]
        for i, line in enumerate(lines):
            self._center_text(line, self.f_txt, MUTED, (cx, r.y + 230 + i * 30))
        pulse = 160 + int(95 * (0.5 + 0.5 * math.sin(time.monotonic() * 4)))
        self._center_text("Press SPACE to start", self.f_big, (pulse, pulse, pulse), (cx, r.bottom - 90))

    def _level_clear(self, now):
        r = self.cam_rect
        overlay = pygame.Surface(r.size, pygame.SRCALPHA)
        overlay.fill((30, 160, 90, 110))
        self.screen.blit(overlay, r.topleft)
        _padlock(self.screen, r.centerx - 30, r.centery - 110, GOOD, True, scale=3)
        self._center_text(f"LOCK {self.engine.level + 1} OPENED", self.f_title, TEXT, (r.centerx, r.centery + 40))
        self._center_text(self.engine.current_message(now), self.f_med, TEXT, (r.centerx, r.centery + 95))

    def _end_screen(self):
        e = self.engine
        r = self.cam_rect
        _dim(self.screen, r, 210)
        won = e.state == State.VICTORY
        self._center_text("YOU ESCAPED!" if won else "GAME OVER", self.f_title, GOOD if won else BAD,
                          (r.centerx, r.y + 90))
        self._center_text(f"Score {e.score}    Best {e.highscore}", self.f_big, TEXT, (r.centerx, r.y + 150))
        y = r.y + 200
        for lvl, desc, secs, pts in e.level_log[-6:]:
            text = f"L{lvl}  {desc[:52]}  -  {secs}s  +{pts}"
            self._center_text(text, self.f_small, MUTED, (r.centerx, y))
            y += 24
        self._center_text("SPACE to play again   ESC to quit", self.f_med, ACCENT, (r.centerx, r.bottom - 60))

    def _toast(self, msg):
        r = self.cam_rect
        surf = self.f_med.render(msg, True, BG)
        box = surf.get_rect(midbottom=(r.centerx, r.bottom - 20)).inflate(28, 14)
        pygame.draw.rect(self.screen, ACCENT, box, border_radius=10)
        self.screen.blit(surf, surf.get_rect(center=box.center))

    # ------------------------------------------------------------------ helpers
    def _card_title(self, text, x, y):
        self.screen.blit(self.f_small.render(text, True, MUTED), (x, y))
        pygame.draw.line(self.screen, CARD, (x, y + 20), (x + config.SIDEBAR_WIDTH - 44, y + 20), 1)
        return y + 26

    def _wrap(self, text, font, color, x, y, width):
        words, line = text.split(), ""
        for word in words:
            test = f"{line} {word}".strip()
            if font.size(test)[0] > width and line:
                self.screen.blit(font.render(line, True, color), (x, y))
                y += font.get_linesize()
                line = word
            else:
                line = test
        if line:
            self.screen.blit(font.render(line, True, color), (x, y))
            y += font.get_linesize()
        return y

    def _center_text(self, text, font, color, center):
        surf = font.render(text, True, color)
        self.screen.blit(surf, surf.get_rect(center=center))

    def _splash(self, text):
        self.screen.fill(BG)
        self._center_text("AI ESCAPE ROOM", self.f_title, ACCENT,
                          (config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 - 30))
        self._center_text(text, self.f_txt, MUTED, (config.WINDOW_WIDTH // 2, config.WINDOW_HEIGHT // 2 + 30))
        pygame.display.flip()
        pygame.event.pump()


# ---------------------------------------------------------------------- drawing primitives
def _open_camera(source):
    if isinstance(source, int) and hasattr(cv2, "CAP_DSHOW") and __import__("os").name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # opens much faster on Windows
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def _bar(surface, x, y, w, h, frac, color):
    pygame.draw.rect(surface, CARD, (x, y, w, h), border_radius=h // 2)
    frac = max(0.0, min(1.0, frac))
    if frac > 0:
        pygame.draw.rect(surface, color, (x, y, max(h, int(w * frac)), h), border_radius=h // 2)


def _padlock(surface, x, y, color, is_open, scale=1):
    s = scale
    body = pygame.Rect(x, y + 14 * s, 30 * s, 22 * s)
    shackle = pygame.Rect(x + 5 * s, y + (0 if not is_open else -8 * s), 20 * s, 28 * s)
    pygame.draw.arc(surface, color, shackle, 0, math.pi, 4 * s)
    pygame.draw.line(surface, color, (shackle.left + 2 * s, shackle.centery),
                     (shackle.left + 2 * s, body.top), 4 * s)
    if not is_open:
        pygame.draw.line(surface, color, (shackle.right - 2 * s, shackle.centery),
                         (shackle.right - 2 * s, body.top), 4 * s)
    pygame.draw.rect(surface, color, body, border_radius=4 * s)
    pygame.draw.circle(surface, PANEL, body.center, 3 * s)


def _heart(surface, cx, cy, color):
    pygame.draw.circle(surface, color, (cx - 5, cy - 3), 6)
    pygame.draw.circle(surface, color, (cx + 5, cy - 3), 6)
    pygame.draw.polygon(surface, color, [(cx - 11, cy - 1), (cx + 11, cy - 1), (cx, cy + 11)])


def _dim(surface, rect, alpha):
    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    overlay.fill((8, 10, 16, alpha))
    surface.blit(overlay, rect.topleft)
