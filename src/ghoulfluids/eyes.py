# src/ghoulfluids/eyes.py
from __future__ import annotations
import math
import random


class EyesController:
    """
    One pair of glowing eyes: waits, fades in somewhere in the smoke, blinks,
    then fades out. With eyes_flee they leave quickly when someone shows up.

    step() returns (alpha, (u, v), openness, scale) for the shader; alpha 0 = hidden.
    """

    def __init__(self, cfg, seed: int = 7, place=None):
        self.cfg = cfg
        self.rng = random.Random(seed)
        # place(rng) -> (u, v); lets EyesSwarm keep pairs apart
        self.place = place or (
            lambda rng: (rng.uniform(0.2, 0.8), rng.uniform(0.45, 0.8))
        )
        self.empty_t = 0.0  # seconds since a person was last seen
        self.alpha = 0.0
        self.pos = (0.5, 0.6)
        self.scale = 1.0
        self.show_t = -1.0  # time into the current appearance; <0 = not showing
        self.blinks: list[float] = []
        self.next_in = self._next_delay()

    @property
    def showing(self) -> bool:
        return self.show_t >= 0.0

    def _next_delay(self) -> float:
        return self.rng.uniform(self.cfg.eyes_min_wait, self.cfg.eyes_max_wait)

    def _start(self):
        self.show_t = 0.0
        self.pos = self.place(self.rng)
        self.scale = self.rng.uniform(0.6, 1.4)  # smaller ones read as farther back
        hold = self.cfg.eyes_hold
        fade = self.cfg.eyes_fade
        self.blinks = sorted(
            fade + self.rng.uniform(0.3, max(0.4, hold - 0.3))
            for _ in range(self.rng.randint(1, 2))
        )

    def _hide(self):
        self.alpha = 0.0
        self.show_t = -1.0
        self.next_in = self._next_delay()

    def step(self, dt: float, person_present: bool):
        cfg = self.cfg
        hidden = (0.0, self.pos, 1.0, self.scale)
        if not cfg.eyes:
            return hidden

        if person_present:
            self.empty_t = 0.0
            if cfg.eyes_flee:
                self.alpha = max(0.0, self.alpha - dt / 0.4)
                if self.alpha == 0.0:
                    self._hide()
                return self.alpha, self.pos, 1.0, self.scale
        else:
            self.empty_t += dt

        if not self.showing:
            self.next_in -= dt
            may_start = cfg.eyes_min_empty <= 0.0 or self.empty_t >= cfg.eyes_min_empty
            if self.next_in <= 0.0 and may_start:
                self._start()
            return hidden

        self.show_t += dt
        t, fade, hold = self.show_t, cfg.eyes_fade, cfg.eyes_hold
        if t < fade:
            self.alpha = t / fade
        elif t < fade + hold:
            self.alpha = 1.0
        elif t < 2 * fade + hold:
            self.alpha = 1.0 - (t - fade - hold) / fade
        else:
            self._hide()
            return hidden

        # blink: lids close and reopen over ~0.25s
        openness = 1.0
        for b in self.blinks:
            x = (t - b) / 0.25
            if 0.0 <= x <= 1.0:
                openness = min(openness, abs(math.cos(math.pi * x)))
        return self.alpha, self.pos, openness, self.scale


class EyesSwarm:
    """Several independent pairs of eyes, kept from appearing on top of each other."""

    def __init__(self, cfg, seed: int = 7):
        self.cfg = cfg
        self.pairs = [
            EyesController(cfg, seed=seed + i, place=self._place)
            for i in range(max(0, cfg.eyes_count))
        ]

    def _place(self, rng):
        taken = [p.pos for p in self.pairs if p.showing]
        best, best_d = None, -1.0
        for _ in range(12):  # keep the candidate farthest from the others
            cand = (rng.uniform(0.12, 0.88), rng.uniform(0.35, 0.85))
            d = min((math.dist(cand, q) for q in taken), default=1.0)
            if d > 0.25:
                return cand
            if d > best_d:
                best, best_d = cand, d
        return best

    def step(self, dt: float, person_present: bool):
        return [p.step(dt, person_present) for p in self.pairs]
