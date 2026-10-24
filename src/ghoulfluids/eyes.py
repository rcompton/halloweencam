# src/ghoulfluids/eyes.py
from __future__ import annotations
import math
import random


class EyesController:
    """
    A pair of glowing eyes that fades in somewhere in the smoke after nobody has
    been in front of the camera for a while, blinks, then fades out. They leave
    quickly if someone shows up.

    step() returns (alpha, (u, v), openness) for the shader; alpha 0 = hidden.
    """

    def __init__(self, cfg, seed: int = 7):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.empty_t = 0.0  # seconds since a person was last seen
        self.alpha = 0.0
        self.pos = (0.5, 0.6)
        self.show_t = -1.0  # time into the current appearance; <0 = not showing
        self.blinks: list[float] = []
        self.next_in = self._next_delay()

    def _next_delay(self) -> float:
        return self.rng.uniform(self.cfg.eyes_min_wait, self.cfg.eyes_max_wait)

    def _start(self):
        self.show_t = 0.0
        self.pos = (self.rng.uniform(0.2, 0.8), self.rng.uniform(0.45, 0.8))
        hold = self.cfg.eyes_hold
        fade = self.cfg.eyes_fade
        self.blinks = sorted(
            fade + self.rng.uniform(0.5, hold - 0.5)
            for _ in range(self.rng.randint(1, 2))
        )

    def step(self, dt: float, person_present: bool):
        cfg = self.cfg
        if not cfg.eyes:
            return 0.0, self.pos, 1.0

        if person_present:
            self.empty_t = 0.0
            # leave fast
            self.alpha = max(0.0, self.alpha - dt / 0.4)
            if self.alpha == 0.0:
                self.show_t = -1.0
                self.next_in = self._next_delay()
            return self.alpha, self.pos, 1.0

        self.empty_t += dt
        if self.show_t < 0.0:
            self.next_in -= dt
            if self.next_in <= 0.0 and self.empty_t >= cfg.eyes_min_empty:
                self._start()
            return 0.0, self.pos, 1.0

        self.show_t += dt
        t, fade, hold = self.show_t, cfg.eyes_fade, cfg.eyes_hold
        if t < fade:
            self.alpha = t / fade
        elif t < fade + hold:
            self.alpha = 1.0
        elif t < 2 * fade + hold:
            self.alpha = 1.0 - (t - fade - hold) / fade
        else:
            self.alpha = 0.0
            self.show_t = -1.0
            self.next_in = self._next_delay()

        # blink: lids close and reopen over ~0.25s
        openness = 1.0
        for b in self.blinks:
            x = (t - b) / 0.25
            if 0.0 <= x <= 1.0:
                openness = min(openness, abs(math.cos(math.pi * x)))
        return self.alpha, self.pos, openness
