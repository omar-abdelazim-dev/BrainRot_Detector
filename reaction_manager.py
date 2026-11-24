"""Temporal smoothing that turns noisy per-frame gestures into a stable one.

A candidate gesture becomes active only when it wins ``activation_frames`` of
the last ``window_frames`` votes (per-gesture, from config/gestures.json), at
least ``min_switch_seconds`` after the previous change. Because the current
gesture stays active until something else wins a majority, brief detection
dropouts don't flicker the reaction (hysteresis). When a gesture ends it
enters its ``cooldown_seconds`` and cannot re-trigger until that elapses.
"""
from collections import deque


class GestureStabilizer:
    def __init__(self, config):
        self.configure(config)
        self.reset()

    def configure(self, config):
        """Apply a (re)loaded GestureConfig without losing the current state."""
        self.config = config
        history = list(getattr(self, 'history', []))
        self.history = deque(history[-config.window_frames:], maxlen=config.window_frames)

    def reset(self):
        self.history.clear()
        self.active = 'idle'
        self.last_change = float('-inf')
        self.cooldown_until = {}

    def required_frames(self, kind):
        spec = self.config.spec(kind)
        return spec.activation_frames if spec else self.config.idle_frames

    def update(self, kind, now):
        self.history.append(kind)
        if (kind != self.active
                and self.history.count(kind) >= self.required_frames(kind)
                and now - self.last_change > self.config.min_switch_seconds
                and now >= self.cooldown_until.get(kind, float('-inf'))):
            previous = self.config.spec(self.active)
            if previous and previous.cooldown_seconds:
                self.cooldown_until[self.active] = now + previous.cooldown_seconds
            self.active = kind
            self.last_change = now
        return self.active
