"""Temporal confirmation for alternating chest-height 6-7 hands."""
class SixSevenMotion:
    def __init__(self):
        self.reset()

    def reset(self):
        self.hand_count = 0
        self.phase = 0
        self.anchor = None
        self.phase_time = float('-inf')
        self.last_candidate = float('-inf')
        self.confirmed_until = float('-inf')

    def filter(self, kind, hands, now):
        if kind != 'sixseven' or len(hands) not in (1, 2):
            if now-self.last_candidate > .30:
                self.reset()
            return kind
        if now-self.last_candidate > .30 or self.hand_count != len(hands):
            self.reset()
        self.hand_count = len(hands)
        self.last_candidate = now
        palms = sorted((sum(h[i].x for i in (0,5,9,13,17))/5,
                        sum(h[i].y for i in (0,5,9,13,17))/5) for h in hands)
        if len(palms) == 1:
            y = palms[0][1]
            if self.anchor is None or now-self.phase_time > 1.5:
                self.anchor, self.phase, self.phase_time = (y,), 0, now
            delta = y-self.anchor[0]
            if abs(delta) >= .025:
                phase = 1 if delta > 0 else -1
                if self.phase and phase != self.phase:
                    self.confirmed_until = now+.75
                self.phase, self.anchor, self.phase_time = phase, (y,), now
            return 'sixseven' if now < self.confirmed_until else 'idle'
        left, right = palms[0][1], palms[1][1]
        difference = left-right
        phase = 1 if difference > .025 else -1 if difference < -.025 else 0
        if phase and (phase != self.phase or now-self.phase_time > 1.5):
            if self.anchor and self.phase and phase != self.phase and now-self.phase_time <= 1.5:
                dl, dr = left-self.anchor[0], right-self.anchor[1]
                # Both hands must move, in opposite directions, beyond tiny jitter.
                if dl*dr < 0 and min(abs(dl), abs(dr)) >= .015:
                    self.confirmed_until = now+.75
            self.phase, self.anchor, self.phase_time = phase, (left,right), now
        return 'sixseven' if now < self.confirmed_until else 'idle'
