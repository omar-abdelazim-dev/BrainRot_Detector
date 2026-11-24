"""Temporal confirmation for alternating chest-height 6-7 hands."""


def resolve(matches, motion, hands, now, outranks=()):
    """Pick the gesture for this frame from priority-ordered ``matches``.

    A 6-7 pose only counts once ``motion`` confirms the bounce; until then the
    next matching gesture is used, so an open hand held still at the face can
    still trigger face/hand gestures below 6-7 in priority.

    While a bounce is confirmed, frames where fast-moving hands blur (pose
    lost, fingers look folded) keep reading as 6-7, unless a gesture in
    ``outranks`` (higher priority than 6-7) is present.
    """
    first = matches[0] if matches else 'idle'
    kind = motion.filter(first, hands, now)
    if first == 'sixseven' and kind != 'sixseven':
        return matches[1] if len(matches) > 1 else 'idle'
    if kind != 'sixseven' and kind not in outranks and hands and motion.confirmed(now):
        return 'sixseven'
    return kind


class SixSevenMotion:
    PHASE = .02          # height difference (fraction of frame) that counts as up/down
    MIN_MOVE = .010      # each hand must move this far between swaps
    SWAP_WINDOW = 1.5    # seconds allowed between up/down swaps
    HOLD = 1.0           # seconds a confirmed bounce stays on
    GRACE = .45          # seconds of lost tracking tolerated before forgetting progress

    def __init__(self):
        self.reset()

    def reset(self):
        self.confirmed_until = float('-inf')
        self.last_candidate = float('-inf')
        self._reset_tracking()

    def _reset_tracking(self):
        """Forget the current up/down reference but keep any confirmed bounce."""
        self.hand_count = 0
        self.phase = 0
        self.anchor = None
        self.phase_time = float('-inf')

    def confirmed(self, now):
        return now < self.confirmed_until

    def filter(self, kind, hands, now):
        if kind != 'sixseven' or len(hands) not in (1, 2):
            if now-self.last_candidate > self.GRACE:
                self.reset()
            return kind
        if now-self.last_candidate > self.GRACE:
            self.reset()
        elif self.hand_count != len(hands):
            # A hand dropped out/reappeared mid-bounce: re-anchor, keep confirmation.
            self._reset_tracking()
        self.hand_count = len(hands)
        self.last_candidate = now
        palms = sorted((sum(h[i].x for i in (0,5,9,13,17))/5,
                        sum(h[i].y for i in (0,5,9,13,17))/5) for h in hands)
        if len(palms) == 1:
            y = palms[0][1]
            if self.anchor is None or now-self.phase_time > self.SWAP_WINDOW:
                self.anchor, self.phase, self.phase_time = (y,), 0, now
            delta = y-self.anchor[0]
            if abs(delta) >= self.PHASE:
                phase = 1 if delta > 0 else -1
                if self.phase and phase != self.phase:
                    self.confirmed_until = now+self.HOLD
                self.phase, self.anchor, self.phase_time = phase, (y,), now
            return 'sixseven' if self.confirmed(now) else 'idle'
        left, right = palms[0][1], palms[1][1]
        difference = left-right
        phase = 1 if difference > self.PHASE else -1 if difference < -self.PHASE else 0
        if phase and (phase != self.phase or now-self.phase_time > self.SWAP_WINDOW):
            if self.anchor and self.phase and phase != self.phase and now-self.phase_time <= self.SWAP_WINDOW:
                dl, dr = left-self.anchor[0], right-self.anchor[1]
                # Both hands must move, in opposite directions, beyond tiny jitter.
                if dl*dr < 0 and min(abs(dl), abs(dr)) >= self.MIN_MOVE:
                    self.confirmed_until = now+self.HOLD
            self.phase, self.anchor, self.phase_time = phase, (left,right), now
        return 'sixseven' if self.confirmed(now) else 'idle'
