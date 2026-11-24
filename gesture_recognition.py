"""Rule-based gesture detectors evaluated through a priority registry.

Each detector is a function ``(frame, params) -> bool`` registered with
``@detector``. ``frame`` is a :class:`FrameLandmarks` snapshot and ``params``
are the detector's thresholds (code defaults, overridable per gesture in
``config/gestures.json``). :class:`GestureRecognizer` runs the enabled
detectors from highest to lowest priority and returns the first match, so
conflicts are resolved by priority rather than by code order.

Adding a gesture = one decorated function + (optionally) a JSON entry.
"""
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import numpy as np

PALM = (0, 5, 9, 13, 17)
FINGERS = ((8, 5), (12, 9), (16, 13), (20, 17))  # (tip, base knuckle)

# Face Landmarker / refined FaceMesh indices (same 478-point topology).
UPPER_LIP, LOWER_LIP = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
CHIN, FACE_LEFT, FACE_RIGHT = 152, 234, 454
EYE_OUTER_LEFT, EYE_OUTER_RIGHT = 33, 263
NOSE_TIP = 1


class ConfigError(ValueError):
    """Raised when a gesture or reaction config file is invalid."""


def pixel(point, w, h):
    return np.array([point.x*w, point.y*h], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a-b))


@dataclass
class FrameLandmarks:
    """Landmarks for one video frame, with shared measurements cached."""
    face: list | None
    hands: list
    world_hands: list
    w: int
    h: int
    blendshapes: dict = field(default_factory=dict)  # e.g. {'jawOpen': .7}

    @classmethod
    def from_results(cls, face, hands_result, w, h):
        """``face`` is a FaceObservation (or None); ``hands_result`` is the
        MediaPipe Hands solution output."""
        hands = [hand.landmark for hand in getattr(hands_result, 'multi_hand_landmarks', None) or []]
        world = [hand.landmark for hand in getattr(hands_result, 'multi_hand_world_landmarks', None) or []]
        return cls(face.landmarks if face else None, hands, world, w, h,
                   face.blendshapes if face else {})

    # --- face measurements (only valid when self.face is not None) ---
    def p(self, index):
        return pixel(self.face[index], self.w, self.h)

    @cached_property
    def face_width(self):
        return max(1.0, dist(self.p(FACE_LEFT), self.p(FACE_RIGHT)))

    @cached_property
    def mouth(self):
        return (self.p(UPPER_LIP) + self.p(LOWER_LIP)) / 2

    @cached_property
    def mouth_gap(self):
        return dist(self.p(UPPER_LIP), self.p(LOWER_LIP)) / self.face_width

    @cached_property
    def mouth_width(self):
        return dist(self.p(MOUTH_LEFT), self.p(MOUTH_RIGHT)) / self.face_width

    def blend(self, *names):
        """Mean score of the named blendshapes, or None without blendshapes."""
        if not self.blendshapes:
            return None
        return sum(self.blendshapes.get(name, 0.0) for name in names) / len(names)

    # --- hand measurements ---
    def hand_px(self, hand, index):
        return pixel(hand[index], self.w, self.h)

    def palm_center(self, hand):
        return np.mean([self.hand_px(hand, i) for i in PALM], axis=0)

    def folded_fingers(self, hand, ratio):
        wrist = self.hand_px(hand, 0)
        return sum(dist(self.hand_px(hand, tip), wrist) < ratio*dist(self.hand_px(hand, base), wrist)
                   for tip, base in FINGERS)

    def finger_extended(self, hand, tip, base, ratio=1.05):
        wrist = self.hand_px(hand, 0)
        return dist(self.hand_px(hand, tip), wrist) > ratio*dist(self.hand_px(hand, base), wrist)

    def is_pointing(self, hand):
        """Index finger out, middle/ring/pinky folded."""
        return (self.finger_extended(hand, 8, 5)
                and not any(self.finger_extended(hand, tip, base) for tip, base in FINGERS[1:]))

    def extended_fingers_3d(self, hand_index, ratio):
        """Extended-finger count using metric world landmarks when available.
        A palm facing up/forward is foreshortened in the image, so 2D tips can
        look as close as a fist; world joints keep the true finger reach."""
        hand = self.hands[hand_index]
        use_world = len(self.world_hands) == len(self.hands)

        def position(index):
            if use_world:
                q = self.world_hands[hand_index][index]
                return np.array([q.x, q.y, q.z], dtype=np.float32)
            q = hand[index]
            return np.array([q.x*self.w, q.y*self.h, getattr(q, 'z', 0.0)*self.w], dtype=np.float32)

        wrist = position(0)
        return sum(dist(position(tip), wrist) > ratio*dist(position(base), wrist)
                   for tip, base in FINGERS)


# ---------------------------------------------------------------- registry
@dataclass
class DetectorInfo:
    fn: callable
    priority: int
    params: dict


DETECTORS: dict[str, DetectorInfo] = {}


def detector(gesture_id, priority, **defaults):
    """Register a detector with its default priority and thresholds."""
    def wrap(fn):
        DETECTORS[gesture_id] = DetectorInfo(fn, priority, defaults)
        return fn
    return wrap


# --------------------------------------------------------------- detectors
def sixseven_pose(frame, params):
    """One or two open hands, including low, upward-facing palms."""
    hands = frame.hands
    if len(hands) not in (1, 2):
        return False
    centers = [frame.palm_center(hand) for hand in hands]
    # The reference pose is below the chin, near the bottom of the frame.
    if any(center[1] > params['max_palm_y']*frame.h for center in centers):
        return False
    palm_sizes = [max(1, dist(frame.hand_px(hand, 5), frame.hand_px(hand, 17))) for hand in hands]
    if len(hands) == 2 and dist(*centers) < params['min_hand_separation']*max(palm_sizes):
        return False
    return all(frame.extended_fingers_3d(i, params['extension_ratio']) >= params['min_extended_fingers']
               for i in range(len(hands)))


SIXSEVEN_DEFAULTS = dict(max_palm_y=.98, min_hand_separation=.9, extension_ratio=1.05,
                         min_extended_fingers=2)


@detector('cinema', priority=100, **SIXSEVEN_DEFAULTS)
def detect_cinema(frame, params):
    """Two open palms raised beside the head."""
    if frame.face is None or len(frame.hands) != 2 or not sixseven_pose(frame, params):
        return False
    centers = [frame.palm_center(hand) for hand in frame.hands]
    return (all(c[1] < frame.p(NOSE_TIP)[1] for c in centers)
            and min(c[0] for c in centers) < frame.p(FACE_LEFT)[0]
            and max(c[0] for c in centers) > frame.p(FACE_RIGHT)[0])


@detector('sixseven', priority=90, **SIXSEVEN_DEFAULTS)
def detect_sixseven(frame, params):
    """Open palm pose; the bounce itself is confirmed by SixSevenMotion."""
    return sixseven_pose(frame, params)


def single_fist(frame, params):
    return (len(frame.hands) == 1
            and not frame.is_pointing(frame.hands[0])   # index out = SHHH/mewing, not a fist
            and frame.folded_fingers(frame.hands[0], params['fold_ratio']) >= params['min_folded_fingers'])


@detector('calm', priority=80, fold_ratio=1.05, min_folded_fingers=3,
          chin_dx=.40, chin_dy_min=-.20, chin_dy_max=.50)
def detect_calm(frame, params):
    """Closed fist resting under the chin."""
    if frame.face is None or not single_fist(frame, params):
        return False
    chin = frame.p(CHIN)
    knuckles = np.mean([frame.hand_px(frame.hands[0], i) for i in (5, 9, 13, 17)], axis=0)
    fw = frame.face_width
    return (abs(knuckles[0]-chin[0]) < params['chin_dx']*fw
            and params['chin_dy_min']*fw < knuckles[1]-chin[1] < params['chin_dy_max']*fw)


@detector('handshake', priority=70, fold_ratio=1.05, min_folded_fingers=3)
def detect_handshake(frame, params):
    """One closed fist anywhere in view."""
    return single_fist(frame, params)


@detector('shush', priority=60, mouth_radius=.145, min_finger_length=.075)
def detect_shush(frame, params):
    """Extended index fingertip on the lips."""
    if frame.face is None:
        return False
    fw = frame.face_width
    return any(dist(frame.hand_px(hand, 8), frame.mouth) < params['mouth_radius']*fw
               and dist(frame.hand_px(hand, 8), frame.hand_px(hand, 6)) > params['min_finger_length']*fw
               for hand in frame.hands)


@detector('mewing', priority=50, chin_radius=.19)
def detect_mewing(frame, params):
    """Index fingertip on the jawline/chin."""
    if frame.face is None:
        return False
    return any(dist(frame.hand_px(hand, 8), frame.p(CHIN)) < params['chin_radius']*frame.face_width
               for hand in frame.hands)


@detector('shock', priority=40, min_jaw_open=.45, min_brow_raise=.30, require_raised_brows=True,
          min_mouth_gap=.095)
def detect_shock(frame, params):
    """Open mouth (+ raised eyebrows when expression scores are available)."""
    if frame.face is None:
        return False
    jaw = frame.blend('jawOpen')
    if jaw is None:  # geometric fallback without blendshapes
        return frame.mouth_gap > params['min_mouth_gap']
    if jaw < params['min_jaw_open']:
        return False
    if not params['require_raised_brows']:
        return True
    brows = max(frame.blend('browInnerUp'), frame.blend('browOuterUpLeft', 'browOuterUpRight'))
    return brows >= params['min_brow_raise']


@detector('smile', priority=30, min_smile=.55, max_jaw_open=.45, min_mouth_width=.49, max_mouth_gap=.085)
def detect_smile(frame, params):
    """Wide smile with a mostly closed mouth."""
    if frame.face is None:
        return False
    smile = frame.blend('mouthSmileLeft', 'mouthSmileRight')
    if smile is None:  # geometric fallback without blendshapes
        return frame.mouth_width > params['min_mouth_width'] and frame.mouth_gap < params['max_mouth_gap']
    return smile >= params['min_smile'] and frame.blend('jawOpen') < params['max_jaw_open']


@detector('tilt', priority=20, min_slope=.18)
def detect_tilt(frame, params):
    """Head rolled sideways (eye line slope; .18 is roughly 10 degrees)."""
    if frame.face is None:
        return False
    eyes = frame.p(EYE_OUTER_RIGHT) - frame.p(EYE_OUTER_LEFT)
    return abs(float(eyes[1])) > abs(float(eyes[0])) * params['min_slope']


BROW_CENTER, IRIS_LEFT, IRIS_RIGHT = 168, 468, 473


@detector('kai', priority=105, **SIXSEVEN_DEFAULTS, max_gap_to_head=.85, max_palm_below_eyes=.25,
          min_jaw_open=.18, min_mouth_gap=.05)
def detect_kai(frame, params):
    """Both open hands right beside the head (by the ears) with the mouth open
    or tongue out. Closed mouth falls through to 'cinema'."""
    if frame.face is None or len(frame.hands) != 2:
        return False
    if not all(frame.extended_fingers_3d(i, params['extension_ratio']) >= params['min_extended_fingers']
               for i in range(2)):
        return False
    jaw = frame.blend('jawOpen')
    if (jaw if jaw is not None else frame.mouth_gap) < (params['min_jaw_open'] if jaw is not None
                                                        else params['min_mouth_gap']):
        return False
    fw = frame.face_width
    left_edge, right_edge = frame.p(FACE_LEFT)[0], frame.p(FACE_RIGHT)[0]
    eyes_y = (frame.p(IRIS_LEFT)[1] + frame.p(IRIS_RIGHT)[1]) / 2 if len(frame.face) > IRIS_RIGHT \
        else frame.p(EYE_OUTER_LEFT)[1]
    centers = sorted((frame.palm_center(hand) for hand in frame.hands), key=lambda c: c[0])
    near_left = left_edge - params['max_gap_to_head']*fw < centers[0][0] < left_edge + .15*fw
    near_right = right_edge - .15*fw < centers[1][0] < right_edge + params['max_gap_to_head']*fw
    high = all(c[1] < eyes_y + params['max_palm_below_eyes']*fw for c in centers)
    return near_left and near_right and high


@detector('tamer', priority=95, wrist_radius=.75)
def detect_tamer(frame, params):
    """Index fingertip tapping the other hand's wrist (\"what time is it?\")."""
    if len(frame.hands) != 2:
        return False
    for pointer, other in ((0, 1), (1, 0)):
        hand, target = frame.hands[pointer], frame.hands[other]
        # Only a real pointing hand counts: open 6-7 palms also have the index out.
        if not frame.is_pointing(hand):
            continue
        palm = max(1.0, dist(frame.hand_px(target, 5), frame.hand_px(target, 17)))
        if dist(frame.hand_px(hand, 8), frame.hand_px(target, 0)) < params['wrist_radius']*palm:
            return True
    return False


@detector('baby', priority=85, mouth_radius=.30)
def detect_baby(frame, params):
    """Fist or palm covering the mouth (giggling). SHHH is a single fingertip
    on the lips with the knuckles further down, so it doesn't match."""
    if frame.face is None:
        return False
    for hand in frame.hands:
        if frame.is_pointing(hand):   # a single finger on the lips is SHHH
            continue
        knuckles = np.mean([frame.hand_px(hand, i) for i in (5, 9, 13, 17)], axis=0)
        if dist(knuckles, frame.mouth) < params['mouth_radius']*frame.face_width:
            return True
    return False


@detector('iguessbro', priority=88, eye_radius=.22)
def detect_iguessbro(frame, params):
    """Fingertips pressed to the eyes or between the eyebrows."""
    if frame.face is None:
        return False
    targets = [frame.p(BROW_CENTER)]
    if len(frame.face) > IRIS_RIGHT:
        targets += [frame.p(IRIS_LEFT), frame.p(IRIS_RIGHT)]
    limit = params['eye_radius']*frame.face_width
    return any(dist(frame.hand_px(hand, tip), target) < limit
               for hand in frame.hands for tip in (8, 12) for target in targets)


@detector('son', priority=35, min_side=.35, min_difference=.25, max_jaw_open=.30)
def detect_son(frame, params):
    """Smirk: one mouth corner up, the other not (needs expression scores)."""
    if frame.face is None or not frame.blendshapes:
        return False
    left, right = frame.blend('mouthSmileLeft'), frame.blend('mouthSmileRight')
    return (max(left, right) >= params['min_side'] and abs(left - right) >= params['min_difference']
            and frame.blend('jawOpen') < params['max_jaw_open'])


# ------------------------------------------------------------------ config
@dataclass
class GestureSpec:
    id: str
    detect: callable
    priority: int
    params: dict
    enabled: bool = True
    activation_frames: int = 5
    cooldown_seconds: float = 0.0


@dataclass
class GestureConfig:
    specs: list  # GestureSpec, highest priority first
    window_frames: int = 7
    idle_frames: int = 5
    min_switch_seconds: float = .40

    def spec(self, gesture_id):
        return next((s for s in self.specs if s.id == gesture_id), None)


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f'{name} must be a number, got {value!r}')
    if minimum is not None and value < minimum:
        raise ConfigError(f'{name} must be >= {minimum}, got {value}')
    return value


def load_gesture_config(path=None):
    """Code defaults, overridden by ``path`` (JSON) when given and present."""
    data = {}
    if path is not None and Path(path).exists():
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise ConfigError(f'{path}: invalid JSON ({exc})') from exc
    smoothing = data.get('smoothing', {})
    window = int(_number(smoothing.get('window_frames', 7), 'smoothing.window_frames', 1))
    overrides = data.get('gestures', {})
    for unknown in set(overrides) - set(DETECTORS):
        print(f'[config] Ignoring unknown gesture {unknown!r} in {path}', flush=True)

    specs = []
    for gesture_id, info in DETECTORS.items():
        entry = overrides.get(gesture_id, {})
        params = dict(info.params)
        for key, value in entry.get('params', {}).items():
            if key not in params:
                print(f'[config] {gesture_id}: ignoring unknown param {key!r}', flush=True)
                continue
            if isinstance(params[key], bool):
                if not isinstance(value, bool):
                    raise ConfigError(f'{gesture_id}.params.{key} must be true/false')
                params[key] = value
            else:
                params[key] = _number(value, f'{gesture_id}.params.{key}')
        frames = int(_number(entry.get('activation_frames', 5), f'{gesture_id}.activation_frames', 1))
        specs.append(GestureSpec(
            id=gesture_id, detect=info.fn,
            priority=int(_number(entry.get('priority', info.priority), f'{gesture_id}.priority')),
            params=params, enabled=bool(entry.get('enabled', True)),
            activation_frames=min(frames, window),
            cooldown_seconds=float(_number(entry.get('cooldown_seconds', 0.0),
                                           f'{gesture_id}.cooldown_seconds', 0))))
    specs.sort(key=lambda s: s.priority, reverse=True)
    return GestureConfig(
        specs, window,
        idle_frames=min(window, int(_number(smoothing.get('idle_frames', 5), 'smoothing.idle_frames', 1))),
        min_switch_seconds=float(_number(smoothing.get('min_switch_seconds', .40),
                                         'smoothing.min_switch_seconds', 0)))


class GestureRecognizer:
    def __init__(self, config=None):
        self.config = config or load_gesture_config()

    def matches(self, frame):
        """Every enabled gesture present in ``frame``, highest priority first."""
        return [spec.id for spec in self.config.specs if spec.enabled and spec.detect(frame, spec.params)]

    def classify(self, frame):
        """Highest-priority enabled gesture present in ``frame``, else 'idle'."""
        for spec in self.config.specs:
            if spec.enabled and spec.detect(frame, spec.params):
                return spec.id
        return 'idle'


_DEFAULT = GestureRecognizer()


# --------------------------------------------- legacy helpers (tests/tools)
def get_gesture(face_result, hands_result, w, h):
    """Classify using code defaults. ``face_result`` may be a FaceObservation
    or a legacy FaceMesh result with ``multi_face_landmarks``."""
    if hasattr(face_result, 'multi_face_landmarks'):
        faces = face_result.multi_face_landmarks
        face_result = type('Face', (), {'landmarks': faces[0].landmark, 'blendshapes': {}})() if faces else None
    return _DEFAULT.classify(FrameLandmarks.from_results(face_result, hands_result, w, h))


def is_sixseven(face, hands, w, h, world_hands=None):
    frame = FrameLandmarks(face, list(hands), list(world_hands or []), w, h)
    return sixseven_pose(frame, SIXSEVEN_DEFAULTS)
