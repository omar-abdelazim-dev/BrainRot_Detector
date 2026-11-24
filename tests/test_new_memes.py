"""Synthetic checks for son / baby / iguessbro / kai / tamer and their neighbours."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import unittest

from gesture_motion import SixSevenMotion, resolve
from gesture_recognition import FrameLandmarks, GestureRecognizer, load_gesture_config
from test_gestures import hand, point

W, H = 960, 720


def face():
    """Face with distinct mouth, nose, eyes and brow positions."""
    f = [point(.5, .4) for _ in range(478)]
    f[234], f[454] = point(.35, .38), point(.65, .38)      # face sides (width 288px)
    f[152] = point(.5, .55)                                 # chin
    f[13], f[14] = point(.5, .45), point(.5, .45)           # lips
    f[61], f[291] = point(.45, .45), point(.55, .45)        # mouth corners
    f[1] = point(.5, .38)                                   # nose tip
    f[168] = point(.5, .30)                                 # between the brows
    f[468], f[473] = point(.44, .32), point(.56, .32)       # irises
    f[33], f[263] = point(.42, .32), point(.58, .32)        # eye corners (level)
    return f


def classify(hands=(), blendshapes=None):
    recognizer = GestureRecognizer(load_gesture_config(Path(__file__).resolve().parents[1] / 'config' / 'gestures.json'))
    frame = FrameLandmarks(face(), list(hands), [], W, H, blendshapes or {})
    # Same selection as the app: an unconfirmed 6-7 pose yields to the next match.
    return resolve(recognizer.matches(frame), SixSevenMotion(), frame.hands, 0.0)


def pointing(x, y):
    """Index out, other fingers folded; index tip at (x-.04, y-.13)."""
    h = hand(x, y, opened=False)
    h[8] = point(x-.04, y-.13)
    return h


NEUTRAL = {'jawOpen': .02}


class NewMemeTests(unittest.TestCase):
    def test_son_smirk(self):
        self.assertEqual(classify(blendshapes={'mouthSmileLeft': .7, 'mouthSmileRight': .15, 'jawOpen': .05}), 'son')

    def test_symmetric_smile_is_not_son(self):
        self.assertEqual(classify(blendshapes={'mouthSmileLeft': .7, 'mouthSmileRight': .7, 'jawOpen': .05}), 'smile')

    def test_baby_fist_over_mouth(self):
        self.assertEqual(classify([hand(.5, .50, opened=False)], NEUTRAL), 'baby')

    def test_finger_on_lips_stays_shush(self):
        self.assertEqual(classify([pointing(.54, .58)], NEUTRAL), 'shush')

    def test_iguessbro_fingers_on_brow(self):
        self.assertEqual(classify([hand(.54, .43)], NEUTRAL), 'iguessbro')

    def test_kai_hands_by_ears_mouth_open(self):
        self.assertEqual(classify([hand(.27, .38), hand(.73, .38)], {'jawOpen': .4}), 'kai')

    def test_same_pose_mouth_closed_is_cinema(self):
        self.assertEqual(classify([hand(.27, .38), hand(.73, .38)], NEUTRAL), 'cinema')

    def test_tamer_finger_on_wrist(self):
        self.assertEqual(classify([pointing(.40, .71), hand(.36, .58, opened=False)], NEUTRAL), 'tamer')

    def test_open_palms_close_together_are_not_tamer(self):
        # Seen live: open 6-7 palms near each other were read as wrist-tapping.
        self.assertNotEqual(classify([hand(.40, .71), hand(.36, .58)], NEUTRAL), 'tamer')

    def test_separated_open_hands_are_not_tamer(self):
        self.assertNotEqual(classify([hand(.2, .75), hand(.8, .75)], NEUTRAL), 'tamer')

    def test_bouncing_hands_still_sixseven(self):
        recognizer = GestureRecognizer(load_gesture_config())
        motion = SixSevenMotion()
        for now, left, right in ((0, .7, .8), (.2, .8, .7)):
            frame = FrameLandmarks(face(), [hand(.2, left), hand(.8, right)], [], W, H, NEUTRAL)
            result = resolve(recognizer.matches(frame), motion, frame.hands, now)
        self.assertEqual(result, 'sixseven')


if __name__ == '__main__':
    unittest.main()
