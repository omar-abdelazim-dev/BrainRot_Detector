"""Synthetic landmark checks for the 6-7 pose boundaries and priority."""
import unittest
from types import SimpleNamespace as NS
import numpy as np
from app import get_gesture, is_sixseven


def point(x, y):
    return NS(x=x, y=y)


def hand(x, y, opened=True, angle=0):
    points = [point(x, y) for _ in range(21)]
    for base, tip, offset in [(5, 8, -.04), (9, 12, -.013), (13, 16, .013), (17, 20, .04)]:
        points[base] = point(x+offset, y-.05)
        points[tip] = point(x+offset, y-(.13 if opened else .025))
    for p in points:
        dx, dy = p.x-x, p.y-y
        p.x = x+dx*np.cos(angle)-dy*np.sin(angle)
        p.y = y+dx*np.sin(angle)+dy*np.cos(angle)
    return points


def face():
    result = [point(.5, .4) for _ in range(478)]
    result[10], result[152] = point(.5, .2), point(.5, .55)
    result[234], result[454] = point(.35, .4), point(.65, .4)
    return result


class SixSevenTests(unittest.TestCase):
    def test_chest_height_and_rotated_palms(self):
        for angle in (0, np.pi/2, -np.pi/2):
            self.assertTrue(is_sixseven(face(), [hand(.25,.72,angle=angle), hand(.75,.72,angle=-angle)], 960,720))

    def test_reject_single_hand_fists_and_low_hands(self):
        for hands in ([hand(.2,.6,False)], [hand(.2,.6,False),hand(.8,.6,False)],
                      [hand(.2,1.1),hand(.8,1.1)]):
            self.assertFalse(is_sixseven(face(),hands,960,720))

    def test_no_face_required(self):
        self.assertTrue(is_sixseven(None,[hand(.2,.75),hand(.8,.75)],960,720))

    def test_two_hands_take_priority_over_chin_touch(self):
        f = face()
        hands = [hand(.2,.6),hand(.8,.6)]
        hands[0][8] = point(.5,.55)
        result = get_gesture(NS(multi_face_landmarks=[NS(landmark=f)]),
                             NS(multi_hand_landmarks=[NS(landmark=h) for h in hands]),960,720)
        self.assertEqual(result,'sixseven')


class NewReactionTests(unittest.TestCase):
    def classify(self, hands):
        return get_gesture(NS(multi_face_landmarks=[NS(landmark=face())]),
                           NS(multi_hand_landmarks=[NS(landmark=h) for h in hands]),960,720)

    def test_raised_palms_are_cinema(self):
        self.assertEqual(self.classify([hand(.2,.3), hand(.8,.3)]), 'cinema')

    def test_chest_palms_remain_sixseven(self):
        self.assertEqual(self.classify([hand(.2,.7), hand(.8,.7)]), 'sixseven')

    def test_single_closed_hand_is_handshake(self):
        self.assertEqual(self.classify([hand(.2,.65,False)]), 'handshake')

    def test_single_open_hand_is_not_handshake(self):
        self.assertNotEqual(self.classify([hand(.2,.65)]), 'handshake')


if __name__ == '__main__':
    unittest.main()
