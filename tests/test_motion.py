import unittest
from gesture_motion import SixSevenMotion
from test_gestures import hand, face, NS
from app import get_gesture


class MotionTests(unittest.TestCase):
    def test_still_hands_do_not_trigger(self):
        detector=SixSevenMotion()
        for i in range(30):
            self.assertEqual(detector.filter('sixseven',[hand(.2,.6),hand(.8,.7)],i*.05),'idle')

    def test_alternating_hands_trigger_and_expire(self):
        detector=SixSevenMotion()
        self.assertEqual(detector.filter('sixseven',[hand(.2,.6),hand(.8,.7)],0),'idle')
        self.assertEqual(detector.filter('sixseven',[hand(.8,.6),hand(.2,.7)],.2),'sixseven')
        for i in range(5,25):
            result=detector.filter('sixseven',[hand(.2,.7),hand(.8,.6)],i*.05)
        self.assertEqual(result,'idle')

    def test_one_moving_hand_does_not_trigger(self):
        detector=SixSevenMotion()
        detector.filter('sixseven',[hand(.2,.6),hand(.8,.7)],0)
        self.assertEqual(detector.filter('sixseven',[hand(.2,.8),hand(.8,.7)],.2),'idle')

    def test_other_poses_are_not_overridden(self):
        detector=SixSevenMotion()
        for kind in ('cinema','handshake','calm'):
            self.assertEqual(detector.filter(kind,[],0),kind)

    def test_fist_under_chin_is_calm(self):
        result=get_gesture(NS(multi_face_landmarks=[NS(landmark=face())]),
                           NS(multi_hand_landmarks=[NS(landmark=hand(.5,.65,False))]),960,720)
        self.assertEqual(result,'calm')


def palm_up(x, y, opened=True):
    points = hand(x,y,False)
    for p in points:
        p.z = 0.0
    for base, tip in ((5,8),(9,12),(13,16),(17,20)):
        points[base].z = -.035
        points[tip].z = -.125 if opened else -.02
    return points


class UpwardPalmTests(unittest.TestCase):
    def classify(self, hands):
        return get_gesture(NS(multi_face_landmarks=[NS(landmark=face())]),
                           NS(multi_hand_landmarks=[NS(landmark=h) for h in hands]),960,720)

    def test_upward_palms_with_foreshortened_fingers(self):
        # The same XY projection would look closed without depth landmarks.
        self.assertEqual(self.classify([palm_up(.2,.6),palm_up(.8,.7)]),'sixseven')

    def test_upward_motion_sequence(self):
        detector=SixSevenMotion()
        for now, left, right in ((0,.6,.7),(.1,.64,.66),(.2,.7,.6)):
            hands=[palm_up(.2,left),palm_up(.8,right)]
            result=detector.filter(self.classify(hands),hands,now)
        self.assertEqual(result,'sixseven')

    def test_two_closed_fists_still_rejected(self):
        self.assertNotEqual(self.classify([palm_up(.2,.6,False),palm_up(.8,.7,False)]),'sixseven')

    def test_upward_palms_still_require_motion(self):
        detector=SixSevenMotion()
        hands=[palm_up(.2,.65),palm_up(.8,.65)]
        for i in range(20):
            self.assertEqual(detector.filter(self.classify(hands),hands,i*.05),'idle')


class ForwardFingersTests(unittest.TestCase):
    def test_camera_pointing_fingers_use_world_reach(self):
        # In the image both hands look compressed; world joints show fingers
        # extending toward the lens with palms horizontal.
        image_hands=[hand(.2,.6,False),hand(.8,.7,False)]
        world_hands=[]
        for _ in image_hands:
            points=[NS(x=0.,y=0.,z=0.) for _ in range(21)]
            for base,tip,x in ((5,8,-.03),(9,12,-.01),(13,16,.01),(17,20,.03)):
                points[base]=NS(x=x,y=0.,z=-.04)
                points[tip]=NS(x=x,y=.005,z=-.10)
            world_hands.append(points)
        result=get_gesture(NS(multi_face_landmarks=[NS(landmark=face())]),
                           NS(multi_hand_landmarks=[NS(landmark=h) for h in image_hands],
                              multi_hand_world_landmarks=[NS(landmark=h) for h in world_hands]),960,720)
        self.assertEqual(result,'sixseven')


class ReferencePoseTests(unittest.TestCase):
    def classify(self, hands):
        return get_gesture(NS(multi_face_landmarks=[NS(landmark=face())]),
                           NS(multi_hand_landmarks=[NS(landmark=h) for h in hands]),1280,720)

    def test_low_visible_palm_is_candidate(self):
        self.assertEqual(self.classify([palm_up(.72,.88)]),'sixseven')

    def test_one_hand_bounce_triggers(self):
        detector=SixSevenMotion()
        for now,y in ((0,.88),(.1,.84),(.2,.88)):
            hands=[palm_up(.72,y)]
            result=detector.filter(self.classify(hands),hands,now)
        self.assertEqual(result,'sixseven')

    def test_single_static_palm_stays_idle(self):
        detector=SixSevenMotion()
        hands=[palm_up(.72,.88)]
        for i in range(20):
            self.assertEqual(detector.filter(self.classify(hands),hands,i*.05),'idle')
