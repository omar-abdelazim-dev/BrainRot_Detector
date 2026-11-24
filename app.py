"""Brainrot Mirror: macOS webcam gesture-to-meme desktop window."""
import sys
import argparse
import time
from pathlib import Path
from collections import deque
import cv2
import numpy as np
from mirror_ui import MirrorUI, REACTIONS, WIDTH, HEIGHT
from camera_capture import open_camera
from gesture_motion import SixSevenMotion

try:
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        raise ImportError('This MediaPipe release no longer includes solutions')
except ImportError as exc:
    print('MediaPipe is not available:', exc)
    print('Install dependencies using ./START.command (Python 3.11 recommended).')
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
MEMES = ROOT / 'memes'
MEMES.mkdir(exist_ok=True)
def pixel(point, w, h):
    return np.array([point.x*w, point.y*h], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a-b))
ض

def is_sixseven(face, hands, w, h, world_hands=None):
    """One or two open hands, including low, upward-facing palms."""
    if len(hands) not in (1, 2):
        return False
    centers = [np.mean([pixel(hand[i], w, h) for i in (0, 5, 9, 13, 17)], axis=0)
               for hand in hands]
    # The reference pose is below the chin, near the bottom of the frame.
    # Do not derive the allowed height from face size/distance to the camera.
    if any(center[1] > .98*h for center in centers):
        return False
    palm_sizes = [max(1, dist(pixel(hand[5], w, h), pixel(hand[17], w, h))) for hand in hands]
    if len(hands) == 2 and dist(*centers) < .9*max(palm_sizes):
        return False
    for hand_index, hand in enumerate(hands):
        # A palm facing upward is foreshortened in the camera image: the fingers
        # extend in depth, so their 2D tips can look as close as a closed fist.
        # MediaPipe z uses the same approximate scale as normalized x.
        def position3d(index):
            if world_hands is not None and len(world_hands) == len(hands):
                point = world_hands[hand_index][index]
                return np.array([point.x, point.y, point.z], dtype=np.float32)
            point = hand[index]
            return np.array([point.x*w, point.y*h, getattr(point, 'z', 0.0)*w], dtype=np.float32)

        # Metric world landmarks preserve finger reach when fingertips point
        # into the lens; image-space depth can collapse in this orientation.
        wrist = position3d(0)
        extended = sum(dist(position3d(tip), wrist) >
                       1.05*dist(position3d(base), wrist)
                       for tip, base in ((8, 5), (12, 9), (16, 13), (20, 17)))
        if extended < 2:
            return False
    return True


def get_gesture(face_result, hands_result, w, h):
    face = face_result.multi_face_landmarks[0].landmark if face_result.multi_face_landmarks else None
    hands = [hand.landmark for hand in hands_result.multi_hand_landmarks or []]
    world_hands = [hand.landmark for hand in
                   getattr(hands_result, 'multi_hand_world_landmarks', None) or []]
    # Evaluate the two-hand pose before a fingertip near the chin can steal it.
    if is_sixseven(face, hands, w, h, world_hands):
        if face is not None and len(hands) == 2:
            centers = [np.mean([pixel(hand[i], w, h) for i in (0, 5, 9, 13, 17)], axis=0)
                       for hand in hands]
            # Raised palms beside the head are distinct from chest-height 6-7.
            if (all(center[1] < face[1].y*h for center in centers)
                    and min(c[0] for c in centers) < face[234].x*w
                    and max(c[0] for c in centers) > face[454].x*w):
                return 'cinema'
        return 'sixseven'
    if len(hands) == 1:
        hand = hands[0]
        wrist = pixel(hand[0], w, h)
        folded = sum(dist(pixel(hand[tip], w, h), wrist) <
                     1.05*dist(pixel(hand[base], w, h), wrist)
                     for tip, base in ((8, 5), (12, 9), (16, 13), (20, 17)))
        if folded >= 3:
            if face is not None:
                chin = pixel(face[152], w, h)
                face_width = max(1, dist(pixel(face[234], w, h), pixel(face[454], w, h)))
                knuckles = np.mean([pixel(hand[i], w, h) for i in (5, 9, 13, 17)], axis=0)
                if (abs(knuckles[0]-chin[0]) < .40*face_width
                        and -.20*face_width < knuckles[1]-chin[1] < .50*face_width):
                    return 'calm'
            return 'handshake'
    if face is None:
        return 'idle'

    p = lambda idx: pixel(face[idx], w, h)
    face_width = max(1, dist(p(234), p(454)))
    mouth = (p(13) + p(14))/2
    mouth_gap = dist(p(13), p(14)) / face_width
    mouth_width = dist(p(61), p(291)) / face_width
    jaw = p(152)

    for hand in hands:
        tip = pixel(hand[8], w, h)
        pip = pixel(hand[6], w, h)
        # Finger on lips = shush; finger near jaw = mewing.
        if dist(tip, mouth) < .145*face_width and dist(tip, pip) > .075*face_width:
            return 'shush'
        if dist(tip, jaw) < .19*face_width:
            return 'mewing'

    if mouth_gap > .095:
        return 'shock'
    if mouth_width > .49 and mouth_gap < .085:
        return 'smile'
    eyes = p(263)-p(33)
    if abs(float(eyes[1])) > abs(float(eyes[0])) * .18:
        return 'tilt'
    return 'idle'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, help='Use a specific camera index (0 or 1).')
    args = parser.parse_args()
    try:
        camera, camera_index = open_camera(args.camera)
    except RuntimeError as exc:
        print(exc)
        sys.exit(2)
    # Keep the device default format and resize in software.
    history = deque(maxlen=7)
    active = 'idle'
    last_change = time.monotonic()
    motion = SixSevenMotion()
    ui = MirrorUI(MEMES)
    window = 'Brainrot Mirror'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, WIDTH, HEIGHT)
    state = {'manual': None, 'landmarks': False, 'paused': False}
    previous_frame = None
    last_frame = time.monotonic()
    fps = 0.0

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected = ui.clicked_reaction(x, y)
            if selected:
                state['manual'] = selected

    cv2.setMouseCallback(window, click)
    try:
        with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=.55, min_tracking_confidence=.55) as face_mesh, \
             mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=.55, min_tracking_confidence=.55) as hand_tracker:
            while True:
                if state['paused'] and previous_frame is not None:
                    frame = previous_frame.copy()
                else:
                    ok, frame = camera.read()
                    if not ok:
                        # Allow a short warm-up or temporary capture interruption.
                        for _ in range(20):
                            time.sleep(.05)
                            ok, frame = camera.read()
                            if ok and frame is not None:
                                break
                        if not ok or frame is None:
                            print('Camera stopped producing frames.')
                            break
                    if frame.shape[1] > 1280:
                        height = round(frame.shape[0] * 1280 / frame.shape[1])
                        frame = cv2.resize(frame, (1280, height), interpolation=cv2.INTER_AREA)
                    frame = cv2.flip(frame, 1)
                    previous_frame = frame.copy()
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                face_result = face_mesh.process(rgb)
                hands_result = hand_tracker.process(rgb)
                kind = get_gesture(face_result, hands_result, w, h)
                now = time.monotonic()
                kind = motion.filter(kind, [hand.landmark for hand in hands_result.multi_hand_landmarks or []], now)
                history.append(kind)
                if history.count(kind) >= 5 and kind != active and time.monotonic()-last_change > .40:
                    active = kind
                    last_change = time.monotonic()
                # Visualize hand landmarks.
                if state['landmarks'] and hands_result.multi_hand_landmarks:
                    for hand in hands_result.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                # Visualize a few face features.
                if state['landmarks'] and face_result.multi_face_landmarks:
                    for ix in (13, 14, 61, 291, 152, 33, 263):
                        pt = pixel(face_result.multi_face_landmarks[0].landmark[ix], w, h).astype(int)
                        cv2.circle(frame, tuple(pt), 3, (39, 232, 255), -1)
                now = time.monotonic()
                fps = .9*fps + .1 / max(now-last_frame, .001)
                last_frame = now
                shown = state['manual'] or active
                combined = ui.compose(frame, shown, bool(state['manual']), state['landmarks'],
                                      bool(face_result.multi_face_landmarks),
                                      len(hands_result.multi_hand_landmarks or []), fps, state['paused'])
                cv2.imshow(window, combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key in range(ord('1'), ord('1') + len(REACTIONS)):
                    state['manual'] = tuple(REACTIONS)[key-ord('1')]
                elif key == ord('a'):
                    motion.reset()
                    state['manual'] = None
                    history.clear()
                    active = 'idle'
                elif key == ord('l'):
                    state['landmarks'] = not state['landmarks']
                elif key == ord(' '):
                    state['paused'] = not state['paused']
                elif key == ord('r'):
                    ui.reload()
                elif key == ord('c'):
                    camera.release()
                    try:
                        camera, camera_index = open_camera(1-camera_index if camera_index in (0, 1) else 0)
                    except RuntimeError:
                        try:
                            camera, camera_index = open_camera(camera_index)
                        except RuntimeError as exc:
                            print(exc)
                            break
                    previous_frame = None
                    state['paused'] = False
                    state['manual'] = None
                    history.clear()
                    active = 'idle'
                    motion.reset()
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
١