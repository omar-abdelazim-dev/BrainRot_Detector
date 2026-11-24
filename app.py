"""Brainrot Mirror: macOS webcam gesture-to-meme desktop window.

Threads: the camera thread reads frames, the vision thread runs MediaPipe and
gesture recognition, and this (main) thread draws the window at camera rate.
"""
import sys
import argparse
import time
from pathlib import Path
import cv2
from mirror_ui import MirrorUI, WIDTH, HEIGHT
from camera_capture import CameraStream, open_camera
from gesture_recognition import ConfigError, DETECTORS, load_gesture_config, pixel
from meme_engine import MemeLibrary, load_reactions

try:
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        raise ImportError('This MediaPipe release no longer includes solutions')
    from vision import VisionWorker
except ImportError as exc:
    print('MediaPipe is not available:', exc)
    print('Install dependencies using ./START.command (Python 3.11 recommended).')
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
MEMES = ROOT / 'memes'
CONFIG = ROOT / 'config'
FACE_MODEL = ROOT / 'models' / 'face_landmarker.task'
MEMES.mkdir(exist_ok=True)
FACE_POINTS = (13, 14, 61, 291, 152, 33, 263)  # drawn when landmarks are on
NOSE_TIP = 1


def load_configs():
    gestures = load_gesture_config(CONFIG / 'gestures.json')
    reactions = load_reactions(CONFIG / 'reactions.json', set(DETECTORS))
    return gestures, reactions


def enhance(frame):
    """Light unsharp mask for the displayed image (detection uses the raw frame)."""
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.3)
    return cv2.addWeighted(frame, 1.45, blurred, -.45, 0)


class FrameSource:
    """What the vision worker sees: the live camera, or the frozen frame when paused."""
    def __init__(self, stream):
        self.stream = stream
        self.frozen = None   # (frame_id, frame)

    def __call__(self):
        return self.frozen or self.stream.latest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, help='Use a specific camera index (0 or 1).')
    parser.add_argument('--stats', action='store_true', help='Print display/detection rates every 5 seconds.')
    parser.add_argument('--trace', metavar='FILE', help='Write per-frame gesture diagnostics (TSV) to FILE.')
    args = parser.parse_args()
    try:
        gesture_config, reactions = load_configs()
    except ConfigError as exc:
        print(exc)
        sys.exit(1)
    if not FACE_MODEL.exists():
        print(f'Face model missing: {FACE_MODEL}\nRun ./START.command to download it.')
        sys.exit(1)
    try:
        stream = CameraStream(*open_camera(args.camera))
    except RuntimeError as exc:
        print(exc)
        sys.exit(2)

    source = FrameSource(stream)
    worker = VisionWorker(source, FACE_MODEL, gesture_config, args.trace)
    try:
        worker.start()
    except Exception as exc:
        print('Could not start tracking:', exc)
        stream.stop()
        sys.exit(1)

    ui = MirrorUI(MemeLibrary(MEMES, reactions))
    window = 'Brainrot Mirror'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, WIDTH*3//4, HEIGHT*3//4)
    state = {'manual': None, 'landmarks': False, 'paused': False, 'enhance': True}
    last_drawn, last_frame_time, fps = None, time.monotonic(), 0.0
    next_stats = time.monotonic() + 5

    try:
        while True:
            if stream.failed:
                print('Camera stopped producing frames.')
                break
            if worker.error:
                print('Tracking stopped:', worker.error)
                break
            frame_id, frame = source()
            fresh = frame_id != last_drawn
            if fresh or state['paused']:
                if fresh and not state['paused']:
                    now = time.monotonic()
                    fps = .9*fps + .1/max(now-last_frame_time, 1e-3)
                    last_frame_time = now
                last_drawn = frame_id
                result = worker.result()
                display = frame
                if display is not None:
                    if state['enhance']:
                        display = enhance(display)
                    if state['landmarks']:
                        display = display.copy() if display is frame else display
                        h, w = display.shape[:2]
                        for hand in result.hand_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(display, hand, mp.solutions.hands.HAND_CONNECTIONS)
                        if result.face:
                            for ix in FACE_POINTS:
                                cv2.circle(display, tuple(pixel(result.face.landmarks[ix], w, h).astype(int)),
                                           3, (39, 232, 255), -1)
                face_x = result.face.landmarks[NOSE_TIP].x if result.face else None
                shown = state['manual'] or result.active
                combined = ui.compose(display, shown, bool(state['manual']), state['landmarks'], state['enhance'],
                                      result.face is not None, len(result.hand_landmarks), fps,
                                      state['paused'], face_x, time.monotonic())
                cv2.imshow(window, combined)
            if args.stats and time.monotonic() >= next_stats:
                next_stats += 5
                r = worker.result()
                print(f'[stats] display {fps:.1f} fps, detection {r.rate:.1f}/s, gesture {r.active}', flush=True)
            key = cv2.waitKey(1 if fresh else 4) & 0xFF
            if key in (ord('q'), 27):
                break
            if ord('1') <= key < ord('1') + len(ui.reactions):
                state['manual'] = ui.reactions[key-ord('1')].gesture
            elif key == ord('a'):
                state['manual'] = None
                worker.reset()
            elif key == ord('l'):
                state['landmarks'] = not state['landmarks']
            elif key == ord('e'):
                state['enhance'] = not state['enhance']
            elif key == ord(' '):
                state['paused'] = not state['paused']
                source.frozen = stream.latest() if state['paused'] else None
            elif key == ord('r'):
                try:
                    gesture_config, reactions = load_configs()
                except ConfigError as exc:
                    print(f'Config not reloaded: {exc}')
                else:
                    worker.reconfigure(gesture_config)
                    ui.library.set_reactions(reactions)
                    print('Reloaded config and memes.', flush=True)
                ui.reload()
            elif key == ord('c'):
                index = stream.index
                stream.stop()
                try:
                    stream = CameraStream(*open_camera(1-index if index in (0, 1) else 0))
                except RuntimeError:
                    try:
                        stream = CameraStream(*open_camera(index))
                    except RuntimeError as exc:
                        print(exc)
                        break
                source.stream, source.frozen = stream, None
                state['paused'] = False
                state['manual'] = None
                worker.reset()
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        worker.stop()
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
