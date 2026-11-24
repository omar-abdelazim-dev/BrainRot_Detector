"""Background vision worker: face + hand tracking and gesture recognition.

Runs on its own thread so the window keeps drawing camera frames at the
camera's rate while MediaPipe works. The MediaPipe graphs are created and
used only on this thread. The UI reads the newest :class:`VisionResult`.
"""
import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp

from face_tracking import FaceTracker
from gesture_motion import SixSevenMotion, resolve
from gesture_recognition import FrameLandmarks, GestureRecognizer
from reaction_manager import GestureStabilizer


@dataclass
class VisionResult:
    frame_id: int = 0
    raw: str = 'idle'            # per-frame classification
    active: str = 'idle'         # stabilized gesture
    face: object = None          # FaceObservation or None
    hand_landmarks: list = field(default_factory=list)  # MediaPipe protos, for drawing
    rate: float = 0.0            # detections per second


def outranking(config):
    """Gestures that may interrupt a confirmed 6-7 bounce (higher priority)."""
    spec = config.spec('sixseven')
    floor = spec.priority if spec else float('inf')
    return {s.id for s in config.specs if s.priority > floor and s.enabled}


class VisionWorker:
    def __init__(self, frame_source, face_model, gesture_config, trace_path=None):
        """``frame_source()`` returns ``(frame_id, bgr_frame)``; the worker only
        processes a frame id once."""
        self._source = frame_source
        self._face_model = face_model
        self._config = gesture_config
        self._trace_path = trace_path
        self._pending_config = None
        self._reset_requested = False
        self._result = VisionResult()
        self._lock = threading.Lock()
        self._running = True
        self._ready = threading.Event()
        self.error = None
        self._thread = threading.Thread(target=self._run, name='vision', daemon=True)

    def start(self, timeout=20):
        """Start the thread and wait until the models are loaded."""
        self._thread.start()
        self._ready.wait(timeout)
        if self.error:
            raise self.error

    # Called from the UI thread; applied by the worker between frames.
    def reconfigure(self, gesture_config):
        self._pending_config = gesture_config

    def reset(self):
        self._reset_requested = True

    def result(self):
        with self._lock:
            return self._result

    def stop(self):
        self._running = False
        self._thread.join(timeout=3)

    def _run(self):
        try:
            face_tracker = FaceTracker(self._face_model)
            hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=.55,
                                             min_tracking_confidence=.55)
        except Exception as exc:  # surfaced to the UI thread by start()
            self.error = exc
            self._ready.set()
            return
        recognizer = GestureRecognizer(self._config)
        outranks = outranking(self._config)
        trace = open(self._trace_path, 'w', buffering=1) if self._trace_path else None
        if trace:
            trace.write('t\thands\tmatches\traw\tactive\tpalm_y\tconfirmed\n')
        stabilizer = GestureStabilizer(self._config)
        motion = SixSevenMotion()
        self._ready.set()
        last_id, rate, last_time = None, 0.0, time.monotonic()
        try:
            while self._running:
                if self._pending_config is not None:
                    config, self._pending_config = self._pending_config, None
                    recognizer.config = config
                    stabilizer.configure(config)
                    outranks = outranking(config)
                if self._reset_requested:
                    self._reset_requested = False
                    motion.reset()
                    stabilizer.reset()
                frame_id, frame = self._source()
                if frame is None or frame_id == last_id:
                    time.sleep(.004)
                    continue
                last_id = frame_id
                h, w = frame.shape[:2]
                now = time.monotonic()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = face_tracker.process(rgb, now)
                rgb.flags.writeable = False
                hands_result = hands.process(rgb)

                landmarks = FrameLandmarks.from_results(face, hands_result, w, h)
                matches = recognizer.matches(landmarks)
                raw = resolve(matches, motion, landmarks.hands, now, outranks)
                active = stabilizer.update(raw, now)

                if trace:
                    palm_y = ','.join(f'{sum(h[i].y for i in (0, 5, 9, 13, 17))/5:.3f}' for h in landmarks.hands)
                    trace.write(f'{now:.3f}\t{len(landmarks.hands)}\t{",".join(matches) or "-"}\t{raw}\t'
                                f'{active}\t{palm_y or "-"}\t{int(motion.confirmed(now))}\n')
                done = time.monotonic()
                rate = .9*rate + .1/max(done-last_time, 1e-3)
                last_time = done
                with self._lock:
                    self._result = VisionResult(frame_id, raw, active, face,
                                                list(hands_result.multi_hand_landmarks or []), rate)
        except Exception as exc:
            self.error = exc
        finally:
            if trace:
                trace.close()
            face_tracker.close()
            hands.close()
