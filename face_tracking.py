"""Face tracking with the MediaPipe Tasks Face Landmarker.

Returns the 478 face landmarks plus 52 expression scores ("blendshapes",
e.g. jawOpen, browInnerUp, mouthSmileLeft). Runs on-device; frames never
leave the Mac.
"""
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
             'face_landmarker/float16/latest/face_landmarker.task')


@dataclass
class FaceObservation:
    landmarks: list      # NormalizedLandmark (.x, .y, .z in 0..1 image space)
    blendshapes: dict    # name -> score in 0..1


class FaceTracker:
    def __init__(self, model_path, min_confidence=.55):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f'Face model not found at {model_path}. Run ./START.command, or download it:\n'
                f'  curl -L -o {model_path} {MODEL_URL}')
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=True)
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, rgb, now_seconds):
        """Track the face in an RGB uint8 frame; ``now_seconds`` is monotonic."""
        # VIDEO mode requires strictly increasing timestamps.
        timestamp = max(int(now_seconds * 1000), self._last_ms + 1)
        self._last_ms = timestamp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp)
        if not result.face_landmarks:
            return None
        shapes = {c.category_name: c.score for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
        return FaceObservation(result.face_landmarks[0], shapes)

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
