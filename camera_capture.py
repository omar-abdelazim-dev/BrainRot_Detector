"""Select a usable local camera and read it on a background thread."""
import itertools
import sys
import threading
import time
import cv2

# Frame ids stay unique across camera switches.
_FRAME_IDS = itertools.count(1)


def open_camera(preferred=None):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    indices = [preferred] if preferred is not None else [0, 1]
    for index in indices:
        camera = cv2.VideoCapture(index, backend)
        if not camera.isOpened():
            camera.release()
            continue
        # Ask for the best mode; the driver falls back to what the sensor supports.
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera.set(cv2.CAP_PROP_FPS, 30)
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            ok, frame = camera.read()
            if ok and frame is not None and frame.size:
                # Some Continuity/virtual cameras open successfully but emit zeros.
                if frame.max() > 8:
                    print(f'Using camera {index} ({frame.shape[1]}x{frame.shape[0]})', flush=True)
                    return camera, index
            time.sleep(.03)
        camera.release()
        print(f'Camera {index}: no usable video frames.', flush=True)
    raise RuntimeError('No usable camera video. Check the lens, camera permissions, and other camera apps.')


class CameraStream:
    """Continuously reads an opened camera so consumers never wait on I/O.

    ``latest()`` returns ``(frame_id, frame)`` for the newest mirrored BGR
    frame. Frames are treated as read-only by consumers; each read produces a
    new array, so no copies are needed.
    """
    MAX_WIDTH = 1920
    FAIL_SECONDS = 1.5

    def __init__(self, camera, index):
        self.camera, self.index = camera, index
        self._lock = threading.Lock()
        self._frame, self._id = None, 0
        self.failed = False
        self._running = True
        self._thread = threading.Thread(target=self._run, name='camera', daemon=True)
        self._thread.start()

    def _run(self):
        last_ok = time.monotonic()
        while self._running:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                if time.monotonic() - last_ok > self.FAIL_SECONDS:
                    self.failed = True
                    return
                time.sleep(.02)
                continue
            last_ok = time.monotonic()
            if frame.shape[1] > self.MAX_WIDTH:
                height = round(frame.shape[0] * self.MAX_WIDTH / frame.shape[1])
                frame = cv2.resize(frame, (self.MAX_WIDTH, height), interpolation=cv2.INTER_AREA)
            frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame, self._id = frame, next(_FRAME_IDS)

    def latest(self):
        with self._lock:
            return self._id, self._frame

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self.camera.release()
