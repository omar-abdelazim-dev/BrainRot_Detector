"""Select a usable local camera without changing its native capture format."""
import sys
import time
import cv2


def open_camera(preferred=None):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    indices = [preferred] if preferred is not None else [0, 1]
    for index in indices:
        camera = cv2.VideoCapture(index, backend)
        if not camera.isOpened():
            camera.release()
            continue
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            ok, frame = camera.read()
            if ok and frame is not None and frame.size:
                # Some Continuity/virtual cameras open successfully but emit zeros.
                if frame.max() > 8:
                    print(f'Using camera {index}', flush=True)
                    return camera, index
            time.sleep(.03)
        camera.release()
        print(f'Camera {index}: no usable video frames.', flush=True)
    raise RuntimeError('No usable camera video. Check the lens, camera permissions, and other camera apps.')
