"""Cached desktop UI artwork; camera frames stay local and use OpenCV compositing."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

WIDTH, HEIGHT = 1600, 1030
CAMERA = (32, 166, 992, 886)
PHOTO = (1021, 167, 1567, 885)
REACTIONS = {
    'shush': ('SHHH...', 'Finger to lips', '#c3f879'),
    'mewing': ('MEWING', 'Finger to chin', '#c4a2ff'),
    'shock': ('WHAT?!', 'Open mouth', '#ffac7f'),
    'sixseven': ('6 - 7', 'Palm up + bounce', '#80d9ff'),
    'smile': ('W RIZZ', 'Big smile', '#ffa6d3'),
    'tilt': ('SIDE EYE', 'Tilt your head', '#f5dc85'),
    'cinema': ('ABSOLUTE CINEMA', 'Palms beside head', '#e4e4e4'),
    'handshake': ('CAT HANDSHAKE', 'Hold out a fist', '#ffbc83'),
    'calm': ('STAY CALM', 'Fist under chin', '#9ed9e8'),
}
FONT_DIR = Path('/System/Library/Fonts/Supplemental')

@lru_cache(maxsize=64)
def font(size, bold=False):
    for path in [FONT_DIR / ('Arial Bold.ttf' if bold else 'Arial.ttf'), Path('/System/Library/Fonts/Helvetica.ttc')]:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(draw, xy, text, size=15, color='#9296a5', bold=False):
    draw.text(xy, text, font=font(size, bold), fill=color)


class MirrorUI:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.photos = {}
        self.loaded = {}
        self.tiles = {}
        self.reload()
        width, height = CAMERA[2]-CAMERA[0], CAMERA[3]-CAMERA[1]
        mask = Image.new('L', (width, height))
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, width-1, height-1), 18, fill=255)
        self.mask = np.asarray(mask)

    def reload(self):
        self.photos.clear()
        self.loaded.clear()
        for key in (*REACTIONS, 'idle'):
            for suffix in ('.png', '.jpg', '.jpeg', '.webp', '.gif'):
                path = self.folder / (key + suffix)
                if path.exists():
                    try:
                        with Image.open(path) as source:
                            self.photos[key] = ImageOps.exif_transpose(source).convert('RGB')
                        self.loaded[key] = path.name
                        break
                    except (OSError, ValueError):
                        continue
        # Reuse supplied photos for related expressions until custom ones are added.
        for key, fallback in [('smile', 'mewing'), ('tilt', 'mewing')]:
            if key not in self.photos and fallback in self.photos:
                self.photos[key] = self.photos[fallback]
        self.tiles.clear()

    def place_photo(self, canvas, source, box, radius=14):
        x, y, right, bottom = box
        size = (right-x, bottom-y)
        backdrop = ImageOps.fit(source, size).filter(ImageFilter.GaussianBlur(24))
        backdrop = Image.blend(backdrop, Image.new('RGB', size, '#11131b'), .65)
        fitted = ImageOps.contain(source, size, Image.Resampling.LANCZOS)
        backdrop.paste(fitted, ((size[0]-fitted.width)//2, (size[1]-fitted.height)//2))
        mask = Image.new('L', size)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0]-1, size[1]-1), radius, fill=255)
        canvas.paste(backdrop, (x, y), mask)

    def artwork(self, active, manual, landmarks):
        cache_key = (active, manual, landmarks)
        if cache_key in self.tiles:
            return self.tiles[cache_key]
        canvas = Image.new('RGB', (WIDTH, HEIGHT), '#0d0f14')
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle((32, 27, 72, 67), 12, fill='#c3f879')
        label(d, (41, 34), 'B', 25, '#131a0b', True)
        label(d, (86, 25), 'BRAINROT MIRROR', 27, '#f4f5f8', True)
        label(d, (87, 59), 'Your face. Internet energy.', 14)
        d.rounded_rectangle((1384, 34, 1568, 67), 16, fill='#1c281b')
        d.ellipse((1398, 46, 1406, 54), fill='#c3f879')
        label(d, (1416, 42), 'ON-DEVICE CAMERA', 11, '#c3f879', True)
        d.line((32, 101, 1568, 101), fill='#282b35')
        label(d, (32, 124), '01  /  YOUR MIRROR', 13, '#d4d7e0', True)
        label(d, (805, 124), 'L  /  LANDMARKS ' + ('ON' if landmarks else 'OFF'), 12)
        d.rounded_rectangle(CAMERA, 18, fill='#171a22')
        d.rounded_rectangle((1020, 166, 1568, 886), 20, fill='#000000')
        if active != 'idle' and active in self.photos:
            self.place_photo(canvas, self.photos[active], PHOTO)
        d = ImageDraw.Draw(canvas)
        for index, (key, (title, hint, accent)) in enumerate(REACTIONS.items()):
            x = 32 + index*171
            chosen = active == key
            d.rounded_rectangle((x, 910, x+161, 977), 12,
                                fill='#242b20' if chosen else '#171a22',
                                outline=accent if chosen else '#303442')
            label(d, (x+12, 920), str(index+1), 13, accent, True)
            label(d, (x+34, 918), title, 11, '#f4f5f8', True)
            label(d, (x+34, 947), hint, 10)
        label(d, (32, 999), '1-9  Preview     A  Auto     SPACE  Pause     L  Landmarks     R  Reload photos     C  Switch camera', 11)
        label(d, (1434, 999), 'Q / ESC  Quit', 11)
        result = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
        self.tiles[cache_key] = result
        return result

    def compose(self, frame, active='idle', manual=False, landmarks=False,
                face=False, hands=0, fps=0, paused=False):
        output = self.artwork(active, manual, landmarks).copy()
        h, w = frame.shape[:2]
        left, top, right, bottom = CAMERA
        camera_width, camera_height = right-left, bottom-top
        scale = min(camera_width/w, camera_height/h)
        resized = cv2.resize(frame, (round(w*scale), round(h*scale)))
        view = np.full((camera_height, camera_width, 3), (22, 19, 17), np.uint8)
        y, x = (camera_height-resized.shape[0])//2, (camera_width-resized.shape[1])//2
        view[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
        roi = output[top:bottom, left:right]
        cv2.copyTo(view, self.mask, roi)
        overlay = output[bottom-51:bottom-14, left+16:right-16]
        overlay[:] = cv2.addWeighted(overlay, .22, np.full_like(overlay, (23, 19, 17)), .78, 0)
        text = ('PAUSED' if paused else ('FACE TRACKED' if face else 'LOOK INTO THE CAMERA'))
        text += f'    /    {hands} HANDS    /    {fps:.0f} FPS'
        cv2.circle(output, (left+32, bottom-32), 4, (121, 248, 195) if face else (133, 145, 160), -1)
        cv2.putText(output, text, (left+46, bottom-27), cv2.FONT_HERSHEY_SIMPLEX, .43, (232, 231, 224), 1, cv2.LINE_AA)
        return output

    @staticmethod
    def clicked_reaction(x, y):
        if 910 <= y <= 977:
            index = (x-32)//171
            if 0 <= index < len(REACTIONS) and x <= 32+index*171+161:
                return tuple(REACTIONS)[index]
        return None
