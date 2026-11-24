"""Window rendering: camera stage, meme card overlay, status pills.

Static chrome is drawn once with Pillow (crisp system fonts) and cached;
each frame only the camera image, status pills and meme card are composited
with OpenCV. Camera frames stay local.
"""
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WIDTH, HEIGHT = 1920, 1180
MARGIN = 48
STAGE = (MARGIN, 96, WIDTH-MARGIN, 96+1026)          # 1824x1026, 16:9
HINTS_Y = 1152
MAX_PREVIEW_KEYS = 9                                  # keys 1-9

BG = '#0b0c10'
SURFACE = '#15171e'
LINE = '#262a35'
TEXT = '#f4f5f8'
MUTED = '#8a90a2'
LIME = '#c3f879'

# Meme card
CARD_MAX_W, CARD_MAX_H = .40, .68   # fraction of the stage
CARD_BORDER, CARD_RADIUS, CARD_PAD, CARD_INSET = 6, 26, 44, 56
ENTER_SECONDS, EXIT_SECONDS = .28, .18

SF = Path('/System/Library/Fonts/SFNS.ttf')
FALLBACK = {False: '/System/Library/Fonts/Supplemental/Arial.ttf',
            True: '/System/Library/Fonts/Supplemental/Arial Bold.ttf'}


@lru_cache(maxsize=64)
def font(size, weight='Regular'):
    try:
        f = ImageFont.truetype(str(SF), size)
        f.set_variation_by_name(weight)
        return f
    except (OSError, ValueError):
        pass
    try:
        return ImageFont.truetype(FALLBACK[weight in ('Semibold', 'Bold', 'Heavy', 'Black')], size)
    except OSError:
        return ImageFont.load_default()


def hex_rgb(color):
    color = color.lstrip('#')
    return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    """Blend hex colour ``a`` toward ``b`` by ``t``."""
    ca, cb = hex_rgb(a), hex_rgb(b)
    return '#' + ''.join(f'{round(x + (y-x)*t):02x}' for x, y in zip(ca, cb))


def rounded_mask(size, radius):
    mask = Image.new('L', size)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0]-1, size[1]-1), radius, fill=255)
    return mask


def ease_out_back(t, overshoot=1.6):
    t -= 1
    return 1 + t*t*((overshoot+1)*t + overshoot)


@lru_cache(maxsize=256)
def pill(text, fg=TEXT, bg=SURFACE, dot=None, height=36, on=BG):
    """Rounded status pill as a BGR array drawn over the ``on`` colour."""
    f = font(13, 'Semibold')
    text_w = f.getbbox(text)[2]
    pad, dot_space = 16, (18 if dot else 0)
    w = pad*2 + dot_space + text_w
    img = Image.new('RGB', (w, height), on)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w-1, height-1), height//2, fill=bg)
    if dot:
        cy = height//2
        d.ellipse((pad, cy-4, pad+8, cy+4), fill=dot)
    d.text((pad+dot_space, height//2), text, font=f, fill=fg, anchor='lm')
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


@lru_cache(maxsize=16)
def pill_alpha(width, height):
    return np.asarray(rounded_mask((width, height), height//2))


def blend(dst, bgr, alpha, x, y, clip):
    """Alpha-blend ``bgr``/``alpha`` (uint8) onto ``dst`` at (x, y) inside ``clip``."""
    left, top, right, bottom = clip
    h, w = alpha.shape
    x0, y0, x1, y1 = max(x, left), max(y, top), min(x+w, right), min(y+h, bottom)
    if x0 >= x1 or y0 >= y1:
        return
    src = bgr[y0-y:y1-y, x0-x:x1-x]
    weight = alpha[y0-y:y1-y, x0-x:x1-x].astype(np.float32) / 255
    roi = dst[y0:y1, x0:x1]
    roi[:] = cv2.blendLinear(src, roi, weight, 1 - weight)


class MirrorUI:
    def __init__(self, library):
        self.library = library
        self.chrome = {}
        stage_w, stage_h = STAGE[2]-STAGE[0], STAGE[3]-STAGE[1]
        self.stage_size = (stage_w, stage_h)
        self.stage_mask = np.asarray(rounded_mask(self.stage_size, 22))
        self.stage_bg = np.full((stage_h, stage_w, 3), hex_rgb(SURFACE)[::-1], np.uint8)
        self.card_bases = {}
        self.cards = OrderedDict()   # (gesture, frame index) -> (bgr, alpha)
        self.current = None          # (gesture, since, side)
        self.leaving = None          # (gesture, since, side, frame index)
        self.prewarm()

    @property
    def reactions(self):
        """Reactions reachable with the 1-9 preview keys."""
        return self.library.reactions[:MAX_PREVIEW_KEYS]

    def reload(self):
        self.library.reload()
        self.chrome.clear()
        self.card_bases.clear()
        self.cards.clear()
        self.current = self.leaving = None
        self.prewarm()

    def prewarm(self):
        """Build each meme's first card frame now so the first pop-in doesn't stutter."""
        for reaction in self.library.reactions:
            if self.library.meme_for(reaction.gesture):
                self.card(reaction.gesture, 0)

    # ------------------------------------------------------------- chrome
    def artwork(self, landmarks, enhance):
        key = (landmarks, enhance)
        if key in self.chrome:
            return self.chrome[key]
        canvas = Image.new('RGB', (WIDTH, HEIGHT), BG)
        d = ImageDraw.Draw(canvas)
        d.text((MARGIN, 52), 'BRAINROT MIRROR', font=font(30, 'Heavy'), fill=TEXT, anchor='lm')
        title_w = font(30, 'Heavy').getbbox('BRAINROT MIRROR')[2]
        d.text((MARGIN + title_w + 18, 54), 'Your face. Internet energy.', font=font(16), fill=MUTED, anchor='lm')
        d.rounded_rectangle(STAGE, 22, fill=SURFACE)


        x = MARGIN
        for keycap, text, lit in (('1–9', 'Preview', False), ('A', 'Auto', False), ('Space', 'Pause', False),
                                  ('L', 'Landmarks', landmarks), ('E', 'Enhance', enhance),
                                  ('R', 'Reload', False), ('C', 'Camera', False), ('Q', 'Quit', False)):
            f = font(12, 'Semibold')
            kw = f.getbbox(keycap)[2] + 16
            d.rounded_rectangle((x, HINTS_Y-11, x+kw, HINTS_Y+11), 6,
                                fill=mix(SURFACE, LIME, .2) if lit else SURFACE, outline=LIME if lit else LINE)
            d.text((x + kw//2, HINTS_Y), keycap, font=f, fill=LIME if lit else TEXT, anchor='mm')
            d.text((x + kw + 8, HINTS_Y), text, font=font(13), fill=MUTED, anchor='lm')
            x += kw + 8 + font(13).getbbox(text)[2] + 26
        d.text((WIDTH-MARGIN, HINTS_Y), 'Frames never leave this Mac', font=font(13), fill=MUTED, anchor='rm')
        result = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
        self.chrome[key] = result
        return result

    def draw_pills(self, output, items):
        x = WIDTH - MARGIN
        for item in reversed(items):
            img = pill(*item)
            x -= img.shape[1]
            output[34:34+img.shape[0], x:x+img.shape[1]] = img
            x -= 10

    # ------------------------------------------------------------- camera
    def draw_stage(self, output, frame):
        stage_w, stage_h = self.stage_size
        view = self.stage_bg.copy()
        if frame is not None:
            h, w = frame.shape[:2]
            scale = max(stage_w/w, stage_h/h)          # cover: fill the stage
            resized = cv2.resize(frame, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_LINEAR)
            y, x = (resized.shape[0]-stage_h)//2, (resized.shape[1]-stage_w)//2
            view = resized[y:y+stage_h, x:x+stage_w]
        roi = output[STAGE[1]:STAGE[3], STAGE[0]:STAGE[2]]
        cv2.copyTo(view, self.stage_mask, roi)

    # ------------------------------------------------------------- meme card
    def card_base(self, gesture, size, accent):
        key = (gesture, size)
        if key not in self.card_bases:
            cw, ch = size
            outer = (cw + 2*CARD_BORDER, ch + 2*CARD_BORDER)
            canvas_size = (outer[0] + 2*CARD_PAD, outer[1] + 2*CARD_PAD)
            shadow = Image.new('L', canvas_size)
            ImageDraw.Draw(shadow).rounded_rectangle(
                (CARD_PAD, CARD_PAD+16, CARD_PAD+outer[0], CARD_PAD+16+outer[1]), CARD_RADIUS, fill=170)
            shadow = shadow.filter(ImageFilter.GaussianBlur(20))
            base = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
            base.putalpha(shadow)
            border = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
            ImageDraw.Draw(border).rounded_rectangle(
                (CARD_PAD, CARD_PAD, CARD_PAD+outer[0]-1, CARD_PAD+outer[1]-1), CARD_RADIUS,
                fill=hex_rgb(accent) + (255,))
            self.card_bases[key] = Image.alpha_composite(base, border)
        return self.card_bases[key]

    def card(self, gesture, index):
        key = (gesture, index)
        if key in self.cards:
            self.cards.move_to_end(key)
            return self.cards[key]
        meme = self.library.meme_for(gesture)
        reaction = self.library.reaction_for(gesture)
        source = meme.frames[index]
        max_w, max_h = self.stage_size[0]*CARD_MAX_W, self.stage_size[1]*CARD_MAX_H
        scale = min(max_w/source.width, max_h/source.height)
        size = (round(source.width*scale), round(source.height*scale))
        card = self.card_base(gesture, size, reaction.accent if reaction else TEXT).copy()
        content = source.resize(size, Image.Resampling.LANCZOS)
        offset = CARD_PAD + CARD_BORDER
        card.paste(content, (offset, offset), rounded_mask(size, CARD_RADIUS - CARD_BORDER))
        rgba = np.asarray(card)
        result = (np.ascontiguousarray(rgba[..., 2::-1]), np.ascontiguousarray(rgba[..., 3]))
        self.cards[key] = result
        while len(self.cards) > 90:
            self.cards.popitem(last=False)
        return result

    def draw_card(self, output, gesture, index, side, scale, opacity):
        bgr, alpha = self.card(gesture, index)
        if scale != 1:
            size = (max(1, round(alpha.shape[1]*scale)), max(1, round(alpha.shape[0]*scale)))
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR)
        if opacity < 1:
            alpha = (alpha * opacity).astype(np.uint8)
        h, w = alpha.shape
        reach = CARD_INSET - CARD_PAD*scale
        x = round(STAGE[0] + reach) if side == 'left' else round(STAGE[2] - reach - w)
        y = (STAGE[1] + STAGE[3])//2 - h//2
        blend(output, bgr, alpha, x, y, STAGE)

    def update_cards(self, output, gesture, face_x, now):
        shown = gesture if gesture != 'idle' and self.library.meme_for(gesture) else None
        if (self.current[0] if self.current else None) != shown:
            if self.current:
                old, since, side = self.current
                self.leaving = (old, now, side, self.frame_index(old, now - since))
            # Put the card on the side away from the face.
            side = 'left' if face_x is not None and face_x > .5 else 'right'
            self.current = (shown, now, side) if shown else None
        if self.leaving:
            old, since, side, index = self.leaving
            p = (now - since) / EXIT_SECONDS
            if p >= 1 or self.library.meme_for(old) is None:
                self.leaving = None
            else:
                self.draw_card(output, old, index, side, 1 - .06*p, 1 - p)
        if self.current:
            gesture, since, side = self.current
            t = now - since
            if t < ENTER_SECONDS:
                scale = .7 + .3*ease_out_back(t / ENTER_SECONDS)
                opacity = min(1.0, t / (ENTER_SECONDS*.5))
            else:
                scale, opacity = 1, 1
            self.draw_card(output, gesture, self.frame_index(gesture, t), side, scale, opacity)

    def frame_index(self, gesture, elapsed):
        meme = self.library.meme_for(gesture)
        return meme.frame_index(elapsed) if meme else 0

    # ------------------------------------------------------------- frame
    def compose(self, frame, active='idle', manual=False, landmarks=False, enhance=True,
                face=False, hands=0, fps=0, paused=False, face_x=None, now=0.0):
        output = self.artwork(landmarks, enhance).copy()
        self.draw_stage(output, frame)
        self.update_cards(output, active, face_x, now)
        if paused:
            badge = pill('PAUSED  ·  SPACE TO RESUME', BG, LIME, None, 36, '#000000')
            alpha = pill_alpha(badge.shape[1], badge.shape[0])
            blend(output, badge, alpha, (STAGE[0] + STAGE[2] - badge.shape[1]) // 2, STAGE[1] + 24, STAGE)
        self.draw_pills(output, [
            ('PREVIEW  ·  A FOR AUTO', BG, '#f5dc85', None) if manual else ('AUTO DETECT', TEXT, SURFACE, LIME),
            ('FACE' if face else 'NO FACE', TEXT, SURFACE, '#79f8c3' if face else '#5b6170'),
            (f'{hands} HAND' + ('' if hands == 1 else 'S'), TEXT, SURFACE, None),
            (f'{fps:.0f} FPS', TEXT, SURFACE, None),
            ('ON-DEVICE', LIME, '#1a2615', LIME),
        ])
        return output
