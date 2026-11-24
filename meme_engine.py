"""Reaction config (config/reactions.json) and meme image/GIF loading.

A reaction maps one gesture id to the meme shown for it. Images live in
``memes/``; animated GIFs (and animated WebP) keep every frame and its delay.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, ImageSequence

from gesture_recognition import ConfigError

IMAGE_SUFFIXES = ('.gif', '.png', '.jpg', '.jpeg', '.webp')  # GIF wins for bare names
MAX_FRAMES = 150           # cap memory for very long GIFs
DEFAULT_FRAME_DELAY = .10  # seconds, for GIFs without a delay
MIN_FRAME_DELAY = .02      # browsers treat tiny delays as ~0.1s; clamp the extremes


@dataclass(frozen=True)
class Reaction:
    id: str
    gesture: str
    title: str
    hint: str = ''
    accent: str = '#e4e4e4'
    image: str | None = None           # 'shush' (any extension) or 'shush.gif'
    fallback_image: str | None = None  # used when ``image`` is missing
    sound: str | None = None           # reserved: sound playback not implemented yet


@dataclass
class Meme:
    """Decoded meme: one frame for stills, several for animations."""
    name: str
    frames: list        # PIL RGB images
    delays: list        # seconds per frame

    @property
    def animated(self):
        return len(self.frames) > 1

    def frame_index(self, elapsed):
        if not self.animated:
            return 0
        t = elapsed % sum(self.delays)
        for index, delay in enumerate(self.delays):
            t -= delay
            if t < 0:
                return index
        return len(self.frames) - 1


def load_reactions(path, known_gestures):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise ConfigError(f'Reaction config missing: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f'{path}: invalid JSON ({exc})') from exc
    entries = data.get('reactions') if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ConfigError(f'{path}: expected {{"reactions": [...]}}')

    reactions, seen_ids, seen_gestures = [], set(), set()
    fields = set(Reaction.__dataclass_fields__)
    for number, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or not entry.get('id') or not entry.get('gesture'):
            raise ConfigError(f'{path}: reaction #{number} needs "id" and "gesture"')
        rid, gesture = entry['id'], entry['gesture']
        if rid in seen_ids:
            raise ConfigError(f'{path}: duplicate reaction id {rid!r}')
        if gesture not in known_gestures:
            print(f'[config] Reaction {rid!r}: unknown gesture {gesture!r}, skipped', flush=True)
            continue
        if gesture in seen_gestures:
            print(f'[config] Reaction {rid!r}: gesture {gesture!r} already has a reaction, skipped', flush=True)
            continue
        unknown = set(entry) - fields
        if unknown:
            print(f'[config] Reaction {rid!r}: ignoring unknown keys {sorted(unknown)}', flush=True)
        values = {k: v for k, v in entry.items() if k in fields}
        values.setdefault('title', rid.upper())
        values.setdefault('image', rid)
        reactions.append(Reaction(**values))
        seen_ids.add(rid)
        seen_gestures.add(gesture)
    return reactions


def decode(path):
    """Decode a still or animated image into RGB frames + delays."""
    with Image.open(path) as source:
        frames, delays = [], []
        for frame in ImageSequence.Iterator(source):
            delay = frame.info.get('duration') or source.info.get('duration')
            delays.append(max(MIN_FRAME_DELAY, delay/1000) if delay else DEFAULT_FRAME_DELAY)
            frames.append(ImageOps.exif_transpose(frame.convert('RGBA')).convert('RGB'))
            if len(frames) >= MAX_FRAMES:
                break
    return Meme(path.name, frames, delays)


class MemeLibrary:
    def __init__(self, folder, reactions):
        self.folder = Path(folder)
        self.set_reactions(reactions)
        self.reload()

    def set_reactions(self, reactions):
        """Swap the reaction list; call reload() afterwards to decode images."""
        self.reactions = list(reactions)
        self.by_gesture = {r.gesture: r for r in self.reactions}

    def resolve(self, name):
        if not name:
            return None
        candidate = self.folder / name
        if candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate if candidate.exists() else None
        return next((self.folder / (name + s) for s in IMAGE_SUFFIXES
                     if (self.folder / (name + s)).exists()), None)

    def reload(self):
        """(Re)decode every reaction's image; missing/broken files are reported."""
        self.memes = {}
        for reaction in self.reactions:
            for name in (reaction.image, reaction.fallback_image):
                path = self.resolve(name)
                if path is None:
                    continue
                try:
                    self.memes[reaction.gesture] = decode(path)
                    break
                except (OSError, ValueError) as exc:
                    print(f'[memes] Could not load {path.name}: {exc}', flush=True)
            else:
                print(f'[memes] No image for reaction {reaction.id!r} (looked for {reaction.image!r})', flush=True)

    def reaction_for(self, gesture):
        return self.by_gesture.get(gesture)

    def meme_for(self, gesture):
        return self.memes.get(gesture)
