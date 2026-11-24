"""Gesture registry/config, expression scores, smoothing, and meme loading."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import json
import tempfile
import unittest

from PIL import Image

from gesture_recognition import (ConfigError, DETECTORS, FrameLandmarks, GestureRecognizer,
                                 load_gesture_config)
from reaction_manager import GestureStabilizer
from meme_engine import MemeLibrary, load_reactions
from test_gestures import face, hand

ROOT = Path(__file__).resolve().parents[1]


def frame(blendshapes=None, hands=()):
    return FrameLandmarks(face(), list(hands), [], 960, 720, blendshapes or {})


def write_json(folder, name, data):
    path = Path(folder) / name
    path.write_text(json.dumps(data))
    return path


class ExpressionTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = GestureRecognizer(load_gesture_config())

    def test_surprise_needs_jaw_and_brows(self):
        self.assertEqual(self.recognizer.classify(frame({'jawOpen': .7, 'browInnerUp': .8})), 'shock')

    def test_open_mouth_without_brows_is_not_shock(self):
        # Talking/yawning: jaw open, brows neutral.
        self.assertNotEqual(self.recognizer.classify(frame({'jawOpen': .7})), 'shock')

    def test_brow_requirement_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(tmp, 'g.json', {'gestures': {'shock': {'params': {'require_raised_brows': False}}}})
            recognizer = GestureRecognizer(load_gesture_config(path))
        self.assertEqual(recognizer.classify(frame({'jawOpen': .7})), 'shock')

    def test_smile_from_blendshapes(self):
        self.assertEqual(self.recognizer.classify(
            frame({'mouthSmileLeft': .8, 'mouthSmileRight': .7, 'jawOpen': .1})), 'smile')

    def test_neutral_face_is_idle(self):
        self.assertEqual(self.recognizer.classify(frame({'jawOpen': .05, 'mouthSmileLeft': .1})), 'idle')


class ConfigTests(unittest.TestCase):
    def test_shipped_config_loads_and_covers_every_detector(self):
        config = load_gesture_config(ROOT / 'config' / 'gestures.json')
        self.assertEqual({s.id for s in config.specs}, set(DETECTORS))
        reactions = load_reactions(ROOT / 'config' / 'reactions.json', set(DETECTORS))
        self.assertEqual({r.gesture for r in reactions}, set(DETECTORS))

    def test_priority_override_resolves_conflicts(self):
        # A closed fist under the chin matches both calm and handshake.
        fist = frame(hands=[hand(.5, .65, False)])
        self.assertEqual(GestureRecognizer(load_gesture_config()).classify(fist), 'calm')
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(tmp, 'g.json', {'gestures': {'handshake': {'priority': 200}}})
            self.assertEqual(GestureRecognizer(load_gesture_config(path)).classify(fist), 'handshake')

    def test_disabled_gesture_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(tmp, 'g.json', {'gestures': {'calm': {'enabled': False}}})
            recognizer = GestureRecognizer(load_gesture_config(path))
        self.assertEqual(recognizer.classify(frame(hands=[hand(.5, .65, False)])), 'handshake')

    def test_bad_values_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            for data in ({'gestures': {'shock': {'params': {'min_jaw_open': 'high'}}}},
                         {'gestures': {'shock': {'cooldown_seconds': -1}}}):
                with self.assertRaises(ConfigError):
                    load_gesture_config(write_json(tmp, 'g.json', data))
            bad = Path(tmp) / 'bad.json'
            bad.write_text('{nope')
            with self.assertRaises(ConfigError):
                load_gesture_config(bad)

    def test_reactions_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_reactions(write_json(tmp, 'r.json', {'reactions': [{'id': 'x'}]}), {'shock'})
            reactions = load_reactions(write_json(tmp, 'r.json', {'reactions': [
                {'id': 'a', 'gesture': 'shock'}, {'id': 'b', 'gesture': 'nope'}]}), {'shock'})
        self.assertEqual([r.id for r in reactions], ['a'])
        self.assertEqual(reactions[0].image, 'a')


class StabilizerTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(tmp, 'g.json', {'gestures': {
                'shock': {'activation_frames': 2, 'cooldown_seconds': 3.0}}})
            self.config = load_gesture_config(path)

    def feed(self, stabilizer, kind, start, frames, step=.1):
        for i in range(frames):
            result = stabilizer.update(kind, start + i*step)
        return result

    def test_default_needs_five_of_seven(self):
        s = GestureStabilizer(self.config)
        self.assertEqual(self.feed(s, 'smile', 1, 4), 'idle')
        self.assertEqual(s.update('smile', 1.5), 'smile')

    def test_per_gesture_activation_frames(self):
        s = GestureStabilizer(self.config)
        self.assertEqual(self.feed(s, 'shock', 1, 2), 'shock')

    def test_single_dropout_does_not_flicker(self):
        s = GestureStabilizer(self.config)
        self.feed(s, 'smile', 1, 7)
        self.assertEqual(s.update('idle', 2), 'smile')

    def test_cooldown_blocks_retrigger(self):
        s = GestureStabilizer(self.config)
        self.feed(s, 'shock', 1, 3)
        self.assertEqual(self.feed(s, 'idle', 2, 7), 'idle')          # shock ended ~2.4s
        self.assertEqual(self.feed(s, 'shock', 3, 7), 'idle')         # still cooling down
        self.assertEqual(self.feed(s, 'shock', 6, 3), 'shock')        # after cooldown


class MemeTests(unittest.TestCase):
    def test_gif_frames_and_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            colors = ['red', 'green', 'blue']
            frames = [Image.new('RGB', (40, 30), c) for c in colors]
            frames[0].save(Path(tmp) / 'shock.gif', save_all=True, append_images=frames[1:],
                           duration=[100, 200, 300], loop=0)
            Image.new('RGB', (40, 30), 'white').save(Path(tmp) / 'shock.png')
            reactions = load_reactions(write_json(tmp, 'r.json', {'reactions': [
                {'id': 'shock', 'gesture': 'shock'}]}), {'shock'})
            meme = MemeLibrary(tmp, reactions).meme_for('shock')
        self.assertEqual(meme.name, 'shock.gif')        # GIF preferred for bare names
        self.assertEqual(len(meme.frames), 3)
        self.assertEqual([round(d, 2) for d in meme.delays], [.1, .2, .3])
        self.assertEqual([meme.frame_index(t) for t in (0, .15, .35, .65)], [0, 1, 2, 0])

    def test_fallback_image_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            Image.new('RGB', (10, 10), 'white').save(Path(tmp) / 'mewing.png')
            reactions = load_reactions(write_json(tmp, 'r.json', {'reactions': [
                {'id': 'smile', 'gesture': 'smile', 'fallback_image': 'mewing'},
                {'id': 'tilt', 'gesture': 'tilt'}]}), {'smile', 'tilt'})
            library = MemeLibrary(tmp, reactions)
        self.assertEqual(library.meme_for('smile').name, 'mewing.png')
        self.assertFalse(library.meme_for('smile').animated)
        self.assertIsNone(library.meme_for('tilt'))


class UITests(unittest.TestCase):
    def test_compose_overlays_meme_inside_camera_stage(self):
        import numpy as np
        from mirror_ui import MirrorUI, STAGE, WIDTH, HEIGHT
        with tempfile.TemporaryDirectory() as tmp:
            Image.new('RGB', (60, 40), (255, 0, 255)).save(Path(tmp) / 'shock.png')
            reactions = load_reactions(write_json(tmp, 'r.json', {'reactions': [
                {'id': 'shock', 'gesture': 'shock'}, {'id': 'smile', 'gesture': 'smile'}]}), {'shock', 'smile'})
            ui = MirrorUI(MemeLibrary(tmp, reactions))
        camera = np.zeros((720, 1280, 3), np.uint8)
        idle = ui.compose(camera, 'idle', now=0)
        self.assertEqual(idle.shape, (HEIGHT, WIDTH, 3))
        ui.compose(camera, 'shock', face_x=.3, now=1.0)           # face left -> card right
        shown = ui.compose(camera, 'shock', face_x=.3, now=1.5)   # after the pop-in
        stage = shown[STAGE[1]:STAGE[3], STAGE[0]:STAGE[2]]
        magenta = (stage[..., 2] > 200) & (stage[..., 0] > 200) & (stage[..., 1] < 60)
        self.assertTrue(magenta.any())
        self.assertGreater(np.nonzero(magenta)[1].mean(), stage.shape[1] / 2)
        self.assertEqual([r.gesture for r in ui.reactions], ['shock', 'smile'])
        ui.compose(camera, 'smile', now=2.0)                        # no image: no crash


if __name__ == '__main__':
    unittest.main()
