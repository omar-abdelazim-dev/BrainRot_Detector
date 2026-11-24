# Brainrot Mirror — Desktop for macOS (Apple Silicon)

A **native desktop window** (not a browser page). Uses OpenCV to access the webcam and MediaPipe to track face/hand landmarks. Displays a meme reaction next to your live camera feed.

## Getting started

1. Install **Python 3.11** (Homebrew: `brew install python@3.11`).
2. Open Terminal in this folder.
3. Run `./START.command` (or double-click `START.command` in Finder; macOS may require right-click > Open on first launch).
4. Allow **Terminal** access to Camera if macOS asks. If no prompt appears, go to **System Settings > Privacy & Security > Camera** and permit Terminal.
5. Press **Q** or **Escape** to quit.

First launch installs Python packages and downloads the MediaPipe face model (`models/face_landmarker.task`, ~4 MB), so it requires internet. Subsequent launches reuse the installed virtual environment (but pip may still check packages). Camera frames are processed on-device only.

## Gestures

- Index finger over lips: **SHHH**
- Index finger near chin: **MEWING**
- Drop your jaw **and** raise your eyebrows: **WHAT?!**
- Alternate both open palms up and down at chest height: **6-7**
- Wide smile: **W RIZZ**
- Tilt your head: **SIDE EYE**
- Smirk (one corner of the mouth up): **SON 😭**
- Fist or palm over your mouth: **BABY GIGGLE**
- Fingertips on your eyes or between your brows: **I GUESS BRO**
- Both open hands beside your head (by the ears) with your mouth open / tongue out: **KAI HAVERTZ CELEB** (with your mouth closed and hands further out it's **ABSOLUTE CINEMA**)
- Tap your index finger on your other wrist, like pointing at a watch: **TAMER HOSNY**

Face expressions (surprise, smile) use the MediaPipe Face Landmarker's expression scores (blendshapes such as `jawOpen`, `browInnerUp`, `mouthSmileLeft`). Hand gestures use geometric rules on MediaPipe Hands landmarks. Thresholds may need adjusting for your lighting and camera distance — see *Tuning gestures* below.

## Interface and controls

The live camera fills the window edge to edge. When a gesture is recognised, its meme pops up as a card **inside the camera view**, on the side away from your face, and fades out when the gesture ends. The pills at the top show mode, face/hand tracking, and frame rate. The original four photos are mapped to shush, mewing, shock, and 6–7. W RIZZ uses the "Monkey With Rizz" macaque (`smile.png`) and SIDE EYE uses the animated Side Eye Dog (`tilt.gif`); if either file is removed they fall back to the mewing image.

Camera reading, face/hand tracking, and drawing run on separate threads, so the video stays at the camera's frame rate (30 fps on the built-in M1 camera) even while detection is busy. The built-in camera captures at most 1280×720; the image is shown full-bleed with light sharpening.

- Press **1–9** to preview the first nine memes in `config/reactions.json` order.
- Press **A** to return to automatic gesture detection.
- Press **Space** to freeze/unfreeze the camera view.
- Press **L** to show/hide landmark dots (hidden by default).
- Press **E** to toggle image enhancement (sharpening; on by default; display only, detection uses the raw frame).
- Press **C** to switch between cameras. Startup automatically skips a camera returning black frames.
- Press **R** to reload memes and both config files without restarting (handy while tuning thresholds).
- Press **Q** or **Escape**, or close the window, to quit.

- **Absolute Cinema:** raise two open palms beside your head (preview **7**).
- **Cat handshake:** hold out one closed fist (preview **8**).
- **Calm monkey / furious lion:** rest a closed fist under your chin (preview **9**).
- **6–7:** alternate both open palms up and down at chest height; with two visible hands, move them in opposite directions and swap heights. If only one open hand is visible, bounce it up and down. Low palms near the bottom of the camera view are supported.

## Custom memes

Place images in `memes/`, named `shush`, `mewing`, `shock`, `sixseven`, `smile`, `tilt`, `cinema`, `handshake`, `calm`, `son`, `baby`, `iguessbro`, `kai`, or `tamer`. GIF, PNG, JPG, JPEG, and WebP are supported. **Animated GIFs play** at their own frame timing and loop while the reaction is shown (up to 150 frames). When several formats exist for the same name, GIF wins, then PNG. Images keep their aspect ratio against a blurred backdrop. Sound playback is not implemented yet.

Reactions are defined in `config/reactions.json`:

```json
{"id": "shock", "gesture": "shock", "title": "WHAT?!", "hint": "Jaw drop + brows",
 "accent": "#ffac7f", "image": "shock", "fallback_image": "mewing"}
```

`image` is a name in `memes/` with or without extension (`"shock"` or `"shock.gif"`). Each gesture has at most one reaction; the first nine get number keys for previewing.

## Tuning gestures

`config/gestures.json` holds every gesture's settings:

- `enabled` — turn a gesture off.
- `priority` — when several gestures match the same frame, the highest priority wins (e.g. `calm` beats `handshake` for a fist under the chin).
- `activation_frames` — how many of the last `smoothing.window_frames` (7) frames must agree before the reaction switches.
- `cooldown_seconds` — after a reaction ends, how long before it can trigger again.
- `params` — detection thresholds. Useful ones: `shock.min_jaw_open`, `shock.min_brow_raise`, `shock.require_raised_brows`, `smile.min_smile`, `tilt.min_slope`.

Expression scores run from 0 to 1. Edit the file, press **R** in the app, and try again. Unknown keys are reported in the terminal; invalid values keep the previous config.

To add a gesture: write a function decorated with `@detector('name', priority=...)` in `gesture_recognition.py`, then add a reaction for it in `config/reactions.json`.

## Project layout

- `app.py` — entry point and main loop
- `camera_capture.py` — webcam selection and background capture thread
- `vision.py` — background tracking/recognition thread
- `face_tracking.py` — MediaPipe Face Landmarker (landmarks + expression scores)
- `gesture_recognition.py` — gesture detectors, priority registry, config loading
- `gesture_motion.py` — 6-7 bounce confirmation over time
- `reaction_manager.py` — smoothing, activation frames, cooldowns
- `meme_engine.py` — reaction config and image/GIF decoding
- `mirror_ui.py` — window drawing and meme card animation
- `config/`, `memes/`, `models/`, `tests/`

Run the tests with `.venv/bin/python -m unittest discover -s tests`. Run `python app.py --stats` to print display and detection rates every 5 seconds.

## Troubleshooting

- If camera won't open: quit other camera apps; enable **Terminal** camera permissions. You may need to restart Terminal afterward.
- If install fails: use Python **3.11** and confirm your Mac has internet access. Do not use Python 3.13/3.14 with these pinned older MediaPipe APIs.
- If `./START.command` gives permission denied: run `chmod +x START.command`.
- If the Mac blocks it as untrusted: use Terminal to run `bash START.command`.
- If it says the face model is missing: run `./START.command` again with internet, or download it with the `curl` command printed in the terminal.
- If the app exits with a config error: the message names the file and field; fix it or restore the file from git.
