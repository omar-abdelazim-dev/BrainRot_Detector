# Brainrot Mirror — Desktop for macOS (Apple Silicon)

A **native desktop window** (not a browser page). Uses OpenCV to access the webcam and MediaPipe to track face/hand landmarks. Displays a meme reaction next to your live camera feed.

## Getting started

1. Install **Python 3.11** (Homebrew: `brew install python@3.11`).
2. Open Terminal in this folder.
3. Run `./START.command` (or double-click `START.command` in Finder; macOS may require right-click > Open on first launch).
4. Allow **Terminal** access to Camera if macOS asks. If no prompt appears, go to **System Settings > Privacy & Security > Camera** and permit Terminal.
5. Press **Q** or **Escape** to quit.

First launch installs Python packages and requires internet. Subsequent launches reuse the installed virtual environment (but pip may still check packages).

## Gestures

- Index finger over lips: **SHHH**
- Index finger near chin: **MEWING**
- Open mouth wide: **WHAT?!**
- Alternate both open palms up and down at chest height: **6-7**
- Wide smile: **W RIZZ**
- Tilt your head: **SIDE EYE**

Detection uses simple rules and may need adjusting depending on lighting/camera distance. This is a prototype, not a trained gesture classifier.

## Interface and controls

The camera and meme reaction appear side by side. The original four photos are mapped to shush, mewing, shock, and 6–7. The smile and tilt reactions reuse the orange cat unless you provide separate photos. The meme panel stays black while idle and shows photos without extra captions. The camera view is 960×720 inside a 1600×1030 window, with capture detail retained up to 1280 pixels wide.

- Click a gesture tile or press **1–9** to preview its photo.
- Press **A** to return to automatic gesture detection.
- Press **Space** to freeze/unfreeze the camera view.
- Press **L** to show/hide landmark dots (hidden by default).
- Press **C** to switch between cameras. Startup automatically skips a camera returning black frames.
- Press **R** to reload custom photos without restarting.
- Press **Q** or **Escape**, or close the window, to quit.

- **Absolute Cinema:** raise two open palms beside your head (preview **7**).
- **Cat handshake:** hold out one closed fist (preview **8**).
- **Calm monkey / furious lion:** rest a closed fist under your chin (preview **9**).
- **6–7:** alternate both open palms up and down at chest height; with two visible hands, move them in opposite directions and swap heights. If only one open hand is visible, bounce it up and down. Low palms near the bottom of the camera view are supported.

## Custom memes

Place images in `memes/`, named `shush`, `mewing`, `shock`, `sixseven`, `smile`, `tilt`, `cinema`, `handshake`, or `calm`. PNG, JPG, JPEG, WebP, and the first frame of GIF files are supported. PNG takes precedence when multiple formats exist for the same reaction. Photos retain their aspect ratio against a blurred backdrop. Sound and animated GIF playback are not implemented.

## Troubleshooting

- If camera won't open: quit other camera apps; enable **Terminal** camera permissions. You may need to restart Terminal afterward.
- If install fails: use Python **3.11** and confirm your Mac has internet access. Do not use Python 3.13/3.14 with these pinned older MediaPipe APIs.
- If `./START.command` gives permission denied: run `chmod +x START.command`.
- If the Mac blocks it as untrusted: use Terminal to run `bash START.command`.
