# mocap-overlay

**Augmented-reality overlays for motion-capture labs.** Calibrate a fixed IP camera to a Vicon
motion capture system, then draw what Vicon tracks (poses, trajectories, 3D models, virtual
worlds) into the camera video: live, or rendered offline in our renderer or in Blender.

Built at ZHAW for autonomous 1:10 RC cars, but nothing is car-specific: any Vicon object works.

```
camera (RTSP) ─┐                          ┌─► live view (OpenGL, 25 fps)
               ├─► calibration ─► sync ───┼─► offline video (MP4)
Vicon ─────────┘                          └─► Blender scene (model, preview, render)
```

## Features

- **Camera calibration** with ChArUco boards: intrinsics (pinhole, rational, fisheye) and the
  camera pose in the Vicon world via robot-world hand-eye calibration, with live checks.
- **Time sync**: measures the camera delay from a moving board, so overlays stay on moving objects.
- **Recorder**: camera video + Vicon poses on one clock, replayable as if live.
- **Live overlay**: 3D scenes rendered with the real lens distortion, shadows on the real floor,
  holdouts for real objects.
- **Blender workflow**: opens a recording in Blender with the calibrated camera and the Vicon
  motion as keyframes; render there (lens distortion in the compositor) or export the models
  for the live view.
- No Vicon Python package needed (small ctypes binding to the DataStream SDK).

## Requirements

- Linux, Python ≥ 3.11 with [uv](https://docs.astral.sh/uv/), ffmpeg, a GPU with EGL (NVIDIA tested)
- Fixed IP camera with RTSP (tested: Axis, 1920x1080, 25 fps)
- Vicon system with the DataStream SDK (Tracker 3.10 tested); SDK binaries from Vicon, not included
- A printed ChArUco board with Vicon markers on it
- Optional: Blender ≥ 4.2 (4.5 tested)

## Quick start

```bash
uv sync && uv run pytest
```
Then follow [docs/cheatsheet.md](docs/cheatsheet.md) (calibration → recording → overlays) and
[docs/blender_cheatsheet.md](docs/blender_cheatsheet.md). Open ideas:
[docs/future_improvements.md](docs/future_improvements.md). Design notes and conventions:
[CLAUDE.md](CLAUDE.md).

Status: **v0.1.0**, first working version, used in one lab. Expect rough edges.

---

# Detailed guide

## Vicon DataStream SDK

The SDK binaries are not in git (Vicon licence). From the SDK download
(`ViconDataStreamSDK_<ver>_Linux64/Linux64/`) copy `libViconDataStreamSDK_C.so` and
`libViconDataStreamSDK_CPP.so` to `third_party/vicon_sdk/`. Python talks to them through a small
ctypes binding (`record/vicon_client.py`), no Vicon Python package needed.

Set the Vicon PC (name or IP) with `export VICON_HOST=...`, then check the connection (Tracker running):
`uv run python scripts/check_vicon.py`

## Step 1: intrinsics

1. Set the real board in `config/board.yaml` and lock the camera settings in `config/camera.yaml`.
2. Print the board at 100 % scale, mount it flat and rigid, measure the square size, update the YAML:
   `uv run python scripts/make_board.py` → `data/board.png`
3. Capture 30–50 varied views (tilted, near the image corners and edges, different distances):
   `AXIS_USER=... AXIS_PASS=... uv run python scripts/capture_frames.py`
4. Calibrate (compare lens models once, then set `intrinsics.model`):
   `uv run python scripts/calibrate_intrinsics.py --compare`
   → `data/intrinsics/intrinsics_<date>_<model>.yaml`. Target RMS < 0.5 px.
5. Check live: `uv run python scripts/check_intrinsics.py` (newest calibration, or `--calib <yaml>`).
   Straight lines must be straight on the right; with the board in view the RMS should stay < ~1 px
   everywhere, especially near the image corners. Keys: G grid, [ ] crop/full view, S snapshot.

## Step 2: extrinsics (Vicon world -> camera)

1. Board with Vicon markers as a rigid object in Tracker; name it in `config/vicon.yaml` (`subjects.board`).
2. Capture ~20 static poses over the whole driving area and image, tilted about different axes:
   `uv run python scripts/capture_static_poses.py` (SPACE capture, U undo; poses are only saved if
   the board was still for camera and Vicon) → `data/extrinsics/<session>/`
3. Solve: `uv run python scripts/calibrate_extrinsics.py data/extrinsics/<session>`
   → `data/extrinsics/extrinsics_<date>.yaml` (contains intrinsics too). Target RMS ~0.5–1 px.
4. Validate live: `uv run python scripts/check_extrinsics.py`. Vicon markers (yellow) must sit on the
   real markers and the predicted board (cyan) on the printed board, everywhere in the image.

## Recording (camera + Vicon)

```bash
uv run python scripts/record_session.py --note "first drive"            # Q / Ctrl+C to stop
uv run python scripts/record_session.py --no-preview --duration 60       # headless, e.g. over SSH
```
→ `data/recordings/<timestamp>/`: `video.mkv` (H.264), `frames.csv` (one row per video frame),
`vicon_poses.csv`, optional `markers.csv` (`--markers`), `meta.yaml`, and a copy of the calibration.
Every CSV starts with `t_host` (this machine's clock), so further sources (e.g. ROS 2 topics) can be
added as more CSVs. The camera delay is not removed yet; that is the time sync step.

Load in Python: `Recording("data/recordings/<ts>")` → `.frames`, `.iter_frames()`,
`.pose_at(subject, times)` (interpolated, invalid across occlusions and gaps).

All camera scripts read through a background grabber (`record/camera_stream.py`), so they no
longer lag behind the stream. `--source <video file>` plays a recording back in real time as if it
were the camera, handy for developing without the lab.

## Time sync (camera delay)

Record ~30 s while moving the ChArUco board (tracked by Vicon) at moderate speed in front of the camera:
```bash
uv run python scripts/record_session.py --duration 30 --note "delay board"
uv run python scripts/estimate_time_offset.py            # newest recording with the board tracked
```
→ `data/time_sync/time_sync_<date>.yaml` with `delay_s`: a frame stamped `t` (arrival on this
machine) was exposed at `t - delay_s`. Also writes `time_sync_check.jpg` into the recording
(red: corners predicted without the correction, green: with). Re-measure whenever the network
path, stream profile or the machine changes.

## Overlays

**Live check** (`scripts/check_extrinsics.py`) applies the measured camera delay: each frame is
drawn with the Vicon data from its exposure time. Press D to toggle the correction.

**Live view** of a scene on the camera stream (camera + Vicon, newest calibration and delay):
```bash
uv run python scripts/live_overlay.py --scene config/scenes/demo_street.yaml
uv run python scripts/live_overlay.py --replay data/recordings/<ts>   # recording played back as if live
```
Keys: Q quit, D delay correction, O overlay, H info, F fullscreen, S snapshot. If Vicon drops out,
the video keeps running without overlay. Streaming output (NDI, virtual webcam, ...) is planned,
see [docs/future_improvements.md](docs/future_improvements.md).

**Offline render** of a scene onto a recording:
```bash
uv run python scripts/render_offline.py data/recordings/<ts> --scene config/scenes/demo_street.yaml
```
→ `data/renders/<recording>_<scene>.mp4`. Scenes are YAML (`config/scenes/`): static objects in the
Vicon world (moved together with `origin`), objects attached to Vicon subjects, holdouts and
trails. Rendering is OpenGL offscreen (EGL, no window needed) with a virtual pinhole camera
remapped onto the real lens distortion (`overlay/gl_renderer.py`).

Virtual objects are always drawn over untracked real objects (people, furniture); only Vicon
subjects can hide them (holdouts). Place scenes on free floor.

## Blender

```bash
uv run python scripts/blender_setup.py data/recordings/<ts>
```
Opens Blender with `data/blender/<recording>/mocap_scene.blend`: the calibrated camera with the
camera video (undistorted) as background, one Empty per Vicon subject animated with the
recorded poses (camera delay applied; Blender frame 1 = video frame 0), sun, Vicon origin and a
floor shadow catcher. Model in the `models` collection: static objects in the Vicon world,
moving objects parented to a subject Empty. Later runs open the existing file (your work is
never overwritten; `--rebuild` makes a fresh one and keeps a backup).

Render scene **`output`** for final results: its compositor renders scene `mocap`, applies the
real lens distortion (Map UV) and puts the CG over the original video at 1920x1080; switch
node off = CG only with transparent background. `blender -b mocap_scene.blend -S output -a`.
EEVEE by default; after changing the light, run the text block `calibrate_shadow_catcher.py`.
Details: the `README_mocap` text block inside the .blend.

Export the models for the live view and the offline renderer (save the .blend first):
```bash
uv run python scripts/blender_export.py data/blender/<recording> --name my_street
```
→ `assets/my_street/` (`world.glb` static, `<subject>.glb` per Vicon object, `holdouts_*.glb`) and
`config/scenes/my_street.yaml`. Supported: meshes, base colour, image textures, transparency.

## License

[MIT](LICENSE). The Vicon DataStream SDK is not part of this repository and has its own licence
from Vicon.
