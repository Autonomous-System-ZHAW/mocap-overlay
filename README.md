# mocap-overlay

Calibrate the lab Axis camera to the Vicon system and overlay Vicon data on the video.
See [CLAUDE.md](CLAUDE.md) for the full plan and conventions.

## Setup

```bash
uv sync
uv run pytest
```

## Vicon DataStream SDK

The SDK binaries are not in git (Vicon licence). From the SDK download
(`ViconDataStreamSDK_<ver>_Linux64/Linux64/`) copy `libViconDataStreamSDK_C.so` and
`libViconDataStreamSDK_CPP.so` to `third_party/vicon_sdk/`. Python talks to them through a small
ctypes binding (`record/vicon_client.py`), no Vicon Python package needed.

Check the connection (Tracker running, host in `config/vicon.yaml` reachable):
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
