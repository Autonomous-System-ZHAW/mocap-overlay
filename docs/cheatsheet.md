# Cheatsheet: mocap-overlay

All commands: `uv run python scripts/<name>.py`, run from the repo root. `--help` on every script.
Environment: `export AXIS_USER=... AXIS_PASS=... VICON_HOST=<vicon pc>` (camera login, Vicon PC name or IP)

## One-time setup (per camera position)

| Step | Command | Result |
|---|---|---|
| Print board | `make_board.py` | `data/board.png` |
| Intrinsics frames | `capture_frames.py` (SPACE = save) | `data/intrinsics/frames/` |
| Intrinsics | `calibrate_intrinsics.py --compare` | `data/intrinsics/*.yaml` |
| Check lens | `check_intrinsics.py` | straight lines? RMS < 1 px? |
| Board poses | `capture_static_poses.py` (SPACE, U = undo) | `data/extrinsics/<session>/` |
| Extrinsics | `calibrate_extrinsics.py` | `data/extrinsics/extrinsics_<date>.yaml` |
| Camera delay | `record_session.py --duration 30` (wave the board), then `estimate_time_offset.py` | `data/time_sync/*.yaml` |
| Check all | `check_extrinsics.py` (D = delay on/off) | markers sit on the real markers? |

Redo everything after changing zoom, focus, resolution or the camera mount.
Redo only the delay when the machine or network changes.

## Daily use

| What | Command |
|---|---|
| Vicon connection test | `check_vicon.py` |
| Record (video + Vicon) | `record_session.py --note "..."` (Q = stop) |
| Live overlay | `live_overlay.py --scene config/scenes/demo_street.yaml` |
| Live overlay on a recording | `live_overlay.py --replay data/recordings/<ts>` |
| Render overlay video | `render_offline.py data/recordings/<ts> --scene ...` → `data/renders/` |
| Open in Blender | `blender_setup.py data/recordings/<ts>` (see [blender_cheatsheet.md](blender_cheatsheet.md)) |
| Blender models → scene | `blender_export.py data/blender/<rec> --name my_street` → `config/scenes/my_street.yaml` |

Live keys: Q quit, D delay, O overlay, H info, F fullscreen, S snapshot.

## Scenes (`config/scenes/*.yaml`)

```yaml
origin: {xyz: [-0.35, -1.75, 0]}            # moves all static objects
objects:
  - name: cone
    mesh: {type: traffic_cone}               # box, plane, cylinder, cone, traffic_cone, gltf (file: ...)
    pose: {xyz: [0, 0.4, 0], rpy_deg: [0, 0, 0]}
    repeat: {count: 4, step: [0.8, 0, 0]}
    color: "#ff7a00"
    shadow: true
  - name: car_body
    attach: athena                           # follows a Vicon object
    mesh: {type: box, size: [0.48, 0.24, 0.14]}
    opacity: 0.5
trails:
  - {subject: athena, seconds: 4, color: "#00e5ff"}
```
Units: metres, degrees, Vicon world (z up, floor z = 0). Put objects on free floor:
virtual objects are always drawn in front of people and furniture.
