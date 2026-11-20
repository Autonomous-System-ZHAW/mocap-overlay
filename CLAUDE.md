# mocap-overlay

Calibrate a lab Axis RGB camera to the Vicon motion capture system, then reproject Vicon data (car pose, trajectories, labels) onto the camera video.

First target: the Athena 1:10 RC car (autonomous driving projects).

## Use cases

1. **Analysis / debugging**: check tracking data against video. Rerun.io for visualization.
2. **Media output**: polished overlay videos for the website, conferences, live streaming.
   - Offline: recorded video + Vicon log → rendered MP4 (best quality, sync fixed afterwards)
   - Live: RTSP + Vicon DataStream → overlay → NDI output (into OBS / vMix / Resolume)

## Hardware

- Axis IP camera (RTSP), fixed mount in the lab
- Vicon MoCap system (DataStream SDK)
- ChArUco boards, with Vicon markers glued on as a rigid body (4–5 markers, asymmetric, not collinear)

## Calibration pipeline

1. **Lock camera settings** before calibrating: fixed resolution/stream profile, fixed focus/zoom, EIS/dewarping off, defined IR-cut mode. Changing any of these invalidates the calibration.
2. **Intrinsics**: OpenCV ChArUco (`CharucoDetector`, `calibrateCamera`), 30–50 varied views covering the image corners. Try the fisheye/rational model if the lens is wide. Target RMS < 0.5 px.
3. **Extrinsics (Vicon world → camera)**: robot-world hand-eye calibration.
   - Per static board pose, record `T_cam_board` (solvePnP on ChArUco corners) and `T_world_body` (Vicon rigid body).
   - Solve `T_cam_world · T_world_body · T_body_board = T_cam_board` with `cv::calibrateRobotWorldHandEye`. This gives `T_cam_world` and `T_body_board`, so the marker-to-board offset never has to be measured.
   - Refine with joint least squares on the reprojection error of all ChArUco corners (optionally free the intrinsics).
   - Cross-check: wave a single marker or the Vicon wand around, collect 3D↔2D pairs, run solvePnP.
4. **Validation**: reproject static markers spread across the driving area; report the error in px over the whole image.
5. **Time sync**: camera and Vicon PC on the same NTP server, per-frame timestamps saved. Estimate the remaining offset by minimizing the reprojection error of a fast-moving marker over time shift. The Axis has a rolling shutter, so expect residual error at speed. A single rosbag (vicon_bridge + camera node) is an option.

## Conventions

- Python, modular and readable. Small single-purpose modules, thin CLI scripts.
- Long-running processes (recorders, live pipeline) use an explicit state machine, e.g. `idle → connecting → streaming → error`.
- Config in YAML (`config/`). Calibration results saved as YAML with the date and the camera settings they were made with.
- Frames: Vicon world is the root. Camera uses the OpenCV convention (x right, y down, z forward). Transforms are named `T_<target>_<source>`.
- Car rigid body: origin at the rear axle centre, x forward.
- Prefer self-hosted / local tools over external services.
- `data/` is gitignored (raw recordings, results).

## Repo layout (planned)

```
config/        board.yaml, camera.yaml, overlay styles
calib/         intrinsics.py, extrinsics.py, time_sync.py, io.py
record/        axis_recorder.py, vicon_recorder.py
overlay/       renderer.py (OpenCV/Skia, anti-aliased), styles.yaml
output/        file_writer.py (ffmpeg MP4), ndi_sender.py
pipeline/      offline.py, live.py
viz/           rerun_logger.py (debug only, not for media output)
scripts/       CLI entry points per step
data/          gitignored
```

## Rerun layout

- `world`: Vicon frame
- `world/camera`: `Transform3D` (T_world_cam) + `Pinhole` + `ViewCoordinates.RDF`; images at `world/camera/image`
- `world/car`: `Transform3D` + `Boxes3D`; trajectory as `LineStrips3D`
- Use `rr.set_time(...)` with the synced timestamps

## Order of work

intrinsics → static board extrinsics → validation → time offset → driving recordings → offline renderer → live pipeline
