# Cheatsheet: Blender

## Start
```bash
uv run python scripts/blender_setup.py data/recordings/<ts>
```
First run builds `data/blender/<recording>/mocap_scene.blend`, later runs just open it.
`--rebuild` = fresh file (old one is kept as a backup).

## What is in the file

| Thing | Meaning |
|---|---|
| Scene **mocap** | where you work |
| Scene **output** | final render: lens distortion + camera video, 1920x1080 |
| `axis_camera` | the real camera (locked, do not move) |
| Empties `athena`, ... | Vicon objects, animated from the recording |
| `floor_shadow_catcher` | catches shadows on the real floor |
| Collection **models** | put your objects here |
| Collection **holdouts** (in models) | invisible stand-ins for real objects |

Frame 1 = first video frame. Units: metres, z up, floor at z = 0.

## Work

| Do | How |
|---|---|
| Look through the camera | Numpad 0 |
| See the video | camera view; Viewport Shading: Material Preview (Z) |
| Play | Space (timeline) |
| Static object | add it (Shift+A), place it on the floor, move it to **models** (M) |
| Object that moves with a Vicon object | see below |

### Attach an object to a Vicon object (e.g. the car)
1. Model it at the world origin in the car's frame: origin = rear axle centre, **x = forward**.
2. Object Properties → Relations → **Parent**: `athena`.
3. It now sits on the car and moves with it. Adjust its Location there (it is relative to the car).

While Vicon loses the object, the Empty is scaled to 0, so the attached object disappears.

### Hide virtual objects behind a real one (holdout)
Put a simple stand-in (e.g. a box the size of the real car, parented to `athena`) into the
**holdouts** collection. It is invisible but hides everything virtual behind it.

## Render

| Want | How |
|---|---|
| Final video (CG over the camera) | top bar: switch scene to **output** → Render → Render Animation (Ctrl+F12) |
| CG only, transparent | scene **output** → Compositor → Switch node "camera video" off |
| Without the lens look (pinhole) | render scene **mocap** |
| From the terminal | `blender -b mocap_scene.blend -S output -a` |

Output: `data/blender/<recording>/render/`. Change format in Output Properties (e.g. FFmpeg video).

## Export for the live view / offline renderer

Save (Ctrl+S), then in the terminal:
```bash
uv run python scripts/blender_export.py data/blender/<recording> --name my_street
uv run python scripts/live_overlay.py --scene config/scenes/my_street.yaml
```
→ `assets/my_street/*.glb` + `config/scenes/my_street.yaml`. Re-run after every change.

| Exported | Not exported |
|---|---|
| mesh objects in **models** (with modifiers applied) | curves / text: convert to mesh first (Object → Convert → Mesh) |
| base colour, image texture, transparency | shader node effects, emission, glass |
| static + parented objects, holdouts | objects hidden in render, your own animations |

## Light and shadows
- Changed the sun or the world light (EEVEE)? Text editor → `calibrate_shadow_catcher.py` → Run Script.
- Cycles: works as is (real shadow catcher on the floor).

## Don't
- move `axis_camera` or the Empties (their animation is the Vicon data)
- delete `floor_shadow_catcher` (no shadows on the floor otherwise)
