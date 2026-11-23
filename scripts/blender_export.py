"""Export the models of a Blender mocap scene for the live view and the offline renderer.

    uv run python scripts/blender_export.py data/blender/<recording> --name my_street

Writes assets/<name>/*.glb and config/scenes/<name>.yaml, then use them with:
    uv run python scripts/live_overlay.py --scene config/scenes/<name>.yaml
    uv run python scripts/render_offline.py data/recordings/<ts> --scene config/scenes/<name>.yaml

Save the .blend in Blender first (Ctrl+S): the saved file is exported.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

EXPORT_SCRIPT = Path(__file__).resolve().parent.parent / "blender" / "export_models.py"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("blend", help="data/blender/<recording> folder or a .blend file")
    p.add_argument("--name", required=True, help="scene name: assets/<name>/, config/scenes/<name>.yaml")
    p.add_argument("--blender", default="blender", help="Blender executable")
    p.add_argument("--assets-root", default="assets")
    p.add_argument("--scenes-dir", default="config/scenes")
    a = p.parse_args()

    blend = Path(a.blend)
    if blend.is_dir():
        blend = blend / "mocap_scene.blend"
    if not blend.exists():
        raise SystemExit(f"{blend} not found")
    if shutil.which(a.blender) is None:
        raise SystemExit(f"Blender not found ({a.blender}); pass --blender /path/to/blender")
    assets = Path(a.assets_root) / a.name
    scene = Path(a.scenes_dir) / f"{a.name}.yaml"
    out = subprocess.run([a.blender, "-b", str(blend), "--factory-startup", "--python", str(EXPORT_SCRIPT), "--",
                          str(assets.resolve()), str(scene.resolve()), a.name, assets.resolve().as_posix()
                          if assets.is_absolute() else assets.as_posix()],
                         capture_output=True, text=True)
    for line in (out.stdout + out.stderr).splitlines():
        if (line.startswith(("EXPORT", "SCENE", "Error", "Traceback", "Writing:")) or "rror" in line) \
                and "Not freed memory" not in line:
            print(line)
    if out.returncode != 0 or not scene.exists():
        raise SystemExit(f"export failed (Blender exit code {out.returncode}, see above; full log with -v)")
    print(f"\nlive:    uv run python scripts/live_overlay.py --scene {scene}")
    print(f"offline: uv run python scripts/render_offline.py data/recordings/<ts> --scene {scene}")


if __name__ == "__main__":
    main()
