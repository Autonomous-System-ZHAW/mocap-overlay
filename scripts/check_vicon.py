"""Check the Vicon DataStream connection.

Lists subjects, frame rate and latency, prints the board pose for a few seconds and
cross-checks the rotation matrix against the quaternion (verifies the SDK binding).
"""

import argparse
import time

import numpy as np

from calib.io import load_yaml
from record.vicon_client import ViconClient, ViconError, quat_to_matrix


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vicon", default="config/vicon.yaml")
    p.add_argument("--subject", help="subject to print (default: subjects.board from the config)")
    p.add_argument("--seconds", type=float, default=3.0)
    a = p.parse_args()

    cfg = load_yaml(a.vicon)
    subject = a.subject or cfg["subjects"]["board"]

    with ViconClient(cfg["sdk_dir"]) as client:
        print(f"connecting to {cfg['host']} ...")
        try:
            client.connect(cfg["host"], cfg.get("stream_mode", "ServerPush"))
        except ViconError as e:
            raise SystemExit(f"{e}. Is Tracker running and the host reachable (VPN / lab network)?")
        client.get_frame()
        subjects = client.subjects()
        print(f"frame rate {client.frame_rate():.1f} Hz, latency {client.latency_s() * 1000:.1f} ms")
        print("subjects: " + ", ".join(f"{s} (root {client.root_segment(s)})" for s in subjects))
        if subject not in subjects:
            raise SystemExit(f"subject {subject!r} not in stream; fix subjects in {a.vicon}")

        markers = client.markers(subject)
        print(f"{subject} markers: {', '.join(markers)}")

        end, n, frames0 = time.monotonic() + a.seconds, 0, client.frame_number()
        while time.monotonic() < end:
            client.get_frame()
            n += 1
            if n % 50:
                continue
            pose = client.segment_pose(subject)
            R_q = quat_to_matrix(client.segment_quaternion(subject))
            diff = np.abs(pose.T_world_segment[:3, :3] - R_q).max()
            t = pose.T_world_segment[:3, 3]
            status = "OCCLUDED" if pose.occluded else f"t = [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m"
            print(f"frame {client.frame_number()}  {subject}: {status}  |R - R(quat)| = {diff:.1e}")
            if not pose.occluded and diff > 1e-6:
                print("  WARNING: matrix and quaternion disagree -> the binding's matrix layout is wrong")
        dropped = client.frame_number() - frames0 - n
        print(f"received {n} frames, {max(dropped, 0)} skipped")


if __name__ == "__main__":
    main()
