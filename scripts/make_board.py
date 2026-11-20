"""Render the ChArUco board from config/board.yaml to a PNG for printing.

Print at 100 % scale (no "fit to page") and measure the square size afterwards.
"""

import argparse

import cv2

from calib.board import BoardConfig, render_board


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--board", default="config/board.yaml")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--margin-mm", type=float, default=10.0)
    p.add_argument("-o", "--out", default="data/board.png")
    a = p.parse_args()

    cfg = BoardConfig.from_yaml(a.board)
    px_per_m = a.dpi / 0.0254
    img = render_board(cfg.build(), px_per_m, margin_px=round(a.margin_mm / 1000 * px_per_m))
    cv2.imwrite(a.out, img)
    w_mm = cfg.squares_x * cfg.square_length_m * 1000
    h_mm = cfg.squares_y * cfg.square_length_m * 1000
    print(f"wrote {a.out}: {img.shape[1]}x{img.shape[0]} px @ {a.dpi} dpi, board {w_mm:.0f} x {h_mm:.0f} mm")


if __name__ == "__main__":
    main()
