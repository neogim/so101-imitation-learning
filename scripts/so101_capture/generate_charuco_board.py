#!/usr/bin/env python3
"""Rebuild the versioned SO-101 Side ChArUco board as PNG and A4 PDF.

Source of truth is assets/calibration/charuco_7x9_20mm_14mm.svg.
This script uses OpenCV's ChArUco implementation with the same board settings.
The output should be printed at 100% (no fit-to-page).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

SQUARES_X = 7
SQUARES_Y = 9
SQUARE_MM = 20.0
MARKER_MM = 14.0
BOARD_SIZE_MM = (SQUARES_X * SQUARE_MM, SQUARES_Y * SQUARE_MM)
IMAGE_SIZE = (1400, 1800)


def generate_board():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "CharucoBoard"):
        raise RuntimeError("OpenCV ChArUco support is required")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_MM / 1000, MARKER_MM / 1000, dictionary)
    image = board.generateImage(IMAGE_SIZE, marginSize=0, borderBits=1)
    detector = aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(image)
    if ids is None or len(ids) != (SQUARES_X - 1) * (SQUARES_Y - 1):
        raise RuntimeError("Generated ChArUco board failed 48-corner validation")
    return image


def create_pdf(image, dest):
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from PIL import Image

    pdf = canvas.Canvas(str(dest), pagesize=A4, pageCompression=1)
    page_w, page_h = A4
    board_w, board_h = (v * mm for v in BOARD_SIZE_MM)
    x = (page_w - board_w) / 2
    y = (page_h - board_h) / 2 + 9 * mm
    # Pixel dimensions 1400x1800 correspond to a 140x180mm board at 10px/mm.
    pdf.drawImage(ImageReader(Image.fromarray(image)), x, y, width=board_w, height=board_h, mask="auto")
    pdf.setFont("Helvetica", 9)
    pdf.drawString(16 * mm, 23 * mm, "ChArUco 7x9 | DICT_4X4_50 | square 20mm | marker 14mm")
    pdf.drawString(16 * mm, 17 * mm, "Print at 100% / Actual size. Measure the 100 mm line below.")
    pdf.setLineWidth(0.3 * mm)
    pdf.line(16 * mm, 10 * mm, 116 * mm, 10 * mm)
    pdf.setTitle("SO-101 ChArUco board 7x9 - 20mm square 14mm marker")
    pdf.showPage()
    pdf.save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/charuco-board"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "charuco_7x9_20mm_14mm.png"
    pdf = args.out_dir / "charuco_A4_7x9_square20mm_marker14mm.pdf"
    for dest in (png, pdf):
        if dest.exists():
            p.error(f"Refusing to overwrite {dest}")
    image = generate_board()
    if not cv2.imwrite(str(png), image):
        raise RuntimeError(f"Failed to save {png}")
    create_pdf(image, pdf)
    print("Saved", png)
    print("Saved", pdf)
    print("Check 100mm scale line after printing; attach board to a rigid flat surface.")


if __name__ == "__main__":
    main()
