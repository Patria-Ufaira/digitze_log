#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
digitize_log_from_image.py

Extract a well-log curve (e.g., DT) from a scanned image (JPG/PNG) using a bounding box.
Assumptions:
- The curve is a relatively dark line against a lighter grid/background.
- Depth increases downward (y axis).
- Log axis is horizontal (x axis), possibly reversed (e.g., DT 800 on left, 100 on right).

Usage example:
python digitize_log_from_image.py \
  --image "log.png" \
  --bbox 820 140 1050 780 \
  --depth-min 0 --depth-max 500 \
  --log-min 100 --log-max 800 \
  --reverse-x True \
  --out-csv "/tmp/dt_extracted.csv" \
  --preview "/tmp/preview.png"

Tips:
- Use any image viewer to read pixel coordinates (x,y) for top-left and bottom-right of the DT track.
- For DT with scale text "DT µs/m 800—100" at the top, you likely want reverse-x=True and log-min=100, log-max=800.
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import os

def parse_args():
    p = argparse.ArgumentParser(description="Digitize a log curve from an image using a pixel bounding box.")
    p.add_argument("--image", required=True, help="Path to the JPG/PNG image.")
    p.add_argument("--bbox", nargs=4, type=int, metavar=("X0","Y0","X1","Y1"),
                   required=True, help="Bounding box of the track in pixel coords: top-left (X0,Y0) to bottom-right (X1,Y1).")
    p.add_argument("--depth-min", type=float, required=True, help="Depth at top of bbox (same units as desired, e.g., ft).")
    p.add_argument("--depth-max", type=float, required=True, help="Depth at bottom of bbox.")
    p.add_argument("--log-min", type=float, required=True, help="Log value at LEFT side of bbox.")
    p.add_argument("--log-max", type=float, required=True, help="Log value at RIGHT side of bbox.")
    p.add_argument("--reverse-x", type=lambda s: s.lower() in ["1","true","yes","y"], default=False,
                   help="If True, the curve increases to the LEFT (e.g., DT 800 on left, 100 on right). Default False.")
    p.add_argument("--median-k", type=int, default=3, help="How many darkest x-pixels to sample per row (take median).")
    p.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel size (odd). 0 to disable.")
    p.add_argument("--canny", nargs=2, type=float, default=[50,150], help="Canny thresholds: low high.")
    p.add_argument("--minlen", type=int, default=5, help="Minimum consecutive non-empty rows to keep (denoise gaps).")
    p.add_argument("--out-csv", default="digitized_curve.csv", help="Output CSV path (depth, value).")
    p.add_argument("--preview", default=None, help="Optional output PNG with overlay preview.")
    return p.parse_args()

def map_x_to_value(x, x0, x1, v_left, v_right, reverse_x=False):
    # Linear mapping from pixel x to value
    # If reverse_x True, visual "left is high" but log-min/log-max still adhere to v_left/v_right arguments.
    # We rely solely on v_left/v_right provided by user.
    frac = (x - x0) / max(1, (x1 - x0))
    value = v_left + frac * (v_right - v_left)
    return value

def map_y_to_depth(y, y0, y1, d_top, d_bot):
    frac = (y - y0) / max(1, (y1 - y0))
    depth = d_top + frac * (d_bot - d_top)
    return depth

def extract_curve(img, bbox, dmin, dmax, v_left, v_right, reverse_x=False, median_k=3, blur=3, canny=(50,150), minlen=5):
    x0,y0,x1,y1 = bbox
    roi = img[y0:y1, x0:x1].copy()
    if roi.size == 0:
        raise ValueError("Empty ROI. Check bbox coordinates.")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi.copy()

    if blur and blur % 2 == 1 and blur > 0:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    # Edges for curve; many logs have dark curves -> Canny works reasonably.
    edges = cv2.Canny(gray, canny[0], canny[1])

    h, w = edges.shape
    xs = []
    ys = []
    vals = []

    # For each row, collect darkest/edge pixels; choose a robust location by median of k smallest intensity positions.
    # We'll use edges mask; fallback to intensity minima if no edges found in that row.
    for row in range(h):
        edge_cols = np.where(edges[row] > 0)[0]
        if edge_cols.size == 0:
            # fallback: pick k darkest pixels by gray value
            order = np.argsort(gray[row])[:max(1, median_k)]
            sel = order
        else:
            # Choose k leftmost/rightmost depending on which side curve likely lies? Use center-of-mass of edges
            # To be robust, take median of up to k smallest gray intensities at edge locations.
            edge_intens = gray[row, edge_cols]
            if edge_intens.size == 0:
                sel = edge_cols[:max(1, median_k)]
            else:
                idx = np.argsort(edge_intens)[:max(1, median_k)]
                sel = edge_cols[idx]

        x_est = int(np.median(sel))
        y_pix = row

        # Map to value/depth
        # If user says reverse_x, swap v_left and v_right to match visual
        vL, vR = (v_right, v_left) if reverse_x else (v_left, v_right)
        val = map_x_to_value(x_est, 0, w-1, vL, vR, reverse_x=False)  # we already handled reverse via swap
        dep = map_y_to_depth(y_pix, 0, h-1, dmin, dmax)

        xs.append(x_est + x0)
        ys.append(y_pix + y0)
        vals.append((dep, val))

    # Remove very short runs of NaNs/gaps if any (simple denoise by continuity)
    df = pd.DataFrame(vals, columns=["depth", "value"]).dropna()
    # Optional smoothing by rolling median over a few rows to reduce jaggy edges
    df["value_smooth"] = df["value"].rolling(window=5, center=True, min_periods=1).median()
    return df, roi, edges

def make_preview(img, bbox, df, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0,y0,x1,y1 = bbox
    roi = img[y0:y1, x0:x1]
    h,w = roi.shape[:2]
    # Build pixel-space x from values using provided linear mapping inverse.
    # For preview, we will scale depth back to pixel y and value to pixel x using axis limits from bbox.
    dtop = df["depth"].iloc[0]
    dbot = df["depth"].iloc[-1]

    # Fit linear mapping from depth->pixel y
    y_pix = np.interp(df["depth"], [df["depth"].min(), df["depth"].max()], [0, h-1])
    # For x_pix, we need to re-map value to bbox width. We don't know v_left/v_right used; derive from first/last x if needed.
    # Instead, we include x_pix produced during extraction? Not stored. So approximate by rank inside width.
    # We'll just normalize value to [0,1] using its min/max—only for visualization overlay.
    vmin, vmax = df["value"].min(), df["value"].max()
    if vmax == vmin:
        frac = np.zeros_like(df["value"])
    else:
        frac = (df["value"] - vmin) / (vmax - vmin)
    x_pix = frac * (w-1)

    fig = plt.figure(figsize=(4, 8), dpi=150)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.plot(x_pix, y_pix, linewidth=1.0)
    plt.gca().invert_yaxis()
    plt.axis("off")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def main():
    args = parse_args()
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    bbox = tuple(args.bbox)
    df, roi, edges = extract_curve(
        img, bbox,
        dmin=args.depth_min, dmax=args.depth_max,
        v_left=args.log_min, v_right=args.log_max,
        reverse_x=args.reverse_x,
        median_k=args.median_k,
        blur=args.blur,
        canny=tuple(args.canny),
        minlen=args.minlen
    )

    # Save CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Optional preview
    if args.preview:
        try:
            make_preview(img, bbox, df, args.preview)
        except Exception as e:
            print("Failed to create preview:", e)

    print(f"Saved {len(df)} samples to: {args.out_csv}")
    if args.preview:
        print(f"Preview saved to: {args.preview}")

if __name__ == "__main__":
    main()
