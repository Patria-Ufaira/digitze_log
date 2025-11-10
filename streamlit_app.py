
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
import cv2
import io
from PIL import Image

st.set_page_config(page_title="Log Digitizer (DT, etc.)", layout="wide")

st.title("🛠️ Well Log Digitizer — Image → CSV")
st.caption("Upload a log image → draw a bounding box → set scales → extract curve → download CSV.")

with st.sidebar:
    st.header("1) Upload")
    img_file = st.file_uploader("JPG/PNG of log", type=["jpg","jpeg","png"])
    st.header("2) Preprocess")
    blur = st.slider("Gaussian blur (odd, 0=off)", 0, 15, 3, step=1)
    canny_low, canny_high = st.slider("Canny thresholds", 0, 300, (50,150), step=5)
    median_k = st.slider("Median k (pixels/row)", 1, 15, 3, step=1)
    st.header("3) Scales")
    depth_min = st.number_input("Depth at TOP of box", value=0.0, step=1.0, format="%.3f")
    depth_max = st.number_input("Depth at BOTTOM of box", value=500.0, step=1.0, format="%.3f")
    log_left  = st.number_input("Log value at LEFT side", value=100.0, step=1.0, format="%.3f")
    log_right = st.number_input("Log value at RIGHT side", value=800.0, step=1.0, format="%.3f")
    reverse_x = st.checkbox("Reverse x (left > right visually)?", value=True)
    smooth_win = st.slider("Smoothing window (rows)", 1, 25, 5, step=2)

def to_cv(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def extract_curve(roi_bgr, dmin, dmax, v_left, v_right, reverse_x=False, median_k=3, blur=3, canny=(50,150)):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if blur and blur % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, canny[0], canny[1])

    h, w = edges.shape
    vals = []
    for row in range(h):
        edge_cols = np.where(edges[row] > 0)[0]
        if edge_cols.size == 0:
            order = np.argsort(gray[row])[:max(1, median_k)]
            sel = order
        else:
            edge_intens = gray[row, edge_cols]
            idx = np.argsort(edge_intens)[:max(1, median_k)]
            sel = edge_cols[idx]
        x_est = int(np.median(sel))
        frac_x = x_est / max(1, w-1)
        # handle reverse_x by swapping v_left/v_right
        vL, vR = (v_right, v_left) if reverse_x else (v_left, v_right)
        value = vL + frac_x * (vR - vL)
        frac_y = row / max(1, h-1)
        depth = dmin + frac_y * (dmax - dmin)
        vals.append((depth, value, x_est, row))
    df = pd.DataFrame(vals, columns=["depth","value","x_pix","y_pix"])
    df["value_smooth"] = df["value"].rolling(window=smooth_win, center=True, min_periods=1).median()
    return df, edges

if img_file is None:
    st.info("Upload an image to begin.")
    st.stop()

# Read and display image
image = Image.open(img_file).convert("RGB")
W, H = image.size

st.subheader("Draw bounding box on the DT track")
st.caption("Use the rectangle tool. When done, click outside to finalize.")

# Drawable canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=2,
    stroke_color="#00ff00",
    background_image=image,
    update_streamlit=True,
    height=min(900, H),
    width=min(800, W),
    drawing_mode="rect",
    key="bbox_canvas",
)

bbox = None
if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    # take the last rectangle drawn
    rects = [o for o in objects if o.get("type") == "rect"]
    if rects:
        r = rects[-1]
        x0 = int(r["left"])
        y0 = int(r["top"])
        x1 = int(r["left"] + r["width"])
        y1 = int(r["top"] + r["height"])
        # clip to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 > x0 and y1 > y0:
            bbox = (x0, y0, x1, y1)

col1, col2 = st.columns([1,1], gap="large")

with col1:
    st.markdown("### Selected ROI")
    if bbox:
        st.code(f"bbox = {bbox}", language="text")
        roi = to_cv(image)[bbox[1]:bbox[1]+(bbox[3]-bbox[1]), bbox[0]:bbox[0]+(bbox[2]-bbox[0])]
        st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="ROI (track)")
    else:
        st.warning("Draw a rectangle to define the ROI (bounding box).")

with col2:
    st.markdown("### Extract")
    if st.button("Run extraction", disabled=bbox is None, use_container_width=True):
        if bbox is None:
            st.error("Please draw a bounding box first.")
        else:
            roi = to_cv(image)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            df, edges = extract_curve(
                roi, depth_min, depth_max, log_left, log_right,
                reverse_x=reverse_x, median_k=median_k, blur=blur, canny=(canny_low, canny_high)
            )
            st.success(f"Extracted {len(df)} samples.")
            st.line_chart(df[["value","value_smooth"]])

            # Show overlay preview
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            h, w = roi.shape[:2]
            # Normalize value to [0,1] for overlay only
            vmin, vmax = df["value"].min(), df["value"].max()
            frac = np.zeros_like(df["value"]) if vmax==vmin else (df["value"]-vmin)/(vmax-vmin)
            x_pix = frac*(w-1)
            y_pix = np.interp(df["depth"], [df["depth"].min(), df["depth"].max()], [0, h-1])
            fig = plt.figure(figsize=(3, 8), dpi=150)
            plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            plt.plot(x_pix, y_pix, linewidth=1.0)
            plt.gca().invert_yaxis()
            plt.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            st.image(buf.getvalue(), caption="Overlay preview")

            # Download CSV
            csv = df[["depth","value","value_smooth"]].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, file_name="digitized_curve.csv", mime="text/csv", use_container_width=True)

st.markdown("---")
with st.expander("Notes & Tips"):
    st.write("""
- Set **Depth at TOP/BOTTOM** sesuai label skala di log.
- **Log value at LEFT/RIGHT**: isi berdasarkan angka di header track (mis. DT µs/m 800—100 → left=800, right=100, centang *Reverse x*).
- Tuning **Blur** & **Canny** bila kurva tipis/tebal atau grid kuat.
- Output `value_smooth` adalah median rolling untuk kurangi noise.
""")
