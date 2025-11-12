import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Log Digitizer + Table OCR (RapidOCR)", layout="wide")
st.title("🛠️ Well Log Digitizer and Table OCR ")

TAB1, TAB2 = st.tabs(["Digitize Log Curve", "OCR Table → CSV (RapidOCR)"])

# ===================== Utilities =====================
def to_cv(img_pil: Image.Image) -> np.ndarray:
    """PIL -> BGR (OpenCV)."""
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def resize_keep(image: Image.Image, max_w=900, max_h=1200):
    """Resize menjaga rasio, tidak upscale."""
    w0, h0 = image.size
    scale = min(max_w / w0, max_h / h0, 1.0)
    wv, hv = int(w0 * scale), int(h0 * scale)
    try:
        resized = image.resize((wv, hv), Image.Resampling.LANCZOS)
    except Exception:
        resized = image.resize((wv, hv), Image.LANCZOS)
    return resized, (w0, h0, wv, hv, scale)

with TAB1:
    st.header("Digitize a Log Curve from Image")

    with st.sidebar:
        st.subheader("Digitizer Settings")
        blur = st.slider("Gaussian blur (odd, 0=off)", 0, 15, 3, step=1)
        canny_low, canny_high = st.slider("Canny thresholds", 0, 300, (50, 150), step=5)
        median_k = st.slider("Median k (pixels/row)", 1, 15, 3, step=1)
        depth_min = st.number_input("Depth at TOP of box", value=0.0, step=1.0, format="%.3f")
        depth_max = st.number_input("Depth at BOTTOM of box", value=500.0, step=1.0, format="%.3f")
        log_left  = st.number_input("Log value at LEFT side", value=100.0, step=1.0, format="%.3f")
        log_right = st.number_input("Log value at RIGHT side", value=800.0, step=1.0, format="%.3f")
        reverse_x = st.checkbox("Reverse x (left > right visually)?", value=True)
        smooth_win = st.slider("Smoothing window (rows)", 1, 25, 5, step=2)

    img_file = st.file_uploader("Upload log image (JPG/PNG)", type=["jpg","jpeg","png"], key="digitizer_upl")
    if img_file is None:
        st.info("Upload an image to begin (Digitizer tab).")
    else:
        image = Image.open(img_file).convert("RGB")
        resized, (W0, H0, Wv, Hv, SCALE) = resize_keep(image, max_w=900, max_h=1200)

        st.subheader("Draw bounding box on the DT track")
        st.caption("Use the rectangle tool. When done, click outside to finalize.")

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=2,
            stroke_color="#00ff00",
            background_image=resized,     
            update_streamlit=True,
            height=Hv, width=Wv,         
            drawing_mode="rect",
            key=f"bbox_canvas_{Wv}x{Hv}", 
        )

        bbox = None  
        if canvas_result.json_data is not None:
            objs = canvas_result.json_data.get("objects", [])
            rects = [o for o in objs if o.get("type") == "rect"]
            if rects:
                r = rects[-1]
                # koordinat pada kanvas (resized)
                x0v = max(0, int(r["left"]))
                y0v = max(0, int(r["top"]))
                x1v = min(Wv, int(r["left"] + r["width"]))
                y1v = min(Hv, int(r["top"] + r["height"]))
                if x1v > x0v and y1v > y0v:
                    # konversi ke skala asli
                    x0 = int(x0v / SCALE); y0 = int(y0v / SCALE)
                    x1 = int(x1v / SCALE); y1 = int(y1v / SCALE)
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(W0, x1), min(H0, y1)
                    bbox = (x0, y0, x1, y1)

        def extract_curve(roi_bgr, dmin, dmax, v_left, v_right, reverse_x=False,
                          median_k=3, blur=3, canny=(50,150)):
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
                frac_x = x_est / max(1, w - 1)
                vL, vR = (v_right, v_left) if reverse_x else (v_left, v_right)
                value = vL + frac_x * (vR - vL)

                frac_y = row / max(1, h - 1)
                depth = dmin + frac_y * (dmax - dmin)

                vals.append((depth, value, x_est, row))

            df = pd.DataFrame(vals, columns=["depth", "value", "x_pix", "y_pix"])
            df["value_smooth"] = df["value"].rolling(window=smooth_win, center=True, min_periods=1).median()
            return df

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Selected ROI")
            if bbox:
                st.code(f"bbox (orig) = {bbox}", language="text")
                roi = to_cv(image)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="ROI (original scale)")
            else:
                st.warning("Draw a rectangle to define the ROI.")

        with col2:
            st.markdown("### Extract")
            if st.button("Run extraction", disabled=bbox is None, use_container_width=True):
                if bbox is None:
                    st.error("Please draw a bounding box first.")
                else:
                    roi = to_cv(image)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    df = extract_curve(
                        roi, depth_min, depth_max, log_left, log_right,
                        reverse_x=reverse_x, median_k=median_k, blur=blur,
                        canny=(canny_low, canny_high),
                    )
                    st.success(f"Extracted {len(df)} samples.")
                    st.line_chart(df[["value", "value_smooth"]])

                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    h, w = roi.shape[:2]
                    vmin, vmax = df["value"].min(), df["value"].max()
                    frac = np.zeros_like(df["value"]) if vmax == vmin else (df["value"] - vmin) / (vmax - vmin)
                    x_pix = frac * (w - 1)
                    y_pix = np.interp(df["depth"], [df["depth"].min(), df["depth"].max()], [0, h - 1])
                    fig = plt.figure(figsize=(3, 8), dpi=150)
                    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    plt.plot(x_pix, y_pix, linewidth=1.0)
                    plt.gca().invert_yaxis()
                    plt.axis("off")
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                    st.image(buf.getvalue(), caption="Overlay preview")

                    csv = df[["depth", "value", "value_smooth"]].to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download CSV", csv, file_name="digitized_curve.csv",
                                       mime="text/csv", use_container_width=True)

with TAB2:
    st.header("OCR a Table from Image ")
    st.caption("Deteksi grid tabel (OpenCV)progress bar per-cell.")

    tbl_img = st.file_uploader("Upload table image (JPG/PNG)", type=["jpg","jpeg","png"], key="table_upl")

    colA, colB, colC = st.columns(3)
    with colA:
        bin_thresh = st.slider("Binary threshold (adaptive C)", 5, 51, 21, step=2)
        morph_h = st.slider("Morph kernel (horizontal)", 5, 80, 30, step=1)
    with colB:
        morph_v = st.slider("Morph kernel (vertical)", 5, 80, 25, step=1)
        min_cell_area = st.slider("Min cell area (px)", 100, 20000, 800, step=100)
    with colC:
        join_lines = st.checkbox("Gabung multi-line dalam cell", value=True)
        strip_space = st.checkbox("Trim whitespace", value=True)
        rec_only = st.checkbox("Recognition only (skip detector, lebih cepat)", value=True)

    st.info("Engine: RapidOCR + ONNXRuntime (CPU). Tidak butuh Torch/Tesseract/Paddle.")

    if tbl_img is not None:
        pil = Image.open(tbl_img).convert("RGB")
        bgr = to_cv(pil)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1) Ekstraksi grid
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, bin_thresh, 10)

        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_h, 1))
        horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, morph_v))
        vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel, iterations=2)

        table_mask = cv2.add(horiz, vert)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dil = cv2.dilate(table_mask, kernel, iterations=1)
        cnts, _ = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        H, W = gray.shape
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            area = w*h
            if area < min_cell_area: 
                continue
            if w > 0.9*W and h > 0.9*H:
                continue
            cells.append((y, x, w, h))

        cells = sorted(cells, key=lambda t: (t[0], t[1]))

        # 2) Kelompokkan jadi baris
        rows, current, last_y = [], [], None
        y_tol = 10
        for (y,x,w,h) in cells:
            if last_y is None or abs(y - last_y) <= y_tol:
                current.append((y,x,w,h))
                last_y = y if last_y is None else (last_y + y)//2
            else:
                rows.append(sorted(current, key=lambda t: t[1]))
                current = [(y,x,w,h)]
                last_y = y
        if current:
            rows.append(sorted(current, key=lambda t: t[1]))

        overlay = bgr.copy()
        for r in rows:
            for (y,x,w,h) in r:
                cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 1)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption=f"Detected {sum(len(r) for r in rows)} cells in {len(rows)} rows")

        # 3) OCR
        from rapidocr_onnxruntime import RapidOCR
        @st.cache_resource
        def get_rapid():
            return RapidOCR()

        rapid = get_rapid()

        data = []
        total_cells = sum(len(r) for r in rows)
        prog = st.progress(0, text="Running OCR...")
        done = 0

        for r in rows:
            row_text = []
            for (y,x,w,h) in r:
                cell = bgr[y:y+h, x:x+w]
                cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
                result, _ = rapid(cell_rgb, use_det=not rec_only)
                if result:
                    texts = [itm[1] for itm in result if isinstance(itm, (list, tuple)) and len(itm) >= 2]
                    if strip_space:
                        texts = [str(t).strip() if isinstance(t, str) else str(t) for t in texts]
                    txt = " ".join(texts) if join_lines else ("\n".join(texts))
                else:
                    txt = ""
                row_text.append(txt)

                done += 1
                prog.progress(min(done / max(1, total_cells), 1.0), text=f"OCR {done}/{total_cells} cells")

            data.append(row_text)

        prog.empty()

        max_cols = max((len(r) for r in data), default=0)
        norm = [r + [""]*(max_cols - len(r)) for r in data]
        df = pd.DataFrame(norm)

        st.subheader("Preview")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, file_name="table_rapidocr.csv",
                           mime="text/csv", use_container_width=True)
