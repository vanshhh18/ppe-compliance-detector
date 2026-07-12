import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "models/best.pt"

st.set_page_config(
    page_title="PPE Compliance Detector",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e2530;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #2d3748;
    }
    .compliant-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .violation-box {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()


# ============================================================
# COMPLIANCE LOGIC
# ============================================================
def check_region(person_box, item_boxes, region_top_pct, region_bottom_pct):
    px1, py1, px2, py2 = person_box
    h = py2 - py1
    region = (px1, py1 + h * region_top_pct, px2, py1 + h * region_bottom_pct)
    for ix1, iy1, ix2, iy2 in item_boxes:
        icx, icy = (ix1 + ix2) / 2, (iy1 + iy2) / 2
        if region[0] <= icx <= region[2] and region[1] <= icy <= region[3]:
            return True
    return False

def check_overlap(person_box, item_boxes, min_overlap=0.15):
    px1, py1, px2, py2 = person_box
    for ix1, iy1, ix2, iy2 in item_boxes:
        # compute intersection
        ix1_i, iy1_i = max(px1, ix1), max(py1, iy1)
        ix2_i, iy2_i = min(px2, ix2), min(py2, iy2)
        if ix2_i > ix1_i and iy2_i > iy1_i:
            inter_area = (ix2_i - ix1_i) * (iy2_i - iy1_i)
            item_area = (ix2 - ix1) * (iy2 - iy1)
            if item_area > 0 and (inter_area / item_area) > min_overlap:
                return True
    return False

def check_compliance(results):
    boxes = results[0].boxes
    names = results[0].names
    persons, helmets, gloves, vests, boots_list, goggles = [], [], [], [], [], []

    for box, cls in zip(boxes.xyxy.tolist(), boxes.cls.tolist()):
        label = names[int(cls)]
        if label == "Person": persons.append(box)
        elif label == "helmet": helmets.append(box)
        elif label == "gloves": gloves.append(box)
        elif label == "vest": vests.append(box)
        elif label == "boots": boots_list.append(box)
        elif label == "goggles": goggles.append(box)

    report = []
    for i, p in enumerate(persons):
        violations = []
        if not check_region(p, helmets, 0.0, 0.25): violations.append("Helmet")
        if not check_region(p, goggles, 0.0, 0.25): violations.append("Goggles")
        if not check_region(p, vests, 0.15, 0.85): violations.append("Vest")
        if not check_region(p, gloves, 0.0, 0.25): violations.append("Gloves")
        if not check_region(p, boots_list, 0.80, 1.0): violations.append("Boots")
        report.append({"person": i, "violations": violations, "compliant": len(violations) == 0})
    return report


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.25, 0.05)
    st.markdown("---")
    st.markdown("### 📋 About")
    st.write(
        "This tool detects PPE items (helmet, vest, gloves, boots, goggles) "
        "using a fine-tuned YOLOv8m model, then applies rule-based logic to "
        "flag missing safety gear per worker."
    )
    st.markdown("---")
    st.markdown("**Model:** YOLOv8m")
    st.markdown("**mAP@0.5:** 0.80")


# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-header">🦺 PPE Compliance Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a construction site image or video to detect PPE and flag safety violations</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📷  Image Detection", "🎥  Video Detection"])

# ---------------- IMAGE TAB ----------------
with tab1:
    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img")

    if uploaded_img:
        image = Image.open(uploaded_img)
        with st.spinner("Running detection..."):
            results = model(image, conf=confidence)
        report = check_compliance(results)

        # ---- Summary metrics ----
        total = len(report)
        compliant = sum(1 for r in report if r["compliant"])
        non_compliant = total - compliant
        rate = (compliant / total * 100) if total > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><h2>{total}</h2>Workers Detected</div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><h2 style="color:#10b981">{compliant}</h2>Compliant</div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><h2 style="color:#ef4444">{non_compliant}</h2>Violations</div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><h2>{rate:.0f}%</h2>Compliance Rate</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.subheader("Detection Result")
            st.image(results[0].plot(), channels="BGR", width="stretch")

        with col2:
            st.subheader("Compliance Report")
            if not report:
                st.info("No persons detected in this image.")
            for r in report:
                if r["compliant"]:
                    st.markdown(
                        f'<div class="compliant-box">✅ <b>Person {r["person"]}</b> — Fully Compliant</div>',
                        unsafe_allow_html=True
                    )
                else:
                    tags = " ".join([f"`{v}`" for v in r["violations"]])
                    st.markdown(
                        f'<div class="violation-box">❌ <b>Person {r["person"]}</b> — Non-Compliant<br>'
                        f'Missing: {", ".join(r["violations"])}</div>',
                        unsafe_allow_html=True
                    )

# ---------------- VIDEO TAB ----------------
with tab2:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="vid")

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("▶️ Run Detection on Video", type="primary"):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            stframe = st.empty()
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                if frame_count % 5 == 0:
                    results = model(frame, conf=confidence)
                    annotated = results[0].plot()
                    stframe.image(annotated, channels="BGR")

                progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            st.success("✅ Video processing complete.")