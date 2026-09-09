import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="Flower AI",
    page_icon="🌸",
    layout="wide"
)

# -------------------------------------------------
# LOAD FLOWER MODEL
# -------------------------------------------------
@st.cache_resource
def load_flower_model():
    return YOLO("best (1).onnx")


flower_model = load_flower_model()

# -------------------------------------------------
# FLOWER CLASSES
# -------------------------------------------------
flower_classes = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🌸 FLOWER AI")
st.subheader("Intelligent Flower Detection & Disease Analysis System")

st.write(
    "Upload a flower image or use your camera to identify the flower "
    "and analyze its condition."
)

st.divider()

# -------------------------------------------------
# INPUT
# -------------------------------------------------
input_type = st.radio(
    "Choose input method",
    ["📁 Upload Image", "📷 Open Camera"],
    horizontal=True
)

image = None

if input_type == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a flower image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

else:

    camera_image = st.camera_input(
        "Take a flower photo"
    )

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")


# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------
if image is not None:

    st.divider()

    # ---------------------------------------------
    # FLOWER DETECTION
    # ---------------------------------------------
    flower_result = flower_model.predict(
        source=image,
        verbose=False
    )[0]

    flower_index = flower_result.probs.top1

    flower_confidence = float(
        flower_result.probs.top1conf
    )

    flower_name = flower_classes[flower_index]

    # ---------------------------------------------
    # DISEASE ANALYSIS
    # ---------------------------------------------
    # IMPORTANT:
    # No disease model has been connected yet.
    # Therefore we DO NOT make a fake disease prediction.

    disease_name = "Not Available"
    disease_confidence = 0.0
    severity = "Pending Segmentation"

    system_status = "Analysis Pending"

    recommendation = (
        "Flower species detected successfully. "
        "A flower-specific disease model is not available yet."
    )

    # ---------------------------------------------
    # IMAGE + RESULT
    # ---------------------------------------------
    left, right = st.columns([1, 1])

    with left:

        st.subheader("📷 Input Image")

        st.image(
            image,
            use_container_width=True
        )

    with right:

        st.subheader("🤖 AI Analysis")

        # Flower
        st.markdown("### 🌸 Flower")

        st.success(
            f"Detected Flower: **{flower_name}**"
        )

        # Confidence
        st.markdown("### 🎯 Confidence")

        st.progress(
            min(max(flower_confidence, 0.0), 1.0)
        )

        st.write(
            f"**{flower_confidence * 100:.2f}%**"
        )

        # System status
        st.markdown("### 🟢 System Status")

        st.info(
            f"**{system_status}**"
        )

    st.divider()

    # ---------------------------------------------
    # DASHBOARD
    # ---------------------------------------------
    st.subheader("📊 Flower AI Dashboard")

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    with c1:

        st.metric(
            "🌸 Flower",
            flower_name
        )

    with c2:

        st.metric(
            "🎯 Confidence",
            f"{flower_confidence * 100:.2f}%"
        )

    with c3:

        st.metric(
            "🟢 System Status",
            system_status
        )

    with c4:

        st.metric(
            "🦠 Disease Detection",
            disease_name
        )

    with c5:

        st.metric(
            "📊 Severity",
            severity
        )

    with c6:

        st.metric(
            "🔬 Disease Confidence",
            "N/A"
        )

    st.divider()

    # ---------------------------------------------
    # RECOMMENDATION
    # ---------------------------------------------
    st.subheader("🌱 Recommendation")

    st.info(recommendation)

    # ---------------------------------------------
    # DISEASE MODEL STATUS
    # ---------------------------------------------
    st.divider()

    st.warning(
        "🦠 Disease detection is currently unavailable. "
        "A separate disease classification model must be trained "
        "and connected for disease identification."
    )

    st.caption(
        "Note: Disease severity will be calculated using "
        "segmentation after a segmentation model is added."
    )
