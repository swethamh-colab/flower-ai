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
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_flower_model():
    return YOLO("best (1).onnx")


@st.cache_resource
def load_rose_disease_model():
    return YOLO("best (1).pt")


flower_model = load_flower_model()

# Your 5 flower classes
flower_classes = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]

# Rose disease classes
rose_disease_classes = [
    "Black Spot",
    "Downy Mildew",
    "Fresh Leaf"
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
    # RUN FLOWER MODEL
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
    # RUN DISEASE MODEL ONLY FOR ROSE
    # ---------------------------------------------
    disease_name = "Not Available"
    disease_confidence = 0.0
    system_status = "Analysis Pending"
    severity = "Pending Segmentation"
    recommendation = (
        "A disease model for this flower is not available yet."
    )

    if flower_name == "Rose":

        try:
            disease_model = load_rose_disease_model()

            disease_result = disease_model.predict(
                source=image,
                verbose=False
            )[0]

            disease_index = disease_result.probs.top1
            disease_confidence = float(
                disease_result.probs.top1conf
            )

            disease_name = rose_disease_classes[disease_index]

            if disease_name == "Fresh Leaf":

                system_status = "Healthy"
                severity = "0%"
                recommendation = (
                    "The rose leaf appears healthy. "
                    "Continue proper watering, sunlight and regular care."
                )

            elif disease_name == "Black Spot":

                system_status = "Disease Detected"
                severity = "Pending Segmentation"

                recommendation = (
                    "Remove badly affected leaves, improve air circulation, "
                    "avoid overhead watering and consider an appropriate "
                    "fungicide according to local guidance."
                )

            elif disease_name == "Downy Mildew":

                system_status = "Disease Detected"
                severity = "Pending Segmentation"

                recommendation = (
                    "Improve ventilation, reduce leaf moisture, remove "
                    "affected material and consider an appropriate fungicide "
                    "according to local guidance."
                )

        except Exception:
            system_status = "Disease Model Error"
            disease_name = "Unable to Analyze"
            recommendation = (
                "The rose disease model could not be loaded."
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

        if system_status == "Healthy":

            st.success(
                f"**{system_status}**"
            )

        elif system_status == "Disease Detected":

            st.error(
                f"**{system_status}**"
            )

        else:

            st.info(
                f"**{system_status}**"
            )

    st.divider()

    # ---------------------------------------------
    # DASHBOARD CARDS
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
            f"{disease_confidence * 100:.2f}%"
            if disease_name != "Not Available"
            else "N/A"
        )

    st.divider()

    # ---------------------------------------------
    # RECOMMENDATION
    # ---------------------------------------------
    st.subheader("🌱 Recommendation")

    st.info(recommendation)

    # ---------------------------------------------
    # FUTURE ANALYSIS
    # ---------------------------------------------
    if flower_name != "Rose":

        st.divider()

        st.warning(
            f"🦠 Disease detection for **{flower_name}** "
            "will be added after its flower-specific disease "
            "model is trained."
        )

    st.caption(
        "Note: Severity will be calculated using image segmentation "
        "after the segmentation model is added."
    )
