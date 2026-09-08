import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

st.set_page_config(
    page_title="Flower AI",
    page_icon="🌸",
    layout="wide"
)

# Load trained model
model = YOLO("best (1).onnx")
# Flower classes
classes = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]

# Header
st.title("🌸 FLOWER AI")
st.subheader("Intelligent Flower Species Detection System")

st.write(
    "Upload a flower image or use your camera to identify the flower species."
)

st.divider()

# Input selection
input_type = st.radio(
    "Choose input method:",
    ["📁 Upload Image", "📷 Open Camera"],
    horizontal=True
)

image = None

# Upload image
if input_type == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a flower image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

# Camera
else:

    camera_image = st.camera_input(
        "Take a picture of the flower"
    )

    if camera_image:
        image = Image.open(camera_image).convert("RGB")


# Prediction
if image:

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Input Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("🤖 AI Analysis")

        # Convert image
        image_array = np.array(image)

        # Prediction
        result = model.predict(
            image_array,
            verbose=False
        )[0]

        # Get prediction
        predicted_index = int(result.probs.top1)

        confidence = float(
            result.probs.top1conf
        )

        flower_name = classes[predicted_index]

        st.success(
            f"🌼 Flower Species: {flower_name}"
        )

        st.metric(
            "AI Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.progress(confidence)

        st.write(
            "The AI model analyzed the uploaded image "
            "and predicted the most likely flower species."
        )


st.divider()

# Future modules
st.subheader("🌱 Flower Health Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("🔬 Disease Detection\n\nComing next")

with col2:
    st.info("🩺 Disease Segmentation\n\nComing next")

with col3:
    st.info("📊 Severity Analysis\n\nComing next")

with col4:
    st.info("🌱 Recommendation\n\nComing next")

st.divider()

st.caption(
    "🌸 Flower AI | YOLO-based Flower Classification"
)
