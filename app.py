import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flower AI | Intelligent Flower Analysis",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Header */
    .main-header {
        padding: 10px 0 5px 0;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0;
    }

    .main-subtitle {
        font-size: 18px;
        color: #64748b;
        margin-top: 4px;
    }

    /* Section titles */
    .section-title {
        font-size: 24px;
        font-weight: 650;
        color: #1f2937;
        margin-top: 10px;
    }

    /* Info cards */
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        min-height: 120px;
    }

    .card-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .card-value {
        color: #1e293b;
        font-size: 25px;
        font-weight: 650;
    }

    .card-small {
        color: #64748b;
        font-size: 13px;
        margin-top: 5px;
    }

    /* Status */
    .status-success {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 15px;
        border-radius: 8px;
        color: #065f46;
        font-weight: 600;
    }

    .status-pending {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        border-radius: 8px;
        color: #1e40af;
        font-weight: 600;
    }

    /* Recommendation */
    .recommendation {
        background: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        line-height: 1.7;
        color: #334155;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding: 25px 0 10px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD FLOWER MODEL
# ============================================================

@st.cache_resource
def load_flower_model():
    return YOLO("best (1).onnx")


flower_model = load_flower_model()


# ============================================================
# FLOWER CLASSES
# ============================================================

flower_classes = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🌸 Flower AI")

    st.markdown("---")

    st.markdown("### System Modules")

    st.markdown("""
    **✓ Flower Classification**

    Identifies the flower species using an AI classification model.

    **○ Disease Classification**

    Separate disease-specific models can be integrated.

    **○ Disease Segmentation**

    Identifies affected regions for severity estimation.

    **○ Recommendation Engine**

    Provides guidance based on detected condition.
    """)

    st.markdown("---")

    st.markdown("### Supported Flowers")

    st.markdown("""
    🌼 Daisy  
    🌻 Dandelion  
    🌹 Rose  
    🌻 Sunflower  
    🌷 Tulip
    """)

    st.markdown("---")

    st.caption("Flower AI — Intelligent Flower Analysis System")
    st.caption("AI-based prototype")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-header">'
    '<div class="main-title">🌸 FLOWER AI</div>'
    '<div class="main-subtitle">'
    'Intelligent Flower Classification and Plant Health Analysis System'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📥 Image Input</div>',
    unsafe_allow_html=True
)

input_type = st.radio(
    "Select input method",
    ["📁 Upload Image", "📷 Camera"],
    horizontal=True
)

image = None

if input_type == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a flower image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image containing a flower."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

else:

    camera_image = st.camera_input(
        "Capture a flower image"
    )

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")


# ============================================================
# AI ANALYSIS
# ============================================================

if image is not None:

    st.divider()

    # --------------------------------------------------------
    # RUN FLOWER CLASSIFICATION
    # --------------------------------------------------------

    with st.spinner("Analyzing flower image..."):

        flower_result = flower_model.predict(
            source=image,
            verbose=False
        )[0]

    flower_index = flower_result.probs.top1

    flower_confidence = float(
        flower_result.probs.top1conf
    )

    flower_name = flower_classes[flower_index]


    # --------------------------------------------------------
    # CURRENT DISEASE MODULE STATUS
    # --------------------------------------------------------

    disease_name = "Model Integration Pending"

    disease_confidence = "N/A"

    severity = "Segmentation Pending"

    system_status = "Species Identified"


    # --------------------------------------------------------
    # IMAGE + PRIMARY RESULT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🔍 AI Analysis Result</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.15, 1])

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with left:

        st.markdown("### 📷 Analyzed Image")

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with right:

        st.markdown("### 🌸 Flower Identification")

        st.success(
            f"**Detected Species: {flower_name}**"
        )

        st.markdown("### 🎯 Classification Confidence")

        st.progress(
            min(max(flower_confidence, 0.0), 1.0)
        )

        st.markdown(
            f"**{flower_confidence * 100:.2f}%**"
        )

        st.markdown("### 🟢 System Status")

        st.markdown(
            f"""
            <div class="status-success">
            ✓ {system_status}
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # DASHBOARD
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Analysis Dashboard</div>',
        unsafe_allow_html=True
    )

    # First row
    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">🌸 FLOWER SPECIES</div>
                <div class="card-value">{flower_name}</div>
                <div class="card-small">AI classification result</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">🎯 CONFIDENCE</div>
                <div class="card-value">
                    {flower_confidence * 100:.2f}%
                </div>
                <div class="card-small">Classification confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">🟢 SYSTEM STATUS</div>
                <div class="card-value">{system_status}</div>
                <div class="card-small">Primary AI module completed</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # Second row
    c4, c5, c6 = st.columns(3)

    with c4:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">🦠 DISEASE DETECTION</div>
                <div class="card-value">
                    {disease_name}
                </div>
                <div class="card-small">
                    Disease-specific model required
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">📊 DISEASE SEVERITY</div>
                <div class="card-value">
                    {severity}
                </div>
                <div class="card-small">
                    Requires disease-region segmentation
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c6:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">🔬 DISEASE CONFIDENCE</div>
                <div class="card-value">
                    {disease_confidence}
                </div>
                <div class="card-small">
                    No disease prediction is made
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🌱 AI Recommendation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="recommendation">

        <b>Detected Flower:</b> {flower_name}<br><br>

        The Flower AI system has successfully identified the
        flower species with a confidence of
        <b>{flower_confidence * 100:.2f}%</b>.

        Disease identification is intentionally not inferred
        from the flower classification model. A separate,
        flower-specific disease model is required before a
        disease can be reported.

        This design prevents unsupported disease claims and
        ensures that disease predictions are only displayed
        when validated disease-model output is available.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SYSTEM PIPELINE
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">⚙️ AI Processing Pipeline</div>',
        unsafe_allow_html=True
    )

    pipeline = st.columns(5)

    pipeline_steps = [
        ("1", "🌸", "Flower\nDetection", True),
        ("2", "🔍", "Species\nClassification", True),
        ("3", "🦠", "Disease\nDetection", False),
        ("4", "📊", "Severity\nAnalysis", False),
        ("5", "🌱", "Recommendation", False)
    ]

    for col, (number, icon, label, completed) in zip(
        pipeline,
        pipeline_steps
    ):

        with col:

            if completed:

                st.success(
                    f"{icon}\n\n**{label}**\n\n✓ Completed"
                )

            else:

                st.info(
                    f"{icon}\n\n**{label}**\n\n○ Module pending"
                )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "📌 Upload a flower image or use the camera to begin AI analysis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
    🌸 <b>FLOWER AI</b> &nbsp;|&nbsp;
    Intelligent Flower Classification and Plant Health Analysis System
    <br>
    AI-based academic prototype
    </div>
    """,
    unsafe_allow_html=True
)
