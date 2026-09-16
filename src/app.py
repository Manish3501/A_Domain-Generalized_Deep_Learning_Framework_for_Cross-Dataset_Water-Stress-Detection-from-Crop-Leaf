import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Water Stress Detection Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# Custom CSS
# -------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a4d2e, #2d7a4f);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #90EE90;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #c8e6c9;
        margin-top: 5px;
    }
    .section-header {
        color: #1a4d2e;
        border-bottom: 3px solid #2d7a4f;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    .prediction-stress {
        background: #ffebee;
        border-left: 6px solid #c62828;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        color: #c62828;
    }
    .prediction-healthy {
        background: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .info-box {
        background: #f1f8e9;
        border: 1px solid #aed581;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Load ONNX model
# -------------------------------------------------------
@st.cache_resource
def load_model():
    paths = [
        "dann_stress_model_single.onnx",
        "Notebooks/dann_stress_model_single.onnx",
        "/Users/manish/Documents/GitHub/leaf-based-water-stress-detection/Notebooks/dann_stress_model_single.onnx"
    ]
    for path in paths:
        if os.path.exists(path):
            return ort.InferenceSession(path)
    st.error("Model file not found. Please check the path.")
    return None

session = load_model()

# -------------------------------------------------------
# Preprocessing
# -------------------------------------------------------
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image):
    image = image.convert("RGB").resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    img_array = img_array.transpose(2, 0, 1)
    return np.expand_dims(img_array, axis=0)

def predict(image):
    input_array = preprocess(image)
    outputs = session.run(None, {"input": input_array})
    logits = outputs[0][0]
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class]) * 100
    non_stress_conf = float(probs[0]) * 100
    stress_conf = float(probs[1]) * 100
    return predicted_class, confidence, non_stress_conf, stress_conf

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
             width=40) if False else None
    st.markdown("## 🌿 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Overview", "🔍 Predict", "📊 Results", "🏗️ Architecture"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    **MSc Capstone Project**
    University of Galway, Ireland

    **Author:** Manish Chaudhari

    **Model:** DANN + MobileNetV2

    **Datasets:** 39,395 leaf images
    """)
    st.markdown("---")
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-View_Code-black?logo=github)](https://github.com/Manish3501/leaf-based-water-stress-detection)
    """)

# -------------------------------------------------------
# PAGE 1 — OVERVIEW
# -------------------------------------------------------
if page == "🏠 Overview":

    st.markdown("# 🌿 Crop Water Stress Detection")
    st.markdown("### Domain-Adversarial Deep Learning across Three Heterogeneous Datasets")
    st.markdown("---")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">97.32%</div>
            <div class="metric-label">Combined Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.9950</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">9,269</div>
            <div class="metric-label">Test Images</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">Domains</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Problem and solution
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🎯 The Problem")
        st.markdown("""
        <div class="info-box">
        Existing deep learning models for water stress detection are trained
        and tested on a <b>single dataset</b>. When applied to images from
        a different crop, sensor, or environment, they fail — a phenomenon
        called <b>domain shift</b>.<br><br>
        No existing model generalises across multiple, independently collected
        leaf image datasets without knowing the source.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### ✅ The Solution")
        st.markdown("""
        <div class="info-box">
        A <b>Domain-Adversarial Neural Network (DANN)</b> with a shared
        MobileNetV2 backbone and Gradient Reversal Layer (GRL).<br><br>
        The GRL forces the backbone to learn features that are
        <b>stress-discriminative but domain-invariant</b> — ignoring
        which dataset an image came from.<br><br>
        Result: <b>one model, one image, one prediction</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📦 Datasets Used")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">20,955</div>
            <div class="metric-label">🍅 Tomato Leaf Images<br>High-res RGB · 4 field-capacity levels</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">400</div>
            <div class="metric-label">🌽 Maize Leaf Images<br>Field RGB · Irrigation categories</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">18,040</div>
            <div class="metric-label">🌽 Maize2 Arrays<br>48×48 NumPy · 3 water-status levels</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------
# PAGE 2 — PREDICT
# -------------------------------------------------------
elif page == "🔍 Predict":

    st.markdown("# 🔍 Predict Water Stress")
    st.markdown("Upload a crop leaf image to get an instant water stress prediction.")
    st.markdown("---")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.markdown("### Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Supports Tomato or Maize leaf images"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col_result:
        st.markdown("### Prediction Result")

        if uploaded_file and session is not None:
            with st.spinner("Analysing leaf image..."):
                predicted_class, confidence, non_stress_conf, stress_conf = predict(image)

            if predicted_class == 1:
                st.markdown(f"""
                <div class="prediction-stress">
                    🔴 Water Stress Detected
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-healthy">
                    🟢 Non-Stressed
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence bar chart
            fig = go.Figure(go.Bar(
                x=[non_stress_conf, stress_conf],
                y=["Non-Stressed", "Stressed"],
                orientation="h",
                marker_color=["#2e7d32", "#c62828"],
                text=[f"{non_stress_conf:.1f}%", f"{stress_conf:.1f}%"],
                textposition="outside"
            ))
            fig.update_layout(
                title="Class Probabilities",
                xaxis_title="Confidence (%)",
                xaxis=dict(range=[0, 110]),
                height=200,
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            **Prediction:** {'Stressed' if predicted_class == 1 else 'Non-Stressed'}
            **Confidence:** {confidence:.2f}%
            """)

        elif not uploaded_file:
            st.info("👆 Upload a leaf image on the left to get a prediction")

# -------------------------------------------------------
# PAGE 3 — RESULTS
# -------------------------------------------------------
elif page == "📊 Results":

    st.markdown("# 📊 Model Results")
    st.markdown("---")

    # Overall comparison table
    st.markdown("### Comparison of All Approaches")

    import pandas as pd

    results_df = pd.DataFrame({
        "Model": ["Ensemble", "Improved Ensemble", "Feature Fusion†", "DANN (Proposed)"],
        "Accuracy (%)": [71.43, 75.40, "~100*", 97.32],
        "Precision": [0.7996, 0.8442, "-", 0.9678],
        "Recall": [0.5740, 0.6325, "-", 0.9723],
        "F1-Score": [0.5449, 0.6331, "-", 0.9700],
        "ROC-AUC": [0.7037, 0.9446, "-", 0.9950]
    })

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )
    st.caption("† Feature Fusion evaluated on only 48 samples — not directly comparable. *Dataset leakage suspected.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Per-domain results
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Per-Domain Accuracy — DANN")

        domains = ["Tomato (D1)", "Maize (D2)", "Maize2 (D3)"]
        accuracies = [99.97, 91.67, 96.01]
        colors = ["#1a4d2e", "#c62828", "#1565c0"]

        fig = go.Figure(go.Bar(
            x=domains,
            y=accuracies,
            marker_color=colors,
            text=[f"{a}%" for a in accuracies],
            textposition="outside"
        ))
        fig.add_hline(
            y=97.32,
            line_dash="dash",
            line_color="orange",
            annotation_text="Combined 97.32%"
        )
        fig.update_layout(
            yaxis=dict(range=[85, 102], title="Accuracy (%)"),
            height=350,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### ROC-AUC Comparison")

        models = ["Ensemble", "Improved\nEnsemble", "DANN\n(Proposed)"]
        aucs = [0.7037, 0.9446, 0.9950]
        colors_auc = ["#ef9a9a", "#ffcc80", "#1a4d2e"]

        fig2 = go.Figure(go.Bar(
            x=models,
            y=aucs,
            marker_color=colors_auc,
            text=[f"{a}" for a in aucs],
            textposition="outside"
        ))
        fig2.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Perfect (1.0)"
        )
        fig2.update_layout(
            yaxis=dict(range=[0.5, 1.05], title="ROC-AUC"),
            height=350,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Domain classifier accuracy over training
    st.markdown("### Domain Classifier Accuracy During Training (GRL Evidence)")

    epochs = list(range(1, 26))
    domain_acc = [
        39.40, 98.20, 89.99, 57.73, 53.77,
        57.88, 56.16, 57.99, 59.57, 60.15,
        59.81, 58.99, 59.61, 59.60, 59.50,
        59.45, 59.40, 59.55, 59.70, 59.65,
        59.60, 59.50, 59.45, 59.75, 59.58
    ]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=epochs, y=domain_acc,
        mode="lines+markers",
        name="Domain Classifier Accuracy",
        line=dict(color="#c62828", width=2),
        marker=dict(size=5)
    ))
    fig3.add_hline(
        y=33.33,
        line_dash="dash",
        line_color="gray",
        annotation_text="Random chance (33.3%)"
    )
    fig3.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Domain Accuracy (%)",
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(x=0.7, y=0.95),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "The GRL forces the domain classifier toward confusion. "
        "Final domain accuracy: 59.77% (down from ~98% at epoch 2), "
        "confirming the backbone learned to suppress domain-specific cues."
    )

# -------------------------------------------------------
# PAGE 4 — ARCHITECTURE
# -------------------------------------------------------
elif page == "🏗️ Architecture":

    st.markdown("# 🏗️ Model Architecture")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### DANN Framework")
        st.markdown("""
        <div class="info-box">
        <b>1. Shared Backbone — MobileNetV2</b><br>
        Pretrained on ImageNet · 18 conv layers<br>
        Global Average Pooling → 1280-dim feature vector<br><br>

        <b>2. Stress Classifier Head</b> (kept at inference)<br>
        Linear(1280→512) → ReLU → Dropout(0.5)<br>
        Linear(512→128) → ReLU → Dropout(0.3)<br>
        Linear(128→2) → Non-stress / Stress<br><br>

        <b>3. Domain Classifier + GRL</b> (discarded at inference)<br>
        GRL(λ) → Linear(1280→256) → ReLU → Dropout(0.4)<br>
        Linear(256→3) → Tomato / Maize / Maize2
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Training Objective")
        st.latex(r"\mathcal{L}_{total} = \mathcal{L}_{stress} + \lambda \cdot \mathcal{L}_{domain}")
        st.markdown("""
        **λ annealing schedule:**
        """)
        st.latex(r"\lambda_p = \frac{2}{1 + e^{-10p}} - 1, \quad p = \frac{\text{epoch}}{N}")

    with col2:
        st.markdown("### Training Configuration")

        config_df = {
            "Setting": [
                "Backbone", "Optimiser", "Learning Rate",
                "Weight Decay", "Batch Size", "Epochs",
                "Scheduler", "Input Size", "Normalisation"
            ],
            "Value": [
                "MobileNetV2 (ImageNet)", "Adam", "1e-4",
                "1e-4", "32", "25",
                "ReduceLROnPlateau (p=3, f=0.5)",
                "224 × 224", "ImageNet mean/std"
            ]
        }
        import pandas as pd
        st.dataframe(
            pd.DataFrame(config_df),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Why DANN Works")
        st.markdown("""
        <div class="info-box">
        The <b>Gradient Reversal Layer</b> is the key mechanism.
        During backpropagation it multiplies the domain gradient
        by <b>−λ</b>, which pushes the backbone in the direction
        that makes domain classification <i>harder</i>.<br><br>
        Result: the backbone learns features that are useful for
        detecting stress but useless for identifying which dataset
        the image came from — <b>domain-invariant representations</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Model Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", "3,274,373")
    with col2:
        st.metric("Backbone Parameters", "~2.2M")
    with col3:
        st.metric("Classifier Parameters", "~1M")

