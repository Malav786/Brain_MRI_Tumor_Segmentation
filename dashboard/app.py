import streamlit as st
import torch
import numpy as np
import pandas as pd
import cv2
import time
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="AI Campus | MRI Segmentation", page_icon="🧠", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dashboard_data")
MODEL_DIR = os.path.join(BASE_DIR, "dashboard_data", "models")
GALLERY_DIR = os.path.join(DATA_DIR, "gallery")

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    *, html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0a0e17; }
    .metric-card {
        background: linear-gradient(135deg, #131924 0%, #1a2332 100%);
        padding: 22px; border-radius: 14px; text-align: center;
        border: 1px solid rgba(0,255,127,0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,255,127,0.1); }
    .metric-card h4 { color: #8892a4; font-size: 0.85rem; margin-bottom: 6px; font-weight: 400; }
    .metric-card h2 { color: #00FF7F; font-size: 1.8rem; margin: 0; font-weight: 700; }
    .metric-card .sub { color: #5a6577; font-size: 0.75rem; margin-top: 4px; }
    h1 { color: #ffffff !important; font-weight: 700 !important; }
    h2, h3 { color: #e0e6ed !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #131924; border-radius: 8px 8px 0 0;
        color: #8892a4; font-weight: 500; padding: 10px 20px;
        border: 1px solid #1e2a3a; border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a2332 !important; color: #00FF7F !important;
        border-color: #00FF7F !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00FF7F, #00cc66); color: #000;
        font-weight: 600; border-radius: 8px; border: none; padding: 0.5rem 1.5rem;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #00cc66, #009950); color: #fff; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #131924 100%);
        border-right: 1px solid #1e2a3a;
    }
    .hero-badge {
        display: inline-block; background: rgba(0,255,127,0.1); color: #00FF7F;
        padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
        border: 1px solid rgba(0,255,127,0.3); margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data Loaders (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource
def get_model_and_device():
    from inference import MetadataUNet, AttentionMetadataUNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(MODEL_DIR, "best_AttentionMetadataUNet_scheduler.pth")
    if os.path.exists(model_path):
        model = AttentionMetadataUNet(meta_features=20)
        model_class_name = 'AttentionMetadataUNet'
    else:
        model_path = os.path.join(MODEL_DIR, "best_MetadataUNet.pth")
        model = MetadataUNet(meta_features=20)
        model_class_name = 'MetadataUNet'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device, model_class_name

# Load data
feature_mapping = load_json(os.path.join(DATA_DIR, "feature_mapping.json"))
training_histories = load_json(os.path.join(DATA_DIR, "training_histories.json"))
dataset_stats = load_json(os.path.join(DATA_DIR, "dataset_stats.json"))
model_comparison = load_csv(os.path.join(DATA_DIR, "final_model_comparison.csv"))
per_sample = load_csv(os.path.join(DATA_DIR, "per_sample_scores.csv"))
tumor_coverage = load_csv(os.path.join(DATA_DIR, "tumor_coverage.csv"))

# Plotly theme
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(19,25,36,0.8)',
    font=dict(family='Inter', color='#8892a4'),
    xaxis=dict(gridcolor='#1e2a3a', zerolinecolor='#1e2a3a'),
    yaxis=dict(gridcolor='#1e2a3a', zerolinecolor='#1e2a3a'),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c0c8d4'))
)
COLORS = ['#00FF7F', '#00BFFF', '#FF6B6B', '#FFD93D', '#C084FC', '#F97316']


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.markdown('<div class="hero-badge">🧠 AI CAMPUS</div>', unsafe_allow_html=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["🏠 Overview", "🔬 Live Inference", "📊 Training Analytics", "🏆 Model Comparison", "🖼️ Prediction Gallery"], label_visibility="collapsed")


# ═══════════════════════════════════════════════
# PAGE 1: Overview
# ═══════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="hero-badge">DEEP LEARNING PIPELINE</div>', unsafe_allow_html=True)
    st.title("🧠 Virtual Radiologist Assistant")
    st.markdown("#### AI-Powered Lower-Grade Glioma (LGG) Segmentation")
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    best = model_comparison.loc[model_comparison['Dice'].idxmax()]
    for col, label, value, sub in [
        (c1, "Best Dice Score", f"{best['Dice']:.4f}", best['Model']),
        (c2, "Best IoU Score", f"{best['IoU']:.4f}", best['Model']),
        (c3, "Total MRI Slices", f"{dataset_stats['total_samples']:,}", f"{dataset_stats['num_patients']} patients"),
        (c4, "Tumor Slices (Val)", f"{dataset_stats['val_tumor_slices']}", f"of {dataset_stats['val_samples']}"),
        (c5, "Avg Tumor Coverage", f"{dataset_stats['avg_tumor_coverage_pct']}%", "tumor-positive slices"),
    ]:
        col.markdown(f'<div class="metric-card"><h4>{label}</h4><h2>{value}</h2><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🏗️ Pipeline Architecture")
        st.markdown("""
        | Phase | Model | Key Innovation |
        |-------|-------|---------------|
        | 1 | BasicUNet | Standard encoder-decoder |
        | 2 | MetadataUNet | Clinical metadata fusion at bottleneck |
        | 3 | AttentionMetadataUNet | + Attention gates on skip connections |
        | 4 | + LR Scheduler | ReduceLROnPlateau for convergence |
        """)

    with col_r:
        st.markdown("### ⚙️ Training Configuration")
        cfg_data = {
            "Parameter": ["Epochs", "Batch Size", "Learning Rate", "Optimizer", "Loss Function", "Image Size", "Mixed Precision"],
            "Value": [dataset_stats['epochs'], dataset_stats['batch_size'], dataset_stats['learning_rate'],
                      dataset_stats['optimizer'], dataset_stats['loss_function'],
                      f"{dataset_stats['image_size'][0]}×{dataset_stats['image_size'][1]}", "✅ AMP"]
        }
        st.dataframe(pd.DataFrame(cfg_data), hide_index=True, use_container_width=True)

    # Quick bar chart
    st.markdown("---")
    st.markdown("### 🏆 Final Model Scores")
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Dice', x=model_comparison['Model'], y=model_comparison['Dice'],
                         marker_color=COLORS[0], text=model_comparison['Dice'].round(4), textposition='outside'))
    fig.add_trace(go.Bar(name='IoU', x=model_comparison['Model'], y=model_comparison['IoU'],
                         marker_color=COLORS[1], text=model_comparison['IoU'].round(4), textposition='outside'))
    fig.update_layout(**PLOTLY_LAYOUT, barmode='group', yaxis_range=[0.7, 0.95],
                      title=None, height=400, yaxis_title='Score')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 2: Live Inference
# ═══════════════════════════════════════════════
elif page == "🔬 Live Inference":
    st.markdown('<div class="hero-badge">REAL-TIME SEGMENTATION</div>', unsafe_allow_html=True)
    st.title("🔬 Live MRI Inference")
    st.markdown("Upload a brain MRI slice and input patient metadata for real-time tumor segmentation.")

    try:
        model, device, model_class = get_model_and_device()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    from inference import preprocess_image, build_metadata_vector

    # Sidebar metadata inputs
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Patient Metadata")
    age = st.sidebar.slider("Age at Diagnosis", 18, 90, 45)
    gender = st.sidebar.selectbox("Gender", ["MALE", "FEMALE"])
    histology = st.sidebar.selectbox("Histological Type", ["astrocytoma", "oligoastrocytoma", "oligodendroglioma"])
    loc_options = list(feature_mapping['categorical_labels']['tumor_location'].keys())
    loc_labels = {"other_1": "Other (1)", "frontal lobe": "Frontal Lobe", "other_4": "Other (4)", "parietal lobe": "Parietal Lobe", "temporal lobe": "Temporal Lobe"}
    location = st.sidebar.selectbox("Tumor Location", loc_options, format_func=lambda x: loc_labels.get(x, x))
    rna_cluster = st.sidebar.selectbox("RNASeq Cluster", ["1", "2", "3", "4"])
    methyl_cluster = st.sidebar.selectbox("Methylation Cluster", ["1", "2", "3", "4", "5"])
    threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

    uploaded_file = st.file_uploader("Upload MRI Slice (.tif, .png, .jpg)", type=["tif", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        img_tensor, original_size = preprocess_image(file_bytes)
        meta_vec = build_metadata_vector(age, gender, histology, location, int(rna_cluster), int(methyl_cluster), feature_mapping)
        meta_tensor = torch.tensor(meta_vec).unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            outputs = model(img_tensor, meta_tensor)
            probs = torch.sigmoid(outputs)
            probs_np = probs.cpu().numpy()[0, 0]
        inference_time = time.time() - start_time

        mask = (probs_np > threshold).astype(np.uint8)
        mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        probs_resized = cv2.resize(probs_np, original_size, interpolation=cv2.INTER_LINEAR)

        img_array = np.frombuffer(file_bytes, np.uint8)
        orig_img = cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        overlay = orig_img.copy()
        overlay[mask_resized == 1] = [0, 255, 127]
        blended = cv2.addWeighted(overlay, 0.4, orig_img, 0.6, 0)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.subheader("Original MRI")
        col1.image(orig_img, use_container_width=True)
        col2.subheader("AI Prediction Mask")
        col2.image(mask_resized * 255, use_container_width=True, clamp=True)
        col3.subheader("Tumor Overlay")
        col3.image(blended, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📈 Clinical Insights")
        tumor_pixels = np.sum(mask_resized)
        total_pixels = mask_resized.size
        tumor_pct = (tumor_pixels / total_pixels) * 100
        avg_conf = float(np.mean(probs_resized[mask_resized == 1])) if tumor_pixels > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        for col, lbl, val in [
            (m1, "Inference Time", f"{inference_time:.3f} s"),
            (m2, "Tumor Coverage", f"{tumor_pct:.2f}%"),
            (m3, "Avg Confidence", f"{avg_conf*100:.1f}%"),
            (m4, "Model Used", model_class),
        ]:
            col.markdown(f'<div class="metric-card"><h4>{lbl}</h4><h2 style="font-size:1.4rem;">{val}</h2></div>', unsafe_allow_html=True)

        # Confidence heatmap
        st.markdown("---")
        st.markdown("### 🌡️ Confidence Heatmap")
        fig_heat = px.imshow(probs_resized, color_continuous_scale='Inferno', aspect='equal',
                             labels=dict(color="Confidence"))
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("👆 Upload an MRI image to run the segmentation pipeline.")


# ═══════════════════════════════════════════════
# PAGE 3: Training Analytics
# ═══════════════════════════════════════════════
elif page == "📊 Training Analytics":
    st.markdown('<div class="hero-badge">TRAINING INSIGHTS</div>', unsafe_allow_html=True)
    st.title("📊 Training Analytics")

    selected_models = st.multiselect("Select models to compare",
        list(training_histories.keys()), default=list(training_histories.keys()))

    if not selected_models:
        st.warning("Select at least one model.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["📉 Loss Curves", "🎯 Dice & IoU", "📋 Epoch Details"])

    with tab1:
        fig = go.Figure()
        for i, name in enumerate(selected_models):
            h = training_histories[name]
            fig.add_trace(go.Scatter(x=h['epochs'], y=h['train_loss'], name=name,
                line=dict(color=COLORS[i % len(COLORS)], width=2.5),
                mode='lines+markers', marker=dict(size=5)))
        fig.update_layout(**PLOTLY_LAYOUT, title='Training Loss per Epoch', height=450,
                          xaxis_title='Epoch', yaxis_title='DiceBCE Loss')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_d, col_i = st.columns(2)
        with col_d:
            fig = go.Figure()
            for i, name in enumerate(selected_models):
                h = training_histories[name]
                fig.add_trace(go.Scatter(x=h['epochs'], y=h['val_dice'], name=name,
                    line=dict(color=COLORS[i % len(COLORS)], width=2.5), mode='lines+markers', marker=dict(size=5)))
            fig.update_layout(**PLOTLY_LAYOUT, title='Validation Dice Score', height=400,
                              xaxis_title='Epoch', yaxis_title='Dice')
            st.plotly_chart(fig, use_container_width=True)
        with col_i:
            fig = go.Figure()
            for i, name in enumerate(selected_models):
                h = training_histories[name]
                fig.add_trace(go.Scatter(x=h['epochs'], y=h['val_iou'], name=name,
                    line=dict(color=COLORS[i % len(COLORS)], width=2.5), mode='lines+markers', marker=dict(size=5)))
            fig.update_layout(**PLOTLY_LAYOUT, title='Validation IoU Score', height=400,
                              xaxis_title='Epoch', yaxis_title='IoU')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        model_pick = st.selectbox("Choose model for detail", selected_models)
        h = training_histories[model_pick]
        df_epoch = pd.DataFrame({
            'Epoch': h['epochs'], 'Train Loss': h['train_loss'],
            'Val Dice': h['val_dice'], 'Val IoU': h['val_iou']
        })
        best_ep = df_epoch.loc[df_epoch['Val Dice'].idxmax()]
        st.success(f"**Best epoch: {int(best_ep['Epoch'])}** — Dice: {best_ep['Val Dice']:.4f}, IoU: {best_ep['Val IoU']:.4f}")
        st.dataframe(df_epoch.style.highlight_max(subset=['Val Dice', 'Val IoU'], color='#1a3a2a')
                     .highlight_min(subset=['Train Loss'], color='#1a3a2a')
                     .format({'Train Loss': '{:.4f}', 'Val Dice': '{:.4f}', 'Val IoU': '{:.4f}'}),
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# PAGE 4: Model Comparison
# ═══════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    st.markdown('<div class="hero-badge">BENCHMARK RESULTS</div>', unsafe_allow_html=True)
    st.title("🏆 Model Comparison & Analysis")

    # Final scores table
    st.markdown("### Final Validation Metrics")
    st.dataframe(model_comparison.style.highlight_max(subset=['Dice', 'IoU'], color='#1a3a2a')
                 .format({'Dice': '{:.6f}', 'IoU': '{:.6f}'}),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📊 Dice Score Distribution (Per-Sample)")
        model_filter = st.selectbox("Filter by model", ["All"] + per_sample['model'].unique().tolist(), key="dice_dist")
        df_plot = per_sample if model_filter == "All" else per_sample[per_sample['model'] == model_filter]
        # Only tumor-positive samples for meaningful dice comparison
        df_tumor = df_plot[df_plot['has_tumor'] == 1]
        fig = px.histogram(df_tumor, x='dice', color='model', nbins=50, barmode='overlay',
                           color_discrete_sequence=COLORS, opacity=0.7,
                           labels={'dice': 'Dice Score', 'model': 'Model'})
        fig.update_layout(**PLOTLY_LAYOUT, title='Dice Distribution (Tumor Slices Only)', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### 🎯 Confidence vs Dice")
        model_filter2 = st.selectbox("Filter by model", ["All"] + per_sample['model'].unique().tolist(), key="conf_dice")
        df_plot2 = per_sample if model_filter2 == "All" else per_sample[per_sample['model'] == model_filter2]
        df_t2 = df_plot2[df_plot2['has_tumor'] == 1]
        fig = px.scatter(df_t2, x='avg_confidence', y='dice', color='model',
                         color_discrete_sequence=COLORS, opacity=0.5,
                         labels={'avg_confidence': 'Avg Confidence', 'dice': 'Dice Score'})
        fig.update_layout(**PLOTLY_LAYOUT, title='Confidence vs Accuracy', height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Tumor Coverage Distribution")
    fig = px.histogram(tumor_coverage, x='tumor_coverage_pct', nbins=60,
                       color_discrete_sequence=[COLORS[0]],
                       labels={'tumor_coverage_pct': 'Tumor Coverage %'})
    fig.update_layout(**PLOTLY_LAYOUT, title='Tumor Coverage Across Validation Slices', height=350,
                      xaxis_title='Tumor Coverage (%)', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)

    # Per-patient breakdown
    st.markdown("---")
    st.markdown("### 🧬 Per-Patient Average Dice (Best Model)")
    best_model_name = model_comparison.loc[model_comparison['Dice'].idxmax(), 'Model']
    # Map scheduler name
    ps_model_name = best_model_name.replace(' (Scheduler)', '_Scheduler').replace(' ', '')
    if ps_model_name in per_sample['model'].values:
        df_best = per_sample[per_sample['model'] == ps_model_name]
    else:
        df_best = per_sample[per_sample['model'] == per_sample['model'].unique()[-1]]

    df_patient_avg = df_best.groupby('patient_id').agg(
        mean_dice=('dice', 'mean'), mean_iou=('iou', 'mean'), num_slices=('dice', 'count')
    ).reset_index().sort_values('mean_dice')

    fig = px.bar(df_patient_avg, x='patient_id', y='mean_dice',
                 color='mean_dice', color_continuous_scale='Viridis',
                 hover_data=['mean_iou', 'num_slices'],
                 labels={'patient_id': 'Patient', 'mean_dice': 'Mean Dice'})
    fig.update_layout(**PLOTLY_LAYOUT, height=400, xaxis_tickangle=-45, title=None,
                      xaxis_title='Patient ID', yaxis_title='Mean Dice Score')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 5: Prediction Gallery
# ═══════════════════════════════════════════════
elif page == "🖼️ Prediction Gallery":
    st.markdown('<div class="hero-badge">VISUAL RESULTS</div>', unsafe_allow_html=True)
    st.title("🖼️ Prediction Gallery")
    st.markdown("Side-by-side comparison of MRI slices, ground truth masks, model predictions, and overlays.")

    if os.path.exists(GALLERY_DIR):
        all_files = os.listdir(GALLERY_DIR)
        # Group by sample (patient_idx)
        samples = {}
        for f in all_files:
            if f.endswith('_mri.png'):
                key = f.replace('_mri.png', '')
                samples[key] = key

        if samples:
            sample_keys = sorted(samples.keys())
            selected_sample = st.selectbox("Select Sample", sample_keys,
                format_func=lambda x: x.rsplit('_', 1)[0].replace('_', ' ') + f" (slice {x.rsplit('_', 1)[1]})")

            mri_path = os.path.join(GALLERY_DIR, f"{selected_sample}_mri.png")
            gt_path = os.path.join(GALLERY_DIR, f"{selected_sample}_mask_gt.png")
            pred_path = os.path.join(GALLERY_DIR, f"{selected_sample}_mask_pred.png")
            overlay_path = os.path.join(GALLERY_DIR, f"{selected_sample}_overlay.png")

            c1, c2, c3, c4 = st.columns(4)
            if os.path.exists(mri_path):
                c1.image(mri_path, caption="MRI Input", use_container_width=True)
            if os.path.exists(gt_path):
                c2.image(gt_path, caption="Ground Truth", use_container_width=True)
            if os.path.exists(pred_path):
                c3.image(pred_path, caption="Model Prediction", use_container_width=True)
            if os.path.exists(overlay_path):
                c4.image(overlay_path, caption="Overlay", use_container_width=True)

            # Show all samples in grid
            st.markdown("---")
            st.markdown("### 📋 All Samples Overview")
            cols = st.columns(5)
            for i, key in enumerate(sample_keys):
                ov = os.path.join(GALLERY_DIR, f"{key}_overlay.png")
                if os.path.exists(ov):
                    cols[i % 5].image(ov, caption=key.rsplit('_', 1)[0][-15:], use_container_width=True)
        else:
            st.warning("No gallery images found.")
    else:
        st.error(f"Gallery directory not found at `{GALLERY_DIR}`.")
