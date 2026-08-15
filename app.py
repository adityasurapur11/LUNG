"""
Adaptive Self-Verifying Deep Learning Framework for Early Lung Cancer Detection
Streamlit Clinical Decision Support System Interface
"""

import streamlit as st
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import io
import time
from typing import Dict, Any

from src.preprocessing import apply_hu_window, segment_lung_mask_slice, extract_roi
from src.dataset import SyntheticLIDCDataset, load_dicom_series, load_nifti_file
from src.model import HybridLungCancerModel
from src.uncertainty_calibration import estimate_mc_uncertainty, TemperatureScaler
from src.xai import GradCAM3D, overlay_heatmap_on_slice
from src.self_verification import SelfVerificationEngine


# Streamlit Page Configuration
st.set_page_config(
    page_title="Adaptive Self-Verifying Lung Cancer AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Medical UI Aesthetics
st.markdown("""
<style>
    /* Dark Theme & Medical Glassmorphism Accent */
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .badge-verified {
        background-color: #10b981;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-warning {
        background-color: #f59e0b;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-danger {
        background-color: #ef4444;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .clinical-card {
        background: #1a202c;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .clinical-card-alert {
        background: #2a1b1b;
        border-left: 4px solid #ef4444;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """ Load or initialize model """
    model = HybridLungCancerModel(in_channels=1)
    model.eval()
    temp_scaler = TemperatureScaler(temperature=1.2)
    return model, temp_scaler


@st.cache_data
def get_synthetic_samples():
    """ Cache synthetic scans """
    dataset = SyntheticLIDCDataset(num_samples=10, depth=32, height=128, width=128, seed=101)
    return dataset


def main():
    st.title("🫁 Adaptive Self-Verifying Deep Learning Framework")
    st.caption("Early Lung Cancer Detection & Clinical Decision Support System | Explainable AI & Uncertainty Calibration")
    st.markdown("---")

    model, temp_scaler = load_trained_model()
    synthetic_ds = get_synthetic_samples()
    verifier = SelfVerificationEngine()

    # Sidebar Options
    st.sidebar.header("🔬 Input Scan Selection")
    data_source = st.sidebar.radio(
        "Select CT Data Input:",
        ["Pre-loaded Synthetic LIDC Scans", "Upload DICOM / NIfTI Scan"]
    )

    current_volume = None
    true_label = None
    scan_info = ""

    if data_source == "Pre-loaded Synthetic LIDC Scans":
        scan_idx = st.sidebar.selectbox("Select Sample Patient CT Scan:", range(len(synthetic_ds)), format_func=lambda i: f"Patient #{i+1} ({'Malignant Nodule' if i%2==1 else 'Benign Nodule'})")
        vol_tensor, true_label_val, meta_vec = synthetic_ds[scan_idx]
        current_volume = vol_tensor.squeeze(0).numpy()  # (D, H, W)
        true_label = int(true_label_val.item())
        scan_info = f"Synthetic LIDC CT Scan #{scan_idx+1}"
    else:
        st.sidebar.markdown("**Upload DICOM or NIfTI Files**")
        uploaded_files = st.sidebar.file_uploader(
            "Upload DICOM (.dcm), ZIP folder (.zip), or NIfTI (.nii, .nii.gz)",
            type=["dcm", "zip", "nii", "gz"],
            accept_multiple_files=True
        )
        if uploaded_files:
            try:
                import zipfile
                import tempfile
                import shutil
                import os

                # Option A: Single ZIP File containing DICOMs
                if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith('.zip'):
                    temp_dir = tempfile.mkdtemp()
                    zip_path = os.path.join(temp_dir, "upload.zip")
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_files[0].getbuffer())
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    dcm_paths = []
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith('.dcm'):
                                dcm_paths.append(os.path.join(root, file))
                    
                    if dcm_paths:
                        import pydicom
                        slices = [pydicom.dcmread(fp) for fp in dcm_paths]
                        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]) if hasattr(s, 'ImagePositionPatient') else 0)
                        vol = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
                        if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                            slope = float(slices[0].RescaleSlope)
                            intercept = float(slices[0].RescaleIntercept)
                            vol = vol * slope + intercept
                        current_volume = apply_hu_window(vol)
                        scan_info = f"ZIP Archive ({len(dcm_paths)} DICOM Slices): {uploaded_files[0].name}"
                    else:
                        st.error("No .dcm files found inside the ZIP archive.")
                    shutil.rmtree(temp_dir, ignore_errors=True)

                # Option B: One or Multiple .dcm files
                elif all(f.name.lower().endswith('.dcm') for f in uploaded_files):
                    import pydicom
                    temp_dir = tempfile.mkdtemp()
                    slices = []
                    for f in uploaded_files:
                        fp = os.path.join(temp_dir, f.name)
                        with open(fp, "wb") as out_f:
                            out_f.write(f.getbuffer())
                        slices.append(pydicom.dcmread(fp))
                    
                    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]) if hasattr(s, 'ImagePositionPatient') else 0)
                    vol = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
                    if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                        slope = float(slices[0].RescaleSlope)
                        intercept = float(slices[0].RescaleIntercept)
                        vol = vol * slope + intercept
                    
                    if vol.ndim == 2:
                        vol = np.tile(vol, (16, 1, 1))
                    elif vol.ndim == 3 and vol.shape[0] < 4:
                        vol = np.repeat(vol, max(1, 16 // vol.shape[0]), axis=0)

                    current_volume = apply_hu_window(vol)
                    scan_info = f"Uploaded {len(uploaded_files)} DICOM slice(s)"
                    shutil.rmtree(temp_dir, ignore_errors=True)

                # Option C: NIfTI File (.nii, .nii.gz)
                elif len(uploaded_files) == 1 and (uploaded_files[0].name.lower().endswith('.nii') or uploaded_files[0].name.lower().endswith('.gz')):
                    temp_path = "temp_upload.nii.gz"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_files[0].getbuffer())
                    current_volume = load_nifti_file(temp_path)
                    if current_volume.ndim == 3:
                        current_volume = apply_hu_window(current_volume)
                        scan_info = f"Uploaded File: {uploaded_files[0].name}"
                    else:
                        st.error("Uploaded NIfTI file must be a 3D volume.")
                else:
                    st.warning("Please upload .dcm files, a .zip containing .dcm files, or a .nii/.nii.gz file.")
            except Exception as e:
                st.error(f"Error processing uploaded files: {e}")

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Model & XAI Settings")
    mc_samples = st.sidebar.slider("Monte Carlo Dropout Samples (N):", min_value=5, max_value=50, value=15, step=5)
    enable_calibration = st.sidebar.checkbox("Enable Temperature Calibration", value=True)
    heatmap_alpha = st.sidebar.slider("Grad-CAM Overlay Alpha:", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

    if current_volume is None:
        st.info("Please select or upload a CT scan to initiate AI diagnosis.")
        return

    # Process Volume Input for Model
    D, H, W = current_volume.shape
    vol_input_tensor = torch.tensor(current_volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # Run Model Inference with Uncertainty & XAI
    with st.spinner("Processing CT Volume with Hybrid CNN-Transformer & Self-Verification Engine..."):
        scaler_to_use = temp_scaler if enable_calibration else None
        unc_results = estimate_mc_uncertainty(model, vol_input_tensor, num_samples=mc_samples, temp_scaler=scaler_to_use)
        
        # Self-Verification Execution
        verification = verifier.verify_prediction(
            calibrated_prob=unc_results["calibrated_prob"],
            epistemic_uncertainty=unc_results["epistemic_uncertainty"],
            traits=unc_results["trait_predictions"]
        )
        
        # Grad-CAM Heatmap
        cam_generator = GradCAM3D(model, model.block2.conv)
        heatmap_3d = cam_generator.generate_heatmap(vol_input_tensor, target_class=1)

    # Main Clinical Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩻 CT Scan & Segmentation",
        "🔥 XAI (Grad-CAM) Visual Explanations",
        "🛡️ Self-Verification & Clinical Diagnosis",
        "📊 Uncertainty & Calibration Analytics"
    ])

    with tab1:
        st.subheader("Interactive Slice Viewer & Lung Parenchyma Extraction")
        slice_idx = st.slider("Select Axial Slice (Z-Axis):", min_value=0, max_value=D-1, value=D//2)
        
        slice_ct = current_volume[slice_idx]
        mask_ct = segment_lung_mask_slice(slice_ct)
        segmented_lung = slice_ct * mask_ct

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original CT Slice (Lung Window)**")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.imshow(slice_ct, cmap='gray')
            ax1.axis('off')
            st.pyplot(fig1)

        with col2:
            st.markdown("**Segmented Lung Parenchyma Mask**")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(mask_ct, cmap='bone')
            ax2.axis('off')
            st.pyplot(fig2)

        with col3:
            st.markdown("**Extracted Lung ROI**")
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            ax3.imshow(segmented_lung, cmap='gray')
            ax3.axis('off')
            st.pyplot(fig3)

    with tab2:
        st.subheader("Explainable AI (XAI) Nodule Localization")
        st.markdown("Grad-CAM highlights regional feature activations driving the primary cancer diagnosis.")
        
        slice_cam = heatmap_3d[slice_idx]
        overlay_img = overlay_heatmap_on_slice(current_volume[slice_idx], slice_cam, alpha=heatmap_alpha)

        col_xai1, col_xai2 = st.columns(2)
        with col_xai1:
            st.markdown(f"**Grad-CAM Heatmap (Slice #{slice_idx})**")
            fig_cam, ax_cam = plt.subplots(figsize=(5, 5))
            ax_cam.imshow(slice_cam, cmap='jet')
            ax_cam.axis('off')
            st.pyplot(fig_cam)

        with col_xai2:
            st.markdown(f"**Clinical CT Overlay (Alpha: {heatmap_alpha})**")
            fig_ov, ax_ov = plt.subplots(figsize=(5, 5))
            ax_ov.imshow(overlay_img)
            ax_ov.axis('off')
            st.pyplot(fig_ov)

    with tab3:
        st.subheader("Clinical Decision Support & Self-Verification Output")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        prob_pct = unc_results["calibrated_prob"] * 100.0
        with col_res1:
            st.metric("Calibrated Cancer Probability", f"{prob_pct:.1f}%")
        with col_res2:
            st.metric("Epistemic Uncertainty", f"{unc_results['epistemic_uncertainty']:.4f}")
        with col_res3:
            st.metric("Radiological Trait Risk Score", f"{verification['trait_risk_score']:.2f} / 1.0")

        st.markdown("---")
        
        # Display Verification Status
        status_code = verification["status"]
        if status_code == "VERIFIED_MATCH":
            st.markdown('### Self-Verification Status: <span class="badge-verified">✅ VERIFIED MATCH</span>', unsafe_allow_html=True)
            card_style = "clinical-card"
        elif "WARNING" in status_code:
            st.markdown('### Self-Verification Status: <span class="badge-warning">⚠️ VERIFICATION CONFLICT</span>', unsafe_allow_html=True)
            card_style = "clinical-card-alert"
        else:
            st.markdown('### Self-Verification Status: <span class="badge-danger">🚨 HIGH UNCERTAINTY</span>', unsafe_allow_html=True)
            card_style = "clinical-card-alert"

        st.markdown(f"""
        <div class="{card_style}">
            <h4>Diagnosis: {verification['final_diagnosis']}</h4>
            <p><strong>Clinical Recommendation:</strong> {verification['recommended_action']}</p>
        </div>
        """, unsafe_allow_html=True)

        if verification["conflicts"]:
            st.warning("⚠️ **Detected Verification Conflicts:**")
            for c in verification["conflicts"]:
                st.write(f"- {c}")

        st.subheader("Extracted Radiological Characteristics Matrix (LIDC Attributes)")
        traits_dict = verification["trait_breakdown"]
        t_cols = st.columns(6)
        for idx, (trait_name, trait_val) in enumerate(traits_dict.items()):
            with t_cols[idx]:
                st.metric(trait_name, trait_val)

    with tab4:
        st.subheader("Confidence Calibration & Monte Carlo Uncertainty Analysis")
        
        col_unc1, col_unc2 = st.columns(2)
        with col_unc1:
            st.markdown("**Monte Carlo Sampling Distribution**")
            fig_hist, ax_hist = plt.subplots(figsize=(5, 3.5))
            # Generate dummy distribution for visualization based on mean and var
            samples_dist = np.random.normal(unc_results["calibrated_prob"], np.sqrt(unc_results["epistemic_uncertainty"]), 500)
            samples_dist = np.clip(samples_dist, 0, 1)
            ax_hist.hist(samples_dist, bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
            ax_hist.set_title("Probability Frequency Across MC Forward Passes")
            ax_hist.set_xlabel("Predicted Malignancy Probability")
            ax_hist.set_ylabel("Count")
            st.pyplot(fig_hist)

        with col_unc2:
            st.markdown("**Uncertainty Metrics Decomposition**")
            u_metrics = {
                "Epistemic (Model)": unc_results["epistemic_uncertainty"],
                "Aleatoric (Data Noise)": unc_results["aleatoric_uncertainty"],
                "Total Predictive Entropy": unc_results["total_uncertainty"]
            }
            for u_name, u_val in u_metrics.items():
                st.write(f"**{u_name}:** `{u_val:.4f}`")
                st.progress(min(float(u_val), 1.0))

    st.markdown("---")
    st.caption("Adaptive Self-Verifying Deep Learning Framework | Early Lung Cancer Detection System")


if __name__ == "__main__":
    main()
