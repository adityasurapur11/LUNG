# Adaptive Self-Verifying Deep Learning Framework for Early Lung Cancer Detection

An end-to-end AI system and Clinical Decision Support Application for early detection of lung cancer from 3D Computed Tomography (CT) scans. Incorporates Hybrid 3D CNN-Transformers, Confidence Calibration, Explainable AI (XAI), and Radiological Self-Verification.

---

## 🌟 Key Features

1. **Multi-Format Medical Scan Ingestion**:
   - **DICOM Support**: Accepts single `.dcm` files, multi-slice `.dcm` selections, or `.zip` archives containing DICOM series. Automatically extracts Z-axis slice order from `ImagePositionPatient` metadata and converts raw values into Hounsfield Units (HU).
   - **NIfTI Support**: Direct upload of 3D NIfTI volumes (`.nii`, `.nii.gz`).
   - **Pre-Loaded Synthetic Scans**: Includes built-in synthetic LIDC-IDRI 3D scans for instant testing without requiring external data downloads.

2. **Automated Preprocessing & Lung Segmentation**:
   - Automated Hounsfield Unit (HU) windowing specifically tuned for lung tissue (Window Center: -600 HU, Window Width: 1500 HU).
   - Lung Parenchyma extraction using adaptive thresholding and morphological filtering to isolate lungs from surrounding ribs, spine, and external air.

3. **Hybrid 3D CNN-Transformer Architecture**:
   - **3D Convolutional Stem**: Extracts local spatial nodule textures and shape features across axial CT slices.
   - **Spatial Transformer Encoder**: Captures long-range global contextual relationships across 3D depth layers.
   - **Multi-Task Heads**: Predicts primary malignancy risk along with 6 auxiliary radiological characteristics (*Spiculation*, *Lobulation*, *Calcification*, *Subtlety*, *Margin*, *Sphericity*).

4. **Confidence Calibration & Uncertainty Quantification**:
   - **Temperature Scaling**: Post-hoc logit scaling to prevent medical AI overconfidence and align predicted probabilities with true clinical risk.
   - **Monte Carlo (MC) Dropout**: Estimates Epistemic (model knowledge) and Aleatoric (data noise) uncertainty across $N$ stochastic forward passes.

5. **Explainable AI (XAI)**:
   - 3D/2D **Grad-CAM** localization heatmaps overlaid onto axial CT slices to spotlight candidate pulmonary nodules, with interactive transparency (alpha) control.

6. **Self-Verification Engine**:
   - Rule-based clinical cross-validation system that checks the primary malignancy score against predicted radiological traits.
   - Flags **False Positives**, **False Negatives**, or **High Uncertainty Anomalies** with clinical action recommendations.

---

## 📁 Repository Structure

```
LUNG/
├── app.py                      # Streamlit Clinical Decision Support System UI
├── train.py                    # Model training & confidence calibration pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # HU windowing, lung parenchyma segmentation, ROI cropping
│   ├── dataset.py              # LIDC-IDRI / LUNA16 DICOM, NIfTI loader & Synthetic CT Generator
│   ├── model.py                # Hybrid 3D CNN-Transformer multi-task architecture
│   ├── uncertainty_calibration.py # Temperature scaling & MC Dropout uncertainty estimation
│   ├── xai.py                  # 3D/2D Grad-CAM implementation & heatmap overlay generator
│   └── self_verification.py    # Adaptive clinical self-verification engine & safety rules
└── tests/
    └── test_model.py           # Pytest unit tests for all components
```

---

## 🚀 Quick Start Guide

### 1. Installation

Install all required Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests

Verify system integrity across all modules:

```bash
python -m pytest tests/test_model.py
```

### 3. Model Training & Calibration

Train the multi-task model and calibrate confidence scores:

```bash
python train.py
```

### 4. Launch Clinical Web Dashboard

Start the interactive Streamlit application:

```bash
python -m streamlit run app.py
```

Open your web browser at **`http://localhost:8501`**.

---

## 🗂️ Supported Datasets & Benchmarks

You can run the system immediately using the **built-in synthetic scans**, or upload real public CT datasets:

| Dataset | Format | Description |
| :--- | :--- | :--- |
| **Pre-loaded Synthetic** | Built-in | Simulated 3D CT volumes containing benign/malignant nodules for instant zero-setup testing. |
| **LIDC-IDRI / LUNA16** | `.nii.gz` / `.dcm` | The gold standard public dataset of 1,018 chest CT scans annotated by expert radiologists. |
| **TCIA Lung-PET-CT-Dx** | `.dcm` / `.zip` | Public collection of DICOM CT scans categorizing major lung cancer types: <br>• **Class A**: Adenocarcinoma (Most common) <br>• **Class G**: Squamous Cell Carcinoma <br>• **Class B**: Small Cell Carcinoma <br>• **Class E**: Large Cell Carcinoma |

---

## 🩺 Clinical Dashboard Walkthrough

### Sidebar Configuration
- **Input Scan Selection**: Toggle between pre-loaded synthetic patient scans or upload custom `.dcm`, `.zip`, or `.nii.gz` files.
- **Monte Carlo Dropout Samples ($N$)**: Adjust sample iterations (5 to 50) for uncertainty estimation. Higher values increase precision.
- **Temperature Calibration**: Toggle temperature scaling ON/OFF to compare raw vs. calibrated confidence scores.
- **Grad-CAM Overlay Alpha**: Adjust transparency (0.10 to 0.90) of the XAI heatmap overlay on the CT image.

### Main Dashboard Tabs
1. **🩻 CT Scan & Segmentation**: Scroll through Z-axis slices with automatic Hounsfield windowing, binary lung parenchyma masks, and extracted ROIs.
2. **🔥 XAI (Grad-CAM) Visual Explanations**: Side-by-side view of raw activation heatmaps and CT overlays showing exact regions driving AI predictions.
3. **🛡️ Self-Verification & Clinical Diagnosis**: Displays the malignancy probability card, safety verification badge (*Verified Consistent*, *High Uncertainty*, *Discrepancy Flagged*), and 6 radiological nodule traits.
4. **📊 Uncertainty & Calibration Analytics**: Detailed breakdown of model confidence, epistemic uncertainty, aleatoric noise, and calibration curves.
