"""
Unit tests for preprocessing, model forward pass, MC uncertainty, and self-verification engine.
"""

import pytest
import torch
import numpy as np
from src.preprocessing import apply_hu_window, segment_lung_mask_slice, extract_roi
from src.dataset import SyntheticLIDCDataset
from src.model import HybridLungCancerModel
from src.uncertainty_calibration import estimate_mc_uncertainty, TemperatureScaler
from src.xai import GradCAM3D
from src.self_verification import SelfVerificationEngine


def test_preprocessing():
    # HU Windowing
    raw_vol = np.random.uniform(-1200, 500, size=(10, 64, 64))
    windowed = apply_hu_window(raw_vol)
    assert windowed.min() >= 0.0 and windowed.max() <= 1.0
    
    # 2D Segmentation
    dummy_slice = np.random.uniform(0, 1, size=(64, 64))
    mask = segment_lung_mask_slice(dummy_slice)
    assert mask.shape == (64, 64)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    
    # ROI Extraction
    vol = np.random.uniform(0, 1, size=(32, 128, 128))
    roi = extract_roi(vol, center=(16, 64, 64), patch_size=(16, 32, 32))
    assert roi.shape == (16, 32, 32)


def test_synthetic_dataset():
    ds = SyntheticLIDCDataset(num_samples=4, depth=16, height=64, width=64)
    assert len(ds) == 4
    vol_tensor, label, meta = ds[0]
    assert vol_tensor.shape == (1, 16, 64, 64)
    assert label.item() in [0, 1]
    assert meta.shape == (6,)


def test_model_forward():
    model = HybridLungCancerModel(in_channels=1)
    dummy_input = torch.randn(2, 1, 16, 64, 64)
    primary_logits, radiology_traits = model(dummy_input)
    assert primary_logits.shape == (2, 2)
    assert radiology_traits.shape == (2, 6)


def test_mc_uncertainty_and_verification():
    model = HybridLungCancerModel(in_channels=1)
    dummy_input = torch.randn(1, 1, 16, 64, 64)
    
    result = estimate_mc_uncertainty(model, dummy_input, num_samples=5)
    assert "calibrated_prob" in result
    assert "epistemic_uncertainty" in result
    assert 0.0 <= result["calibrated_prob"] <= 1.0
    
    verifier = SelfVerificationEngine()
    verification = verifier.verify_prediction(
        calibrated_prob=result["calibrated_prob"],
        epistemic_uncertainty=result["epistemic_uncertainty"],
        traits=result["trait_predictions"]
    )
    assert "final_diagnosis" in verification
    assert "status" in verification
    assert "recommended_action" in verification


def test_gradcam():
    model = HybridLungCancerModel(in_channels=1)
    target_layer = model.block2.conv
    cam_gen = GradCAM3D(model, target_layer)
    dummy_input = torch.randn(1, 1, 16, 64, 64)
    
    heatmap = cam_gen.generate_heatmap(dummy_input, target_class=1)
    assert heatmap.shape == (16, 64, 64)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
