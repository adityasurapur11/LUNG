"""
Confidence Calibration Module & Monte Carlo Uncertainty Estimation.
Implements Temperature Scaling for calibrating prediction confidence and
Monte Carlo (MC) Dropout for quantifying Epistemic and Aleatoric Uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional



class TemperatureScaler(nn.Module):
    """
    Applies Temperature Scaling to calibrate raw neural network logits.
    Temperature T > 0 scales logits: logit / T.
    T > 1 softens probabilities, correcting overconfidence.
    """
    def __init__(self, temperature: float = 1.25):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Scale logits by temperature T
        return logits / torch.clamp(self.temperature, min=0.1)

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        """ Optimizes Temperature T using Negative Log Likelihood (NLL) on validation logits. """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)


def compute_expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ Calculates Expected Calibration Error (ECE) across confidence bins. """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def estimate_mc_uncertainty(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_samples: int = 20,
    temp_scaler: Optional[TemperatureScaler] = None
) -> Dict[str, float]:
    """
    Performs Monte Carlo (MC) Dropout sampling to compute prediction uncertainty scores.
    Returns:
        - calibrated_prob: Calibrated malignancy probability (0.0 to 1.0)
        - epistemic_uncertainty: Variance of Monte Carlo probabilities (Model uncertainty)
        - aleatoric_uncertainty: Mean entropy of individual forward passes (Data noise uncertainty)
        - total_uncertainty: Predictive entropy of mean probability
        - trait_predictions: Mean predicted radiological characteristic traits
        - trait_variances: Variances of predicted traits
    """
    model.eval()
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()
    else:
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()

    mc_logits = []
    mc_traits = []

    with torch.no_grad():
        for _ in range(num_samples):
            primary_logits, traits = model(input_tensor)
            
            if temp_scaler is not None:
                primary_logits = temp_scaler(primary_logits)
                
            mc_logits.append(primary_logits)
            mc_traits.append(traits)

    # Stack samples: (N_samples, B, 2)
    mc_logits_stack = torch.stack(mc_logits, dim=0)
    mc_probs_stack = F.softmax(mc_logits_stack, dim=-1)  # (N_samples, B, 2)
    mc_traits_stack = torch.stack(mc_traits, dim=0)      # (N_samples, B, 6)

    # Extract class 1 (Cancerous) probabilities
    cancer_probs = mc_probs_stack[:, 0, 1].cpu().numpy()  # Array of shape (N_samples,)
    
    # 1. Calibrated Mean Probability
    mean_prob = float(np.mean(cancer_probs))
    
    # 2. Epistemic Uncertainty (Variance across MC samples)
    epistemic_unc = float(np.var(cancer_probs))
    
    # 3. Aleatoric Uncertainty (Average Entropy)
    eps = 1e-7
    probs_all = mc_probs_stack[:, 0, :].cpu().numpy()
    entropies = -np.sum(probs_all * np.log(probs_all + eps), axis=-1)
    aleatoric_unc = float(np.mean(entropies))
    
    # 4. Total Predictive Entropy
    total_unc = - (mean_prob * np.log(mean_prob + eps) + (1.0 - mean_prob) * np.log(1.0 - mean_prob + eps))
    
    # Traits mean and variance
    traits_mean = mc_traits_stack[:, 0, :].mean(dim=0).cpu().numpy().tolist()
    traits_var = mc_traits_stack[:, 0, :].var(dim=0).cpu().numpy().tolist()

    return {
        "calibrated_prob": mean_prob,
        "epistemic_uncertainty": epistemic_unc,
        "aleatoric_uncertainty": aleatoric_unc,
        "total_uncertainty": float(total_unc),
        "trait_predictions": traits_mean,
        "trait_variances": traits_var
    }
