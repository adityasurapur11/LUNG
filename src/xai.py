"""
Explainable Artificial Intelligence (XAI) Module.
Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for 3D CT volumes and 2D slices,
generating visual explanations and highlighting diagnostic regions of interest (nodules, lesions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import cv2
except ImportError:
    cv2 = None

from typing import Tuple, Optional


class GradCAM3D:
    """
    3D Grad-CAM implementation for PyTorch 3D Convolutional Models.
    Hooks into target convolutional layer to record activations and gradients.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)
        
    def _save_activations(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generates 3D Grad-CAM heatmap array matching the input spatial dimensions (D, H, W).
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        primary_logits, _ = self.model(input_tensor)
        
        # Target score for specified class
        target_score = primary_logits[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Gradients & Activations: Shape (1, C, D', H', W')
        grads = self.gradients[0]       # (C, D', H', W')
        acts = self.activations[0]      # (C, D', H', W')
        
        # Global average pooling over spatial dimensions to get channel weights
        weights = torch.mean(grads, dim=(1, 2, 3), keepdim=True)  # (C, 1, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=0)  # (D', H', W')
        cam = F.relu(cam)                       # Keep positive influences
        
        # Normalize cam to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        cam_np = cam.cpu().numpy()
        
        # Interpolate cam_np to input volume shape (D, H, W)
        target_shape = input_tensor.shape[2:]  # (D, H, W)
        cam_tensor = torch.tensor(cam_np).unsqueeze(0).unsqueeze(0)  # (1, 1, D', H', W')
        cam_resized = F.interpolate(cam_tensor, size=target_shape, mode='trilinear', align_corners=False)
        
        return cam_resized.squeeze().cpu().numpy()


def overlay_heatmap_on_slice(ct_slice: np.ndarray, heatmap_slice: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlays a 2D Grad-CAM heatmap on a grayscale CT slice using Jet colormap.
    Returns RGB image array (H, W, 3) normalized to [0, 255] uint8.
    """
    if ct_slice.max() <= 1.0:
        ct_norm = ct_slice.astype(np.float32)
    else:
        ct_norm = (ct_slice / 255.0).astype(np.float32)
        
    ct_rgb = np.stack([ct_norm]*3, axis=-1)
    
    # Normalize heatmap slice to [0, 1]
    if heatmap_slice.max() > heatmap_slice.min():
        heatmap_norm = (heatmap_slice - heatmap_slice.min()) / (heatmap_slice.max() - heatmap_slice.min())
    else:
        heatmap_norm = heatmap_slice

    if cv2 is not None:
        heatmap_uint8 = (heatmap_norm * 255.0).astype(np.uint8)
        color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        # Use matplotlib colormap as fallback
        cmap = cm.get_cmap('jet')
        color_map_rgb = cmap(heatmap_norm)[:, :, :3]
    
    # Blended image
    overlay = (1.0 - alpha) * ct_rgb + alpha * color_map_rgb
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return overlay

