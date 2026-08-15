"""
Dataset Loader and Synthetic Generator for LIDC-IDRI / LUNA16 CT Scans.
Supports DICOM, NIfTI loading, as well as synthetic multi-slice CT volume generation
with realistic pulmonary nodule features and LIDC radiological annotations.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


class SyntheticLIDCDataset(Dataset):
    """
    Generates realistic 3D synthetic CT volumes containing simulated lung fields,
    vessels, and pulmonary nodules of varying sizes, shapes, and radiological traits.
    Used for rapid local development, unit testing, and demonstration.
    """
    def __init__(self, num_samples: int = 40, depth: int = 32, height: int = 128, width: int = 128, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.depth = depth
        self.height = height
        self.width = width
        np.random.seed(seed)
        
        self.data = []
        for i in range(num_samples):
            is_malignant = int(i % 2 == 1)
            vol, nodule_meta = self._generate_synthetic_ct(is_malignant)
            self.data.append((vol, is_malignant, nodule_meta))
            
    def _generate_synthetic_ct(self, is_malignant: int) -> Tuple[np.ndarray, Dict[str, float]]:
        vol = np.zeros((self.depth, self.height, self.width), dtype=np.float32)
        
        # 3D Mesh Grids
        z_grid, y_grid, x_grid = np.ogrid[:self.depth, :self.height, :self.width]
        center_y, center_x = self.height // 2, self.width // 2
        body_radius = min(self.height, self.width) // 2 - 10
        
        body_mask_2d = ((y_grid - center_y)**2 + (x_grid - center_x)**2) <= body_radius**2
        body_mask_3d = np.broadcast_to(body_mask_2d, vol.shape)
        
        noise_bg = np.random.normal(0, 0.02, size=vol.shape)
        vol[body_mask_3d] = 0.35 + noise_bg[body_mask_3d]
        
        # Lungs (two dark ellipses)
        left_lung_2d = ((y_grid - center_y)**2 / (35**2) + (x_grid - (center_x - 22))**2 / (20**2)) <= 1.0
        right_lung_2d = ((y_grid - center_y)**2 / (35**2) + (x_grid - (center_x + 22))**2 / (20**2)) <= 1.0
        
        left_lung_3d = np.broadcast_to(left_lung_2d, vol.shape)
        right_lung_3d = np.broadcast_to(right_lung_2d, vol.shape)
        
        vol[left_lung_3d] = 0.05 + noise_bg[left_lung_3d] * 0.5
        vol[right_lung_3d] = 0.05 + noise_bg[right_lung_3d] * 0.5
        
        # Add random pulmonary vessels inside lungs
        num_vessels = np.random.randint(5, 12)
        for _ in range(num_vessels):
            vz = np.random.randint(5, self.depth - 5)
            vy = np.random.randint(center_y - 25, center_y + 25)
            vx = np.random.randint(center_x - 35, center_x + 35)
            r = np.random.randint(1, 3)
            vessel_mask = ((z_grid - vz)**2 + (y_grid - vy)**2 + (x_grid - vx)**2) <= r**2
            vol[vessel_mask] += 0.25
            
        # Place Nodule inside lung
        nz = self.depth // 2 + np.random.randint(-4, 5)
        lung_choice = np.random.choice([-1, 1])
        nx = center_x + lung_choice * 22 + np.random.randint(-5, 6)
        ny = center_y + np.random.randint(-10, 11)
        
        nodule_radius = np.random.uniform(3.0, 7.0) if is_malignant else np.random.uniform(1.5, 3.5)
        
        # Radiologic characteristics
        spiculation = np.random.uniform(3.5, 5.0) if is_malignant else np.random.uniform(1.0, 2.0)
        lobulation = np.random.uniform(3.0, 5.0) if is_malignant else np.random.uniform(1.0, 2.5)
        calcification = np.random.uniform(5.0, 6.0) if is_malignant else np.random.uniform(1.0, 3.0)
        subtlety = np.random.uniform(3.0, 5.0)
        margin = np.random.uniform(1.0, 2.5) if is_malignant else np.random.uniform(4.0, 5.0)
        sphericity = np.random.uniform(1.0, 3.0) if is_malignant else np.random.uniform(4.0, 5.0)
        
        # Generate Nodule intensity sphere with optional spiculation noise
        dist = np.sqrt((z_grid - nz)**2 + (y_grid - ny)**2 + (x_grid - nx)**2)
        if is_malignant and spiculation > 3.0:
            noise_spic = np.random.uniform(0, 0.4, size=vol.shape) * (dist < nodule_radius * 1.5)
            nodule_mask = (dist <= nodule_radius) | (noise_spic > 0.25)
        else:
            nodule_mask = dist <= nodule_radius
            
        vol[nodule_mask] = np.clip(vol[nodule_mask] + 0.45 + np.random.normal(0, 0.05, size=vol[nodule_mask].shape), 0, 1.0)

        
        metadata = {
            "spiculation": float(spiculation),
            "lobulation": float(lobulation),
            "calcification": float(calcification),
            "subtlety": float(subtlety),
            "margin": float(margin),
            "sphericity": float(sphericity),
            "nodule_center": (nz, ny, nx),
            "nodule_radius": float(nodule_radius)
        }
        
        return np.clip(vol, 0, 1.0), metadata

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        vol, is_malignant, nodule_meta = self.data[idx]
        # Add channel dim: (1, D, H, W)
        vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(is_malignant, dtype=torch.long)
        
        meta_vector = torch.tensor([
            nodule_meta["spiculation"],
            nodule_meta["lobulation"],
            nodule_meta["calcification"],
            nodule_meta["subtlety"],
            nodule_meta["margin"],
            nodule_meta["sphericity"]
        ], dtype=torch.float32)
        
        return vol_tensor, label_tensor, meta_vector


def load_dicom_series(folder_path: str) -> np.ndarray:
    """ Loads a folder containing DICOM slices and stacks them into a 3D volume (D, H, W). """
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files. Please install pydicom.")
    
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm files found in {folder_path}")
        
    slices = [pydicom.dcmread(f) for f in dicom_files]
    # Sort slices by ImagePositionPatient Z coordinate
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]) if hasattr(s, 'ImagePositionPatient') else 0)
    
    vol = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
    
    # Rescale to Hounsfield Units if parameters exist
    if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
        slope = float(slices[0].RescaleSlope)
        intercept = float(slices[0].RescaleIntercept)
        vol = vol * slope + intercept
        
    return vol


def load_nifti_file(file_path: str) -> np.ndarray:
    """ Loads a .nii or .nii.gz file into a 3D NumPy array (D, H, W). """
    if sitk is None:
        raise ImportError("SimpleITK is required to read NIfTI files. Please install SimpleITK.")
    
    img = sitk.ReadImage(file_path)
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    return vol
