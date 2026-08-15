"""
Training Script for Adaptive Self-Verifying Hybrid CNN-Transformer Framework.
Trains multi-task primary classifier and radiological trait heads, calibrates confidence,
and saves model checkpoints.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import SyntheticLIDCDataset
from src.model import HybridLungCancerModel
from src.uncertainty_calibration import TemperatureScaler, compute_expected_calibration_error


def train_model(epochs: int = 5, batch_size: int = 4, lr: float = 1e-3, lambda_aux: float = 0.5):
    print("=" * 60)
    print("Starting Adaptive Hybrid CNN-Transformer Training Pipeline...")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Dataset & DataLoader
    train_dataset = SyntheticLIDCDataset(num_samples=32, depth=16, height=64, width=64, seed=42)
    val_dataset = SyntheticLIDCDataset(num_samples=12, depth=16, height=64, width=64, seed=123)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model & Optimizer
    model = HybridLungCancerModel(in_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss Functions
    criterion_primary = nn.CrossEntropyLoss()
    criterion_aux = nn.MSELoss()

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        primary_loss_sum = 0.0
        aux_loss_sum = 0.0

        for vols, labels, traits in train_loader:
            vols, labels, traits = vols.to(device), labels.to(device), traits.to(device)

            optimizer.zero_grad()
            primary_logits, trait_preds = model(vols)

            loss_p = criterion_primary(primary_logits, labels)
            loss_a = criterion_aux(trait_preds, traits)
            loss = loss_p + lambda_aux * loss_a

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * vols.size(0)
            primary_loss_sum += loss_p.item() * vols.size(0)
            aux_loss_sum += loss_a.item() * vols.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_dataset)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f} (Primary: {primary_loss_sum/len(train_dataset):.4f}, Aux Traits: {aux_loss_sum/len(train_dataset):.4f})")

    # 4. Validation & Confidence Calibration
    print("\nCalibrating Temperature Scaling on Validation Set...")
    model.eval()
    val_logits_list = []
    val_labels_list = []

    with torch.no_grad():
        for vols, labels, _ in val_loader:
            vols = vols.to(device)
            p_logits, _ = model(vols)
            val_logits_list.append(p_logits.cpu())
            val_labels_list.append(labels)

    val_logits = torch.cat(val_logits_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)

    scaler = TemperatureScaler(temperature=1.2)
    scaler.calibrate(val_logits, val_labels)
    print(f"Calibrated Optimal Temperature T: {scaler.temperature.item():.4f}")

    # 5. Save Model Checkpoint
    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join("models", "hybrid_lung_cancer_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'temperature': scaler.temperature.item()
    }, checkpoint_path)
    print(f"Model saved successfully to {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_model(epochs=3)
