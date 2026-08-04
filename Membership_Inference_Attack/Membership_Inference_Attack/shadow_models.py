"""
shadow_models.py
================
PyTorch CNN wrappers that expose an sklearn-compatible interface
(fit / predict_proba) so they can be used as shadow models inside mia.py.

Supported architectures:
  - resnet18
  - mobilenet_v3_small
  - efficientnet_b0
  - densenet121
  - shufflenet_v2_x1_0

Each shadow model:
  - Takes an array of image file paths as X
  - Takes a 2D float32 multi-hot label array as y  (shape: N × num_classes)
  - Returns sigmoid probabilities from predict_proba()  (shape: N × num_classes)
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ShadowDataset(Dataset):
    """Loads images from file paths with optional labels (for training)."""

    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 224, 224)

        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.float32)


class InferenceDataset(Dataset):
    """Loads images from file paths without labels (for inference)."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 224, 224)
        return img


# ---------------------------------------------------------------------------
# PyTorchShadowModel
# ---------------------------------------------------------------------------

class PyTorchShadowModel:
    """sklearn-compatible wrapper around a pretrained PyTorch CNN.

    Parameters
    ----------
    architecture : str
        One of 'resnet18', 'mobilenet_v3_small', 'efficientnet_b0', 'densenet121'.
    num_classes : int
        Number of output labels (multi-label).
    epochs : int
        Training epochs per shadow model.
    batch_size : int
        Training and inference batch size.
    lr : float
        Adam learning rate.
    random_state : int
        Random seed for reproducibility.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 15,
        epochs: int = 25,       # increased from 15 — shadow models need more training
        batch_size: int = 32,
        lr: float = 1e-3,
        random_state: int = 42,
    ):
        self.architecture = architecture
        self.num_classes  = num_classes
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Build the backbone
    # ------------------------------------------------------------------
    def _build_model(self) -> nn.Module:
        torch.manual_seed(self.random_state)

        arch = self.architecture.lower()

        if arch == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        elif arch == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
            model.classifier[3] = nn.Linear(
                model.classifier[3].in_features, self.num_classes
            )

        elif arch == "efficientnet_b0":
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, self.num_classes
            )

        elif arch == "densenet121":
            model = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1
            )
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_classes),
            )

        elif arch == "shufflenet_v2_x1_0":
            model = models.shufflenet_v2_x1_0(
                weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
            )
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        else:
            raise ValueError(
                f"Unsupported architecture: '{self.architecture}'. "
                f"Choose from: resnet18, mobilenet_v3_small, efficientnet_b0, "
                f"densenet121, shufflenet_v2_x1_0."
            )

        return model.to(self.device)

    # ------------------------------------------------------------------
    # Training (sklearn-style fit)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchShadowModel":
        """Train on image paths X with multi-hot labels y.

        Parameters
        ----------
        X : np.ndarray of str, shape (N,)
            Array of absolute file paths to images.
        y : np.ndarray of float32, shape (N, num_classes)
            Multi-hot label matrix (pseudo-labels from victim API).
        """
        self.model = self._build_model()
        self.model.train()

        num_workers = 2 if self.device.type == "cuda" else 0
        pin_memory  = self.device.type == "cuda"

        dataset = ShadowDataset(X, y, self.transform)
        loader  = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # pos_weight: same class imbalance as victim — pseudo-labels from the
        # victim API are sparse (most classes = 0 for most images).  Without
        # pos_weight the shadow model collapses to all-zeros, which makes the
        # attack dataset meaningless (members and non-members look identical).
        pos_counts = y.sum(axis=0).clip(min=1)
        neg_counts = len(y) - pos_counts
        pw = np.clip(neg_counts / pos_counts, 0.1, 10.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(images)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            print(
                f"      [{self.architecture}] "
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Avg Loss: {avg_loss:.4f}",
                flush=True,
            )
            sys.stdout.flush()

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid probabilities for image paths X.

        Parameters
        ----------
        X : np.ndarray of str, shape (N,)
            Array of absolute file paths to images.

        Returns
        -------
        probs : np.ndarray of float32, shape (N, num_classes)
            Sigmoid confidence scores.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")

        self.model.eval()

        num_workers = 2 if self.device.type == "cuda" else 0
        pin_memory  = self.device.type == "cuda"

        dataset = InferenceDataset(X, self.transform)
        loader  = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        all_probs = []
        with torch.no_grad():
            for images in loader:
                images = images.to(self.device, non_blocking=True)
                logits = self.model(images)
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)
