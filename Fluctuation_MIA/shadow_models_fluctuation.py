"""
shadow_models_fluctuation.py
=============================
Adversarially-trained PyTorch CNN shadow models for the Fluctuation MIA.

Extends the base PyTorchShadowModel with:
  - Adversarial training loop (FGSM, matching the victim's training procedure)
  - ``predict_proba_adversarial()`` — returns (clean_probs, adv_probs) for
    computing fluctuation features
  - ``generate_adversarial_tensors()`` — creates FGSM perturbations for a batch
    of image tensors (used during victim attack inference)

The shadow models are trained to OVERFIT so they faithfully mimic the victim
model's adversarial training behavior (robust overfitting).
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import base classes from the existing shadow_models module
sys.path.insert(0, __import__("os").path.join(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
    "..", "Membership_Inference_Attack",
))
from shadow_models import PyTorchShadowModel, ShadowDataset, InferenceDataset


class FluctuationShadowModel(PyTorchShadowModel):
    """Adversarially-trained shadow model for the Fluctuation MIA.

    Differences from ``PyTorchShadowModel``:
      1. ``fit()`` uses an adversarial training loop (FGSM) so the shadow
         model mimics the victim's robust overfitting behavior.
      2. ``predict_proba_adversarial()`` returns both clean and adversarial
         sigmoid scores for computing fluctuation features.
      3. ``generate_adversarial_tensors()`` creates FGSM perturbations from
         raw image tensors (used during victim attack inference).
      4. Default epochs increased to 40 to ensure the shadow model overfits.

    Parameters
    ----------
    architecture : str
        One of 'resnet18', 'mobilenet_v3_small', 'efficientnet_b0',
        'densenet121', 'shufflenet_v2_x1_0'.
    num_classes : int
        Number of output labels (multi-label).
    epochs : int
        Training epochs.  Set high (40) to ensure overfitting.
    batch_size : int
        Training and inference batch size.
    lr : float
        Adam learning rate.
    epsilon : float
        FGSM perturbation magnitude for adversarial training.
    alpha : float
        Weight for clean loss in the combined objective.
        Combined loss = alpha * clean_loss + (1 - alpha) * adv_loss.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 15,
        epochs: int = 40,
        batch_size: int = 32,
        lr: float = 1e-3,
        epsilon: float = 0.01,
        alpha: float = 0.5,
        random_state: int = 42,
    ):
        super().__init__(
            architecture=architecture,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            random_state=random_state,
        )
        self.epsilon = epsilon
        self.alpha   = alpha

    # ------------------------------------------------------------------
    # Training: adversarial training loop (FGSM)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FluctuationShadowModel":
        """Adversarial training on image paths X with multi-hot labels y.

        For each batch the training loop:
          1. Generates FGSM adversarial examples using the model's own gradients
          2. Trains on combined clean + adversarial loss

        This mimics the victim model's adversarial training procedure so the
        shadow model exhibits the same robust overfitting signature.

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

        # pos_weight: handle class imbalance (same as base class)
        pos_counts = y.sum(axis=0).clip(min=1)
        neg_counts = len(y) - pos_counts
        pw = np.clip(neg_counts / pos_counts, 0.1, 10.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # ── Step 1: Generate FGSM adversarial examples ────────────
                x_grad = images.clone().detach().requires_grad_(True)
                logits_temp = self.model(x_grad)
                loss_temp   = criterion(logits_temp, labels)
                loss_temp.backward()
                grad_sign  = x_grad.grad.data.sign()
                adv_images = torch.clamp(
                    images + self.epsilon * grad_sign, -3.0, 3.0
                ).detach()

                # ── Step 2: Combined clean + adversarial training ─────────
                optimizer.zero_grad()

                clean_logits = self.model(images)
                clean_loss   = criterion(clean_logits, labels)
                (self.alpha * clean_loss).backward()

                adv_logits = self.model(adv_images)
                adv_loss   = criterion(adv_logits, labels)
                ((1.0 - self.alpha) * adv_loss).backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )
                optimizer.step()

                epoch_loss += (
                    self.alpha * clean_loss.item()
                    + (1.0 - self.alpha) * adv_loss.item()
                )

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
    # Adversarial inference: clean + adversarial probs
    # ------------------------------------------------------------------
    def predict_proba_adversarial(
        self,
        X: np.ndarray,
        pseudo_labels: np.ndarray,
        epsilon: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (clean_probs, adv_probs) for image paths X.

        Generates FGSM adversarial examples using this shadow model's own
        gradients and the provided pseudo-labels.

        Parameters
        ----------
        X : np.ndarray of str, shape (N,)
            Image file paths.
        pseudo_labels : np.ndarray of float32, shape (N, num_classes)
            Labels used to compute FGSM loss direction.
        epsilon : float or None
            FGSM perturbation magnitude. Defaults to self.epsilon.

        Returns
        -------
        clean_probs : np.ndarray, shape (N, num_classes)
        adv_probs   : np.ndarray, shape (N, num_classes)
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba_adversarial().")
        if epsilon is None:
            epsilon = self.epsilon

        num_workers = 2 if self.device.type == "cuda" else 0
        pin_memory  = self.device.type == "cuda"

        dataset = ShadowDataset(X, pseudo_labels, self.transform)
        loader  = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        criterion = nn.BCEWithLogitsLoss()

        all_clean_probs = []
        all_adv_probs   = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # ── Clean forward pass ────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                clean_logits = self.model(images)
                clean_probs  = torch.sigmoid(clean_logits).cpu().numpy()
                all_clean_probs.append(clean_probs)

            # ── Generate FGSM adversarial examples ────────────────────────
            # Eval mode: use running stats for BatchNorm (consistent preds)
            # requires_grad on input: needed for FGSM gradient computation
            self.model.eval()
            x = images.clone().detach().requires_grad_(True)
            logits = self.model(x)
            loss   = criterion(logits, labels)
            loss.backward()
            grad_sign  = x.grad.data.sign()
            adv_images = torch.clamp(
                images + epsilon * grad_sign, -3.0, 3.0
            ).detach()

            # ── Adversarial forward pass ──────────────────────────────────
            with torch.no_grad():
                adv_logits = self.model(adv_images)
                adv_probs  = torch.sigmoid(adv_logits).cpu().numpy()
                all_adv_probs.append(adv_probs)

        return np.vstack(all_clean_probs), np.vstack(all_adv_probs)

    # ------------------------------------------------------------------
    # Generate adversarial tensors (for victim attack inference)
    # ------------------------------------------------------------------
    def generate_adversarial_tensors(
        self,
        images_tensor: torch.Tensor,
        pseudo_labels_tensor: torch.Tensor,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        """Create FGSM adversarial image tensors using this model's gradients.

        Used during victim attack inference: the shadow model's gradients
        serve as a proxy for the victim's gradients (transfer attack).

        Parameters
        ----------
        images_tensor : torch.Tensor, shape (N, 3, H, W)
            Clean image tensors (already preprocessed).
        pseudo_labels_tensor : torch.Tensor, shape (N, num_classes)
            Labels for FGSM loss direction.
        epsilon : float or None
            Perturbation magnitude. Defaults to self.epsilon.

        Returns
        -------
        adv_images : torch.Tensor, shape (N, 3, H, W)
        """
        if self.model is None:
            raise RuntimeError("Call fit() before generate_adversarial_tensors().")
        if epsilon is None:
            epsilon = self.epsilon

        self.model.eval()

        x = images_tensor.clone().detach().requires_grad_(True)
        logits = self.model(x)
        loss   = nn.BCEWithLogitsLoss()(logits, pseudo_labels_tensor)
        loss.backward()

        grad_sign  = x.grad.data.sign()
        adv_images = torch.clamp(
            images_tensor + epsilon * grad_sign, -3.0, 3.0
        )
        return adv_images.detach()
