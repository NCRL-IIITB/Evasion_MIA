"""
mia_fluctuation.py
==================
Fluctuation-Based Membership Inference Attack

Instead of training the attack model on raw confidence scores (as in the
standard MIA), this attack trains on **fluctuation features** — the absolute
difference between a model's output on clean vs adversarial inputs.

Key insight
-----------
Adversarially trained models exhibit "Robust Overfitting": they are much more
stable (lower fluctuation) on members than on non-members.  This differential
fluctuation is the MIA signal.

Attack pipeline
---------------
  1. Train adversarially-trained shadow models (via FluctuationShadowModel)
  2. For each shadow model's members / non-members:
       a. Query the shadow model on clean AND adversarial images
       b. Compute fluctuation = |clean_probs - adv_probs|
       c. Record: fluct_0…fluct_14, max_confidence (clean), max_class (clean),
          is_part_of_dataset
  3. Train an attack model on the fluctuation features
  4. Attack the victim:
       a. Load eval images as tensors
       b. Pass through victim model → clean probs
       c. Use a shadow model's gradients to craft FGSM adversarial images
       d. Pass adversarial images through victim model → adv probs
       e. Compute fluctuation → feed to attack model → membership prediction

The victim model is treated as a BLACKBOX.  Shadow model gradients serve as a
proxy for the victim's gradients (transfer attack).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Resolve imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIA_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), "Membership_Inference_Attack")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, MIA_DIR)

from mia import MIA, ModelParameters, API
from shadow_models_fluctuation import FluctuationShadowModel

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper dataset: loads images from paths as tensors (no labels)
# ---------------------------------------------------------------------------
class _PathImageDataset(Dataset):
    """Loads images from file paths and returns preprocessed tensors."""

    def __init__(self, paths, transform):
        self.paths     = paths
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
# FluctuationMIA
# ---------------------------------------------------------------------------
class FluctuationMIA(MIA):
    """Fluctuation-Based Membership Inference Attack.

    Inherits all constructor arguments from MIA and adds:

    Parameters
    ----------
    epsilon : float
        FGSM perturbation magnitude for adversarial generation.
        Default: 0.01.
    """

    def __init__(self, *args, epsilon: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

        # Cache: pseudo-labels per shadow model (needed for adversarial queries)
        self._shadow_pseudo_labels: dict = {}

        # Reference to the first shadow model (used for transfer attacks on victim)
        self._reference_shadow: FluctuationShadowModel | None = None

    # ------------------------------------------------------------------
    # Public API overrides
    # ------------------------------------------------------------------
    def execute(self) -> "FluctuationMIA":
        """Full pipeline: train shadows → build fluctuation dataset → train attack."""
        self.execute_shadow_phase()
        print("[FluctuationMIA] Step 3/3: Training attack model …", flush=True)
        self._train_attack_model()
        self._is_trained = True
        print("[FluctuationMIA] Pipeline complete ✓", flush=True)
        return self

    def execute_shadow_phase(self) -> "FluctuationMIA":
        """Steps 1 + 2: train shadow models and build fluctuation attack dataset."""
        print("[FluctuationMIA] Step 1/2: Training shadow models (adversarial) …",
              flush=True)
        self._train_shadow_models()
        print("[FluctuationMIA] Step 2/2: Preparing fluctuation attack dataset …",
              flush=True)
        self._prepare_attack_dataset()
        print("[FluctuationMIA] Shadow phase complete ✓", flush=True)
        return self

    def evaluate_attack_model(
        self,
        attack_params: "ModelParameters",
        member_data: np.ndarray,
        non_member_data: np.ndarray,
    ) -> dict:
        """Train ONE attack model on the fluctuation dataset and evaluate it."""
        if self.attack_dataset is None:
            raise RuntimeError(
                "Call execute_shadow_phase() before evaluate_attack_model()."
            )

        feature_cols = (
            [f"fluct_{c}" for c in range(self.num_classes)]
            + ["max_confidence", "max_confidence_class"]
        )
        X = self.attack_dataset[feature_cols].values
        y = self.attack_dataset["is_part_of_dataset"].values

        model = attack_params.build(random_state=self.random_state)
        model.fit(X, y)
        print(f"  Attack model trained: {attack_params.model_type}", flush=True)

        # Temporarily swap in this attack model for the evaluate() call
        prev_model   = self.attack_model
        prev_params  = self.attack_model_parameters
        prev_trained = self._is_trained

        self.attack_model            = model
        self.attack_model_parameters = attack_params
        self._is_trained             = True

        X_eval = np.concatenate([member_data, non_member_data])
        y_true = np.concatenate([
            np.ones(len(member_data)),
            np.zeros(len(non_member_data)),
        ])
        y_pred = self.attack(X_eval)

        metrics = {
            "attack_model": attack_params.model_type,
            "accuracy":     accuracy_score(y_true, y_pred),
            "precision":    precision_score(y_true, y_pred, zero_division=0),
            "recall":       recall_score(y_true, y_pred, zero_division=0),
            "f1":           f1_score(y_true, y_pred, zero_division=0),
        }

        # Restore previous state
        self.attack_model            = prev_model
        self.attack_model_parameters = prev_params
        self._is_trained             = prev_trained

        return metrics

    # ------------------------------------------------------------------
    # Step 1: Train adversarial shadow models
    # ------------------------------------------------------------------
    def _train_shadow_models(self):
        """Train adversarial shadow models and cache pseudo-labels.

        Each shadow model is adversarially trained (FGSM) to mimic the
        victim's robust overfitting behavior.  Pseudo-labels from the
        victim API are cached for later use in adversarial inference.
        """
        rng = np.random.RandomState(self.random_state)
        n   = len(self.unlabelled_data)

        for i in range(self.num_shadow_models):
            # Sample training paths for this shadow model
            train_indices = rng.choice(
                n, size=self.shadow_model_dataset_size, replace=False
            )
            train_paths = self.unlabelled_data[train_indices]

            # Query victim API → pseudo multi-hot labels (threshold at 0.5)
            victim_scores = self.victim_model_api.predict(train_paths)
            pseudo_labels = (victim_scores > 0.5).astype(np.float32)

            # Pick shadow model parameters (cycle through list if given)
            if isinstance(self.shadow_model_parameters, list):
                param = self.shadow_model_parameters[
                    i % len(self.shadow_model_parameters)
                ]
            else:
                param = self.shadow_model_parameters

            shadow_model = param.build(random_state=self.random_state + i)
            shadow_model.fit(train_paths, pseudo_labels)

            self.shadow_models[i] = {
                "model":         shadow_model,
                "train_indices": set(train_indices.tolist()),
            }

            # Cache pseudo-labels for adversarial inference during dataset prep
            self._shadow_pseudo_labels[i] = {
                "train_paths":   train_paths,
                "pseudo_labels": pseudo_labels,
            }

            # Keep first shadow model as reference for transfer attacks
            if self._reference_shadow is None:
                self._reference_shadow = shadow_model

            arch_name = getattr(shadow_model, "architecture", param.model_type)
            print(
                f"  Shadow model {i + 1}/{self.num_shadow_models} trained "
                f"({arch_name}).",
                flush=True,
            )
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 2: Build fluctuation attack dataset
    # ------------------------------------------------------------------
    def _prepare_attack_dataset(self):
        """Build attack dataset with fluctuation features.

        For each shadow model:
          1. Select member and non-member subsets
          2. Query shadow model for clean AND adversarial probabilities
          3. Compute fluctuation = |clean - adv|
          4. Record: fluct_0…fluct_14, max_confidence (clean),
             max_confidence_class (clean), is_part_of_dataset

        Columns: fluct_0…fluct_14, max_confidence, max_confidence_class,
                 is_part_of_dataset
        """
        n           = len(self.unlabelled_data)
        all_indices = set(range(n))

        if self.attack_model_dataset_size is not None:
            num_per_model = self.attack_model_dataset_size // (
                2 * self.num_shadow_models
            )
        else:
            num_per_model = self.shadow_model_dataset_size // 2

        rng  = np.random.RandomState(self.random_state + 1)
        rows = []

        for idx, info in self.shadow_models.items():
            model         = info["model"]
            train_indices = info["train_indices"]
            non_train_idx = all_indices - train_indices

            # Get stored pseudo-labels for this shadow model
            cached = self._shadow_pseudo_labels[idx]

            # ── Positive samples (member = 1) ─────────────────────────────
            pos_pool = list(train_indices)
            k_pos    = min(num_per_model, len(pos_pool))
            pos_idx  = rng.choice(pos_pool, size=k_pos, replace=False)
            pos_data = self.unlabelled_data[pos_idx]

            # Build pseudo-labels for these specific images
            # Map global indices → pseudo-labels from training cache
            train_idx_list = cached["train_paths"].tolist()
            idx_to_label   = {
                path: cached["pseudo_labels"][j]
                for j, path in enumerate(train_idx_list)
            }
            pos_pseudo = np.array([
                idx_to_label.get(p, np.zeros(self.num_classes))
                for p in pos_data
            ], dtype=np.float32)

            clean_pos, adv_pos = model.predict_proba_adversarial(
                pos_data, pos_pseudo, self.epsilon
            )
            fluct_pos = np.abs(clean_pos - adv_pos)

            for j in range(k_pos):
                row = {
                    f"fluct_{c}": fluct_pos[j, c]
                    for c in range(self.num_classes)
                }
                row["max_confidence"]       = float(np.max(clean_pos[j]))
                row["max_confidence_class"] = int(np.argmax(clean_pos[j]))
                row["is_part_of_dataset"]   = 1
                rows.append(row)

            # ── Negative samples (non-member = 0) ─────────────────────────
            neg_pool = list(non_train_idx)
            k_neg    = min(num_per_model, len(neg_pool))
            neg_idx  = rng.choice(neg_pool, size=k_neg, replace=False)
            neg_data = self.unlabelled_data[neg_idx]

            # For non-members: use shadow model's own predictions as pseudo-labels
            neg_clean_probs = model.predict_proba(neg_data)
            neg_pseudo = (neg_clean_probs > 0.5).astype(np.float32)

            clean_neg, adv_neg = model.predict_proba_adversarial(
                neg_data, neg_pseudo, self.epsilon
            )
            fluct_neg = np.abs(clean_neg - adv_neg)

            for j in range(k_neg):
                row = {
                    f"fluct_{c}": fluct_neg[j, c]
                    for c in range(self.num_classes)
                }
                row["max_confidence"]       = float(np.max(clean_neg[j]))
                row["max_confidence_class"] = int(np.argmax(clean_neg[j]))
                row["is_part_of_dataset"]   = 0
                rows.append(row)

        self.attack_dataset = pd.DataFrame(rows)
        n_pos = int(self.attack_dataset["is_part_of_dataset"].sum())
        n_neg = len(self.attack_dataset) - n_pos
        print(
            f"  Attack dataset built: {len(self.attack_dataset)} samples "
            f"({n_pos} positive, {n_neg} negative).",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Step 3: Train attack model on fluctuation features
    # ------------------------------------------------------------------
    def _train_attack_model(self):
        feature_cols = (
            [f"fluct_{c}" for c in range(self.num_classes)]
            + ["max_confidence", "max_confidence_class"]
        )
        X = self.attack_dataset[feature_cols].values
        y = self.attack_dataset["is_part_of_dataset"].values

        self.attack_model = self.attack_model_parameters.build(
            random_state=self.random_state
        )
        self.attack_model.fit(X, y)
        print(
            f"  Attack model trained ({self.attack_model_parameters.model_type}) "
            f"on {X.shape[1]} fluctuation features.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Attack: victim inference via transfer-based adversarial fluctuation
    # ------------------------------------------------------------------
    def attack(self, data: np.ndarray, return_confidence: bool = False):
        """Predict membership using fluctuation features.

        The victim model is a BLACKBOX.  We use a trained shadow model's
        gradients to craft FGSM adversarial examples (transfer attack),
        then observe the victim model's differential response.

        Steps:
          1. Load images as tensors
          2. Forward through victim model → clean probs
          3. Use shadow model gradients to create adversarial images
          4. Forward adversarial images through victim model → adv probs
          5. Compute fluctuation = |clean - adv|
          6. Feed to attack model → membership prediction

        Parameters
        ----------
        data : np.ndarray of str, shape (N,)
            Image file paths.
        return_confidence : bool
            If True, also return attack model probability estimates.

        Returns
        -------
        predictions : np.ndarray of {0, 1}
        confidences : np.ndarray (only if return_confidence=True)
        """
        if not self._is_trained:
            raise RuntimeError("Call .execute() before .attack().")
        if self._reference_shadow is None:
            raise RuntimeError(
                "No reference shadow model available. "
                "Call execute_shadow_phase() first."
            )

        victim_model = self.victim_model_api.model
        victim_device = self.victim_model_api.device
        victim_transform = self.victim_model_api.transform

        shadow = self._reference_shadow
        shadow_device = shadow.device

        # Load images as tensors
        dataset = _PathImageDataset(data, victim_transform)
        loader  = DataLoader(
            dataset,
            batch_size=self.victim_model_api.batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_fluct       = []
        all_max_conf    = []
        all_max_class   = []

        for images in loader:
            images_v = images.to(victim_device)

            # ── Victim clean forward ──────────────────────────────────────
            victim_model.eval()
            with torch.no_grad():
                clean_logits = victim_model(images_v)
                clean_probs  = torch.sigmoid(clean_logits)

            # ── Generate adversarial via shadow model (transfer attack) ───
            # Use victim's clean predictions as pseudo-labels for FGSM
            pseudo_labels = (clean_probs > 0.5).float().detach()

            images_s = images.to(shadow_device)
            pseudo_s = pseudo_labels.to(shadow_device)
            adv_images_s = shadow.generate_adversarial_tensors(
                images_s, pseudo_s, self.epsilon
            )

            # Move adversarial images to victim device
            adv_images_v = adv_images_s.to(victim_device)

            # ── Victim adversarial forward ────────────────────────────────
            victim_model.eval()
            with torch.no_grad():
                adv_logits = victim_model(adv_images_v)
                adv_probs  = torch.sigmoid(adv_logits)

            # ── Compute fluctuation ───────────────────────────────────────
            fluct = torch.abs(clean_probs - adv_probs).cpu().numpy()
            all_fluct.append(fluct)

            # max_confidence and max_class from CLEAN (not adversarial)
            clean_np = clean_probs.cpu().numpy()
            all_max_conf.append(np.max(clean_np, axis=1))
            all_max_class.append(np.argmax(clean_np, axis=1).astype(float))

        fluctuations  = np.vstack(all_fluct)           # (N, num_classes)
        max_conf_arr  = np.concatenate(all_max_conf)   # (N,)
        max_class_arr = np.concatenate(all_max_class)   # (N,)

        features = np.hstack([
            fluctuations,
            max_conf_arr.reshape(-1, 1),
            max_class_arr.reshape(-1, 1),
        ])  # (N, num_classes + 2)

        predictions = self.attack_model.predict(features)

        if return_confidence:
            if hasattr(self.attack_model, "predict_proba"):
                proba = self.attack_model.predict_proba(features)
            else:
                proba = self.attack_model.decision_function(features)
            return predictions, proba

        return predictions

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        member_data: np.ndarray,
        non_member_data: np.ndarray,
    ) -> dict:
        """Evaluate attack on known member / non-member image paths."""
        X      = np.concatenate([member_data, non_member_data])
        y_true = np.concatenate([
            np.ones(len(member_data)),
            np.zeros(len(non_member_data)),
        ])
        y_pred = self.attack(X)
        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        }

    # ------------------------------------------------------------------
    def __repr__(self):
        status = "trained" if self._is_trained else "not trained"
        return (
            f"FluctuationMIA(num_classes={self.num_classes}, "
            f"num_shadow_models={self.num_shadow_models}, "
            f"epsilon={self.epsilon}, "
            f"shadow={self.shadow_model_parameters!r}, "
            f"attack={self.attack_model_parameters!r}, "
            f"features=fluctuation, "
            f"status={status})"
        )
