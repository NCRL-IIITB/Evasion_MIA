"""
mia_fluctuation_variance.py
============================
Variance-Enhanced Fluctuation MIA

Extends FluctuationMIA by adding one extra feature per sample:
    variance_of_max — bootstrapped variance of the max confidence score
    across multiple data points.

IMPORTANT: The variance is calculated on the max confidence score of the
CLEAN (normal) image only, NOT the adversarial image.

Feature columns
---------------
    fluct_0, …, fluct_14,    ← |clean_probs - adv_probs| per class
    max_confidence,          ← max(clean_probs) — from clean image
    max_confidence_class,    ← argmax(clean_probs) — from clean image
    variance_of_max,         ← bootstrapped var(max(clean_probs)) across K samples
    is_part_of_dataset       ← 1 = member, 0 = non-member
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIA_DIR    = os.path.join(os.path.dirname(SCRIPT_DIR), "Membership_Inference_Attack")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, MIA_DIR)

from mia import ModelParameters
from mia_fluctuation import FluctuationMIA

# Bootstrap sample size for computing per-row variance
BOOTSTRAP_K = 100


class FluctuationVarianceMIA(FluctuationMIA):
    """Variance-Enhanced Fluctuation MIA.

    Inherits all parameters from FluctuationMIA.  Adds bootstrapped
    variance_of_max (computed from CLEAN max-confidence scores only).

    Shadow models are trained ONCE and shared between the base
    FluctuationMIA and this variance-enhanced version.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache for the victim model's CLEAN max-confidence pool
        self._victim_max_conf_pool = None

    # ------------------------------------------------------------------
    # Helper: bootstrap variance
    # ------------------------------------------------------------------
    @staticmethod
    def _bootstrap_variance(
        max_conf_pool: np.ndarray,
        n_samples: int,
        rng: np.random.RandomState,
        k: int = BOOTSTRAP_K,
    ) -> np.ndarray:
        """Generate n_samples variance values via bootstrap sampling.

        For each of the n_samples rows, draw k values from max_conf_pool
        (with replacement) and compute their variance.

        Parameters
        ----------
        max_conf_pool : np.ndarray, shape (M,)
            Pool of CLEAN max-confidence values from a model's outputs.
        n_samples : int
            Number of variance values to produce.
        rng : np.random.RandomState
            Random state for reproducibility.
        k : int
            Bootstrap sample size per variance computation.

        Returns
        -------
        variances : np.ndarray, shape (n_samples,)
        """
        indices = rng.randint(0, len(max_conf_pool), size=(n_samples, k))
        samples = max_conf_pool[indices]
        return np.var(samples, axis=1)

    # ------------------------------------------------------------------
    # Override: execute
    # ------------------------------------------------------------------
    def execute(self) -> "FluctuationVarianceMIA":
        """Full pipeline."""
        self.execute_shadow_phase()
        print("[FluctuationVarianceMIA] Training attack model …", flush=True)
        self._train_attack_model()
        self._is_trained = True
        print("[FluctuationVarianceMIA] Pipeline complete ✓", flush=True)
        return self

    def execute_shadow_phase(self) -> "FluctuationVarianceMIA":
        """Steps 1 + 2: train shadow models and build fluctuation+variance dataset."""
        print(
            "[FluctuationVarianceMIA] Step 1/2: Training shadow models (adversarial) …",
            flush=True,
        )
        self._train_shadow_models()
        print(
            "[FluctuationVarianceMIA] Step 2/2: Preparing fluctuation+variance dataset …",
            flush=True,
        )
        self._prepare_attack_dataset()
        print("[FluctuationVarianceMIA] Shadow phase complete ✓", flush=True)
        return self

    # ------------------------------------------------------------------
    # Override: train shadow models + cache victim CLEAN max-conf pool
    # ------------------------------------------------------------------
    def _train_shadow_models(self):
        """Train adversarial shadow models and cache victim's CLEAN max-conf pool.

        During shadow model training, the victim API is queried on each shadow
        model's training data.  We reuse the CLEAN victim API responses to
        build the victim's max-confidence pool for the variance feature.
        """
        rng = np.random.RandomState(self.random_state)
        n   = len(self.unlabelled_data)

        victim_max_scores = []  # accumulate CLEAN max-confidence from victim queries

        for i in range(self.num_shadow_models):
            train_indices = rng.choice(
                n, size=self.shadow_model_dataset_size, replace=False
            )
            train_paths = self.unlabelled_data[train_indices]

            # Query victim API (CLEAN) → pseudo-labels + max-confidence cache
            victim_scores = self.victim_model_api.predict(train_paths)
            pseudo_labels = (victim_scores > 0.5).astype(np.float32)

            # Cache CLEAN max-confidence scores for variance feature
            victim_max = np.max(victim_scores, axis=1)
            victim_max_scores.append(victim_max)

            # Pick shadow model parameters
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

            self._shadow_pseudo_labels[i] = {
                "train_paths":   train_paths,
                "pseudo_labels": pseudo_labels,
            }

            if self._reference_shadow is None:
                self._reference_shadow = shadow_model

            arch_name = getattr(shadow_model, "architecture", param.model_type)
            print(
                f"  Shadow model {i + 1}/{self.num_shadow_models} trained "
                f"({arch_name}).",
                flush=True,
            )
            sys.stdout.flush()

        # Build victim's CLEAN max-confidence pool from all API queries
        self._victim_max_conf_pool = np.concatenate(victim_max_scores)
        print(
            f"  Victim CLEAN max-confidence pool: "
            f"{len(self._victim_max_conf_pool)} scores "
            f"(mean={self._victim_max_conf_pool.mean():.4f}, "
            f"var={np.var(self._victim_max_conf_pool):.6f})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Override: build fluctuation + variance attack dataset
    # ------------------------------------------------------------------
    def _prepare_attack_dataset(self):
        """Build attack dataset with fluctuation features + variance_of_max.

        Same as FluctuationMIA._prepare_attack_dataset() but adds one extra
        column: variance_of_max, bootstrapped from each shadow model's CLEAN
        max-confidence pool.

        Columns: fluct_0…fluct_14, max_confidence, max_confidence_class,
                 variance_of_max, is_part_of_dataset
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

            cached = self._shadow_pseudo_labels[idx]

            # ── Positive samples (member = 1) ─────────────────────────────
            pos_pool = list(train_indices)
            k_pos    = min(num_per_model, len(pos_pool))
            pos_idx  = rng.choice(pos_pool, size=k_pos, replace=False)
            pos_data = self.unlabelled_data[pos_idx]

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

            # ── Negative samples (non-member = 0) ─────────────────────────
            neg_pool = list(non_train_idx)
            k_neg    = min(num_per_model, len(neg_pool))
            neg_idx  = rng.choice(neg_pool, size=k_neg, replace=False)
            neg_data = self.unlabelled_data[neg_idx]

            neg_clean_probs = model.predict_proba(neg_data)
            neg_pseudo      = (neg_clean_probs > 0.5).astype(np.float32)

            clean_neg, adv_neg = model.predict_proba_adversarial(
                neg_data, neg_pseudo, self.epsilon
            )
            fluct_neg = np.abs(clean_neg - adv_neg)

            # Build shadow model's CLEAN max-confidence pool for variance
            all_clean   = np.vstack([clean_pos, clean_neg])
            shadow_max_pool = np.max(all_clean, axis=1)

            # Bootstrap variance for positive samples
            pos_var = self._bootstrap_variance(shadow_max_pool, k_pos, rng)
            for j in range(k_pos):
                row = {
                    f"fluct_{c}": fluct_pos[j, c]
                    for c in range(self.num_classes)
                }
                row["max_confidence"]       = float(np.max(clean_pos[j]))
                row["max_confidence_class"] = int(np.argmax(clean_pos[j]))
                row["variance_of_max"]      = pos_var[j]
                row["is_part_of_dataset"]   = 1
                rows.append(row)

            # Bootstrap variance for negative samples
            neg_var = self._bootstrap_variance(shadow_max_pool, k_neg, rng)
            for j in range(k_neg):
                row = {
                    f"fluct_{c}": fluct_neg[j, c]
                    for c in range(self.num_classes)
                }
                row["max_confidence"]       = float(np.max(clean_neg[j]))
                row["max_confidence_class"] = int(np.argmax(clean_neg[j]))
                row["variance_of_max"]      = neg_var[j]
                row["is_part_of_dataset"]   = 0
                rows.append(row)

            print(
                f"  Shadow model {idx + 1}: pool_size={len(shadow_max_pool)}, "
                f"mean_var={np.mean(np.concatenate([pos_var, neg_var])):.6f}",
                flush=True,
            )

        self.attack_dataset = pd.DataFrame(rows)
        n_pos = int(self.attack_dataset["is_part_of_dataset"].sum())
        n_neg = len(self.attack_dataset) - n_pos
        print(
            f"  Attack dataset built: {len(self.attack_dataset)} samples "
            f"({n_pos} positive, {n_neg} negative), "
            f"{self.num_classes + 3} features "
            f"(fluctuation + max_conf + max_class + variance).",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Override: train attack model on fluctuation + variance features
    # ------------------------------------------------------------------
    def _train_attack_model(self):
        feature_cols = (
            [f"fluct_{c}" for c in range(self.num_classes)]
            + ["max_confidence", "max_confidence_class", "variance_of_max"]
        )

        X = self.attack_dataset[feature_cols].values
        y = self.attack_dataset["is_part_of_dataset"].values

        self.attack_model = self.attack_model_parameters.build(
            random_state=self.random_state
        )
        self.attack_model.fit(X, y)
        print(
            f"  Attack model trained ({self.attack_model_parameters.model_type}) "
            f"on {X.shape[1]} features (fluctuation + variance).",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Override: evaluate_attack_model with variance features
    # ------------------------------------------------------------------
    def evaluate_attack_model(
        self,
        attack_params: "ModelParameters",
        member_data: np.ndarray,
        non_member_data: np.ndarray,
    ) -> dict:
        """Train ONE attack model on the fluctuation+variance dataset and evaluate."""
        if self.attack_dataset is None:
            raise RuntimeError(
                "Call execute_shadow_phase() before evaluate_attack_model()."
            )

        feature_cols = (
            [f"fluct_{c}" for c in range(self.num_classes)]
            + ["max_confidence", "max_confidence_class", "variance_of_max"]
        )
        X = self.attack_dataset[feature_cols].values
        y = self.attack_dataset["is_part_of_dataset"].values

        model = attack_params.build(random_state=self.random_state)
        model.fit(X, y)
        print(
            f"  Attack model trained: {attack_params.model_type} "
            f"({X.shape[1]} features)",
            flush=True,
        )

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

        self.attack_model            = prev_model
        self.attack_model_parameters = prev_params
        self._is_trained             = prev_trained

        return metrics

    # ------------------------------------------------------------------
    # Override: attack with variance feature
    # ------------------------------------------------------------------
    def attack(self, data: np.ndarray, return_confidence: bool = False):
        """Predict membership using fluctuation + variance features.

        Same as FluctuationMIA.attack() but adds the variance_of_max
        feature, bootstrapped from the victim's CLEAN max-confidence pool.

        Parameters
        ----------
        data : np.ndarray of str, shape (N,)
            Image file paths.
        return_confidence : bool
            If True, also return attack model probability estimates.
        """
        if not self._is_trained:
            raise RuntimeError("Call .execute() before .attack().")
        if self._reference_shadow is None:
            raise RuntimeError("No reference shadow model. Call execute_shadow_phase().")
        if self._victim_max_conf_pool is None:
            raise RuntimeError(
                "Victim CLEAN max-confidence pool not available. "
                "Call execute_shadow_phase() via FluctuationVarianceMIA."
            )

        import torch
        from mia_fluctuation import _PathImageDataset
        from torch.utils.data import DataLoader

        victim_model     = self.victim_model_api.model
        victim_device    = self.victim_model_api.device
        victim_transform = self.victim_model_api.transform

        shadow       = self._reference_shadow
        shadow_device = shadow.device

        dataset = _PathImageDataset(data, victim_transform)
        loader  = DataLoader(
            dataset,
            batch_size=self.victim_model_api.batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_fluct     = []
        all_max_conf  = []
        all_max_class = []

        for images in loader:
            images_v = images.to(victim_device)

            # Victim clean forward
            victim_model.eval()
            with torch.no_grad():
                clean_logits = victim_model(images_v)
                clean_probs  = torch.sigmoid(clean_logits)

            # Generate adversarial via shadow model (transfer attack)
            pseudo_labels = (clean_probs > 0.5).float().detach()
            images_s = images.to(shadow_device)
            pseudo_s = pseudo_labels.to(shadow_device)
            adv_images_s = shadow.generate_adversarial_tensors(
                images_s, pseudo_s, self.epsilon
            )
            adv_images_v = adv_images_s.to(victim_device)

            # Victim adversarial forward
            victim_model.eval()
            with torch.no_grad():
                adv_logits = victim_model(adv_images_v)
                adv_probs  = torch.sigmoid(adv_logits)

            fluct = torch.abs(clean_probs - adv_probs).cpu().numpy()
            all_fluct.append(fluct)

            # max_confidence from CLEAN only
            clean_np = clean_probs.cpu().numpy()
            all_max_conf.append(np.max(clean_np, axis=1))
            all_max_class.append(np.argmax(clean_np, axis=1).astype(float))

        fluctuations  = np.vstack(all_fluct)
        max_conf_arr  = np.concatenate(all_max_conf)
        max_class_arr = np.concatenate(all_max_class)

        n = len(fluctuations)

        # Bootstrap variance from victim's CLEAN max-confidence pool
        rng = np.random.RandomState(self.random_state + 99)
        variance = self._bootstrap_variance(
            self._victim_max_conf_pool, n, rng
        ).reshape(-1, 1)

        features = np.hstack([
            fluctuations,
            max_conf_arr.reshape(-1, 1),
            max_class_arr.reshape(-1, 1),
            variance,
        ])

        predictions = self.attack_model.predict(features)

        if return_confidence:
            if hasattr(self.attack_model, "predict_proba"):
                proba = self.attack_model.predict_proba(features)
            else:
                proba = self.attack_model.decision_function(features)
            return predictions, proba

        return predictions

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
        pool_info = (
            f", victim_pool_size={len(self._victim_max_conf_pool)}"
            if self._victim_max_conf_pool is not None else ""
        )
        return (
            f"FluctuationVarianceMIA(num_classes={self.num_classes}, "
            f"num_shadow_models={self.num_shadow_models}, "
            f"epsilon={self.epsilon}, "
            f"shadow={self.shadow_model_parameters!r}, "
            f"attack={self.attack_model_parameters!r}, "
            f"features=fluctuation+variance, "
            f"status={status}{pool_info})"
        )
