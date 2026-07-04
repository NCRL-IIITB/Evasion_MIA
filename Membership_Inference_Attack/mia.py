"""
mia.py
======
Membership Inference Attack (MIA) Implementation

Based on Shokri et al. "Membership Inference Attacks Against Machine Learning Models"

This module implements the MIA class which orchestrates the full attack pipeline:
  1. Train shadow models to mimic the victim model's behavior
  2. Build an attack dataset from shadow model confidence scores
  3. Train an attack model to distinguish members from non-members
  4. Attack: given new data, predict whether it was in the victim's training set

Shadow models are PyTorch CNNs (via PyTorchShadowModel) that take raw image
paths as input.  The victim model API also takes image paths and returns
(N, num_classes) sigmoid scores.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

from shadow_models import PyTorchShadowModel

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Placeholder API class
# ---------------------------------------------------------------------------
class API:
    """Placeholder for the victim model API.

    The real implementation (VictimAPI in api.py) must expose:
        predict(X: np.ndarray) -> np.ndarray
            X   : array of image file path strings, shape (N,)
            Returns: array of sigmoid probabilities, shape (N, num_classes)
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "API.predict() is not implemented. "
            "Replace this placeholder with your VictimAPI."
        )


# ---------------------------------------------------------------------------
# Supported model registry
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    "random_forest":      RandomForestClassifier,
    "gradient_boosting":  GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "mlp":                MLPClassifier,
    "svm":                SVC,
    "pytorch_cnn":        PyTorchShadowModel,
}


# ---------------------------------------------------------------------------
# ModelParameters
# ---------------------------------------------------------------------------
class ModelParameters:
    """Bundles a model type string with arbitrary hyper-parameters.

    Parameters
    ----------
    model_type : str
        One of the keys in _MODEL_REGISTRY.
    **kwargs
        Forwarded to the model constructor.

    Examples
    --------
    >>> shadow = ModelParameters("pytorch_cnn", architecture="resnet18",
    ...                          num_classes=15, epochs=15, batch_size=32, lr=1e-3)
    >>> attack  = ModelParameters("gradient_boosting", n_estimators=100,
    ...                           learning_rate=0.1)
    """

    def __init__(self, model_type: str = "random_forest", **kwargs):
        if model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Choose from: {list(_MODEL_REGISTRY.keys())}"
            )
        self.model_type = model_type
        self.params     = dict(kwargs)

    def build(self, random_state: int = 42):
        """Construct and return a fresh (unfitted) model instance."""
        cls          = _MODEL_REGISTRY[self.model_type]
        final_params = dict(self.params)

        # Inject sensible defaults for sklearn models
        if self.model_type == "svm" and "probability" not in final_params:
            final_params["probability"] = True
        if self.model_type == "logistic_regression" and "max_iter" not in final_params:
            final_params["max_iter"] = 1000
        if self.model_type == "mlp" and "max_iter" not in final_params:
            final_params["max_iter"] = 500

        # Only inject random_state for sklearn models; PyTorchShadowModel handles its own
        if self.model_type != "pytorch_cnn":
            final_params.setdefault("random_state", random_state)
        else:
            final_params.setdefault("random_state", random_state)

        return cls(**final_params)

    def __repr__(self):
        param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return (
            f"ModelParameters('{self.model_type}', {param_str})"
            if param_str
            else f"ModelParameters('{self.model_type}')"
        )


# ---------------------------------------------------------------------------
# MIA
# ---------------------------------------------------------------------------
class MIA:
    """Membership Inference Attack.

    Trains CNN shadow models to mimic a victim model, then trains an attack
    model (GradientBoosting by default) to predict whether a given image
    was part of the victim model's training set.

    Parameters
    ----------
    victim_model_api : API
        Object whose ``predict(X)`` takes image paths and returns
        (N, num_classes) sigmoid scores.
    unlabelled_data : np.ndarray
        Array of image file paths (strings), shape (N,).
        This is the attacker's shuffled pool of all available images.
    num_classes : int
        Number of output classes of the victim model (15 for NIH).
    num_shadow_models : int
        How many shadow models to train.
    shadow_model_dataset_size : int or None
        Images sampled from ``unlabelled_data`` per shadow model.
        Defaults to ``len(unlabelled_data) // 2``.
    attack_model_dataset_size : int or None
        Total attack dataset size across all shadow models.
        If None, uses ``shadow_model_dataset_size // 2`` per model.
    shadow_model_parameters : ModelParameters or list[ModelParameters] or None
        Shadow model type/params. List entries cycle (e.g. 3 archs for 8 models).
    attack_model_parameters : ModelParameters or None
        Attack model type/params.
    """

    def __init__(
        self,
        victim_model_api: API,
        unlabelled_data: np.ndarray,
        num_classes: int,
        num_shadow_models: int = 8,
        shadow_model_dataset_size: int | None = None,
        attack_model_dataset_size: int | None = None,
        shadow_model_parameters: "ModelParameters | list | None" = None,
        attack_model_parameters: "ModelParameters | None" = None,
    ):
        self.victim_model_api = victim_model_api
        self.unlabelled_data  = np.asarray(unlabelled_data)
        self.num_classes      = num_classes
        self.num_shadow_models = num_shadow_models

        self.shadow_model_dataset_size = (
            shadow_model_dataset_size
            if shadow_model_dataset_size is not None
            else len(self.unlabelled_data) // 2
        )
        self.attack_model_dataset_size = attack_model_dataset_size

        self.shadow_model_parameters = (
            shadow_model_parameters
            if shadow_model_parameters is not None
            else ModelParameters("pytorch_cnn", num_classes=num_classes)
        )
        self.attack_model_parameters = (
            attack_model_parameters
            if attack_model_parameters is not None
            else ModelParameters("gradient_boosting")
        )

        self.random_state: int = 42

        # Internal state
        self.shadow_models: dict  = {}     # idx → {"model": ..., "train_indices": set}
        self.attack_model         = None
        self.attack_dataset: pd.DataFrame | None = None
        self._is_trained: bool    = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self) -> "MIA":
        """Run the full MIA pipeline (Steps 1 → 3). Kept for backwards compat."""
        self.execute_shadow_phase()
        print("[MIA] Step 3/3: Training attack model …", flush=True)
        self._train_attack_model()
        self._is_trained = True
        print("[MIA] Pipeline complete ✓", flush=True)
        return self

    def execute_shadow_phase(self) -> "MIA":
        """Steps 1 + 2 only: train shadow models and build attack dataset.

        Call this ONCE, then call evaluate_attack_model() for each attack
        model variant you want to compare — no repeated shadow model training.
        """
        print("[MIA] Step 1/2: Training shadow models …", flush=True)
        self._train_shadow_models()
        print("[MIA] Step 2/2: Preparing attack dataset …", flush=True)
        self._prepare_attack_dataset()
        print("[MIA] Shadow phase complete ✓", flush=True)
        return self

    def evaluate_attack_model(
        self,
        attack_params: "ModelParameters",
        member_data: np.ndarray,
        non_member_data: np.ndarray,
    ) -> dict:
        """Train ONE attack model on the pre-built attack dataset and evaluate it.

        Shadow models must already be trained (call execute_shadow_phase first).

        Parameters
        ----------
        attack_params : ModelParameters
            The attack model to train (e.g. GradientBoosting, RandomForest, MLP).
        member_data, non_member_data : np.ndarray of str
            Image paths with known ground-truth membership for evaluation.

        Returns
        -------
        dict with keys: attack_model, accuracy, precision, recall, f1
        """
        if self.attack_dataset is None:
            raise RuntimeError("Call execute_shadow_phase() before evaluate_attack_model().")

        feature_cols = [f"class_{c}" for c in range(self.num_classes)] + ["max_confidence", "max_confidence_class"]
        X = self.attack_dataset[feature_cols].values
        y = self.attack_dataset["is_part_of_dataset"].values

        model = attack_params.build(random_state=self.random_state)
        model.fit(X, y)
        print(f"  Attack model trained: {attack_params.model_type}", flush=True)

        # Temporarily swap in this attack model for the evaluate() call
        prev_model        = self.attack_model
        prev_params       = self.attack_model_parameters
        prev_trained      = self._is_trained
        self.attack_model            = model
        self.attack_model_parameters = attack_params
        self._is_trained             = True

        X_eval = np.concatenate([member_data, non_member_data])
        y_true = np.concatenate([
            np.ones(len(member_data)),
            np.zeros(len(non_member_data)),
        ])
        y_pred = self.attack(X_eval)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

    def attack(self, data: np.ndarray, return_confidence: bool = False):
        """Predict membership of *data* in the victim model's training set.

        Parameters
        ----------
        data : np.ndarray
            Image file paths (strings), shape (N,).
        return_confidence : bool
            If True, also return attack model probability estimates.

        Returns
        -------
        predictions : np.ndarray of {0, 1}
        confidences : np.ndarray (only if return_confidence=True)
        """
        if not self._is_trained:
            raise RuntimeError("Call .execute() before .attack().")

        # Query victim → confidence scores (N, num_classes)
        confidence_scores = self.victim_model_api.predict(data)
        confidence_scores = np.asarray(confidence_scores)
        if confidence_scores.ndim == 1:
            confidence_scores = confidence_scores.reshape(1, -1)

        # Add max_confidence and max_confidence_class features
        max_conf  = np.max(confidence_scores, axis=1, keepdims=True)   # (N, 1)
        max_class = np.argmax(confidence_scores, axis=1).reshape(-1, 1).astype(float)  # (N, 1)
        features  = np.hstack([confidence_scores, max_conf, max_class])  # (N, num_classes+2)

        predictions = self.attack_model.predict(features)

        if return_confidence:
            if hasattr(self.attack_model, "predict_proba"):
                proba = self.attack_model.predict_proba(features)
            else:
                proba = self.attack_model.decision_function(features)
            return predictions, proba

        return predictions

    def evaluate(
        self,
        member_data: np.ndarray,
        non_member_data: np.ndarray,
    ) -> dict:
        """Evaluate the attack on known member / non-member image paths.

        Returns
        -------
        dict with keys: accuracy, precision, recall, f1
        """
        X = np.concatenate([member_data, non_member_data])
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
    # Step 1: Train shadow models
    # ------------------------------------------------------------------
    def _train_shadow_models(self):
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
                param = self.shadow_model_parameters[i % len(self.shadow_model_parameters)]
            else:
                param = self.shadow_model_parameters

            shadow_model = param.build(random_state=self.random_state + i)
            shadow_model.fit(train_paths, pseudo_labels)

            self.shadow_models[i] = {
                "model":         shadow_model,
                "train_indices": set(train_indices.tolist()),
            }

            arch_name = getattr(shadow_model, "architecture", param.model_type)
            print(
                f"  Shadow model {i + 1}/{self.num_shadow_models} trained "
                f"({arch_name}).",
                flush=True,
            )
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Step 2: Build attack dataset
    # ------------------------------------------------------------------
    def _prepare_attack_dataset(self):
        """Build self.attack_dataset DataFrame with columns:
            class_0, …, class_{K-1}, is_part_of_dataset
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

            def _padded_proba(data):
                """Return (N, num_classes) proba array; handles partial class sets."""
                raw = model.predict_proba(data)
                # PyTorchShadowModel always returns (N, num_classes) directly
                if (
                    hasattr(model, "architecture")
                    or (isinstance(raw, np.ndarray) and raw.ndim == 2
                        and raw.shape[1] == self.num_classes)
                ):
                    return raw
                # Sklearn models may return fewer columns if some classes missing
                if raw.shape[1] == self.num_classes:
                    return raw
                full = np.zeros((raw.shape[0], self.num_classes))
                for col_i, cls_label in enumerate(model.classes_):
                    if int(cls_label) < self.num_classes:
                        full[:, int(cls_label)] = raw[:, col_i]
                return full

            # Positive samples (in shadow training set → member=1)
            pos_pool = list(train_indices)
            k_pos    = min(num_per_model, len(pos_pool))
            pos_idx  = rng.choice(pos_pool, size=k_pos, replace=False)
            pos_conf = _padded_proba(self.unlabelled_data[pos_idx])

            for j in range(k_pos):
                row = {f"class_{c}": pos_conf[j, c] for c in range(self.num_classes)}
                row["max_confidence"]       = float(np.max(pos_conf[j]))
                row["max_confidence_class"] = int(np.argmax(pos_conf[j]))
                row["is_part_of_dataset"]   = 1
                rows.append(row)

            # Negative samples (not in shadow training set → non-member=0)
            neg_pool = list(non_train_idx)
            k_neg    = min(num_per_model, len(neg_pool))
            neg_idx  = rng.choice(neg_pool, size=k_neg, replace=False)
            neg_conf = _padded_proba(self.unlabelled_data[neg_idx])

            for j in range(k_neg):
                row = {f"class_{c}": neg_conf[j, c] for c in range(self.num_classes)}
                row["max_confidence"]       = float(np.max(neg_conf[j]))
                row["max_confidence_class"] = int(np.argmax(neg_conf[j]))
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
    # Step 3: Train attack model
    # ------------------------------------------------------------------
    def _train_attack_model(self):
        feature_cols = [f"class_{c}" for c in range(self.num_classes)] + ["max_confidence", "max_confidence_class"]
        X = self.attack_dataset[feature_cols].values
        y = self.attack_dataset["is_part_of_dataset"].values

        self.attack_model = self.attack_model_parameters.build(
            random_state=self.random_state
        )
        self.attack_model.fit(X, y)
        print(
            f"  Attack model trained ({self.attack_model_parameters.model_type}).",
            flush=True,
        )

    # ------------------------------------------------------------------
    def __repr__(self):
        status = "trained" if self._is_trained else "not trained"
        return (
            f"MIA(num_classes={self.num_classes}, "
            f"num_shadow_models={self.num_shadow_models}, "
            f"shadow={self.shadow_model_parameters!r}, "
            f"attack={self.attack_model_parameters!r}, "
            f"status={status})"
        )
