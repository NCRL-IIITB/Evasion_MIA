"""
mia_variance.py
===============
Variance-Enhanced Membership Inference Attack (VarianceMIA)

Extends the standard MIA by adding one extra feature per sample:
    variance_of_max — a model-level statistic capturing the variance of the
    max sigmoid score across multiple images queried to a model.

Philosophy
----------
  The variance of max(confidence) across K data points characterises a model's
  overall confidence behaviour:
    • A model that is uniformly confident on all inputs → low variance
    • A model that is confident on some and unsure on others → high variance

  This model-level fingerprint differs between model architectures, training
  procedures, and data distributions, making it a discriminative signal for MIA.

How it works
------------
  TRAINING (attack dataset construction, per shadow model i):
    1. Query shadow model i on its member and non-member data → get confidence
       scores for each image.
    2. Collect max(score) from ALL those queries → the shadow model's
       "max-confidence pool".
    3. For each data point added to the attack dataset, bootstrap sample K
       values from this pool and compute variance of the sample → 16th feature.
       This gives slightly different variance values per data point (natural
       variation from bootstrap, no artificial noise needed).

  EVALUATION (at attack time against the victim model):
    1. The attacker already queries the victim model on pool data (to create
       pseudo-labels for shadow model training). Reuse those scores to build the
       victim's max-confidence pool.
    2. For each eval data point: bootstrap sample from the victim pool → compute
       variance → 16th feature.

Attack dataset columns
----------------------
    class_0, …, class_14,   ← shadow model sigmoid scores (from shadow model)
    variance_of_max,         ← bootstrapped var(max(shadow_i(x_k))) across K samples
    is_part_of_dataset       ← 1 = member, 0 = non-member
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mia import MIA, ModelParameters, API

# Bootstrap sample size for computing per-row variance
BOOTSTRAP_K = 100


class VarianceMIA(MIA):
    """Variance-Enhanced Membership Inference Attack.

    Inherits all parameters and constructor arguments from MIA.
    Overrides:
      - ``_prepare_attack_dataset()`` → adds bootstrapped variance_of_max
      - ``_train_attack_model()``     → trains on 16-dim features
      - ``attack()``                  → uses victim's variance at inference
      - ``evaluate()``                → uses extended attack pipeline
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache for the victim model's max-confidence pool
        # Populated during shadow model training (from victim API queries)
        self._victim_max_conf_pool = None

    # ------------------------------------------------------------------
    # Helper: padded predict_proba from one shadow model
    # ------------------------------------------------------------------
    def _padded_predict_proba(self, model, data: np.ndarray) -> np.ndarray:
        """Return (N, num_classes) sigmoid proba; handles partial class sets."""
        raw = model.predict_proba(data)

        if (
            hasattr(model, "architecture")
            or (isinstance(raw, np.ndarray) and raw.ndim == 2
                and raw.shape[1] == self.num_classes)
        ):
            return raw

        if raw.shape[1] == self.num_classes:
            return raw

        full = np.zeros((raw.shape[0], self.num_classes))
        for col_i, cls_label in enumerate(model.classes_):
            if int(cls_label) < self.num_classes:
                full[:, int(cls_label)] = raw[:, col_i]
        return full

    # ------------------------------------------------------------------
    # Helper: bootstrap variance from a max-confidence pool
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
            Pool of max-confidence values from a model's outputs.
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
        # Draw all samples at once: shape (n_samples, k)
        indices = rng.randint(0, len(max_conf_pool), size=(n_samples, k))
        samples = max_conf_pool[indices]  # (n_samples, k)
        return np.var(samples, axis=1)    # (n_samples,)

    # ------------------------------------------------------------------
    # Override: Train shadow models AND cache victim max-confidence pool
    # ------------------------------------------------------------------
    def _train_shadow_models(self):
        """Train shadow models and also cache the victim's max-confidence pool.

        During shadow model training, the victim API is queried on each shadow
        model's training data to produce pseudo-labels. We reuse these victim
        API responses to build the victim's max-confidence pool.
        """
        rng = np.random.RandomState(self.random_state)
        n   = len(self.unlabelled_data)

        victim_max_scores = []  # accumulate max-confidence from victim queries

        for i in range(self.num_shadow_models):
            # Sample training paths for this shadow model
            train_indices = rng.choice(
                n, size=self.shadow_model_dataset_size, replace=False
            )
            train_paths = self.unlabelled_data[train_indices]

            # Query victim API → pseudo multi-hot labels (threshold at 0.5)
            victim_scores = self.victim_model_api.predict(train_paths)
            pseudo_labels = (victim_scores > 0.5).astype(np.float32)

            # Cache victim max-confidence scores
            victim_max = np.max(victim_scores, axis=1)  # (shadow_dataset_size,)
            victim_max_scores.append(victim_max)

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

        # Build victim's max-confidence pool from all API queries
        self._victim_max_conf_pool = np.concatenate(victim_max_scores)
        print(
            f"  Victim max-confidence pool: {len(self._victim_max_conf_pool)} "
            f"scores (mean={self._victim_max_conf_pool.mean():.4f}, "
            f"var={np.var(self._victim_max_conf_pool):.6f})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Step 2: Build attack dataset (with bootstrapped variance feature)
    # ------------------------------------------------------------------
    def _prepare_attack_dataset(self):
        """Build self.attack_dataset with 16 features (confidence + variance).

        For each shadow model, query it on member and non-member data to get
        confidence scores. Then compute the shadow model's max-confidence pool
        from those queries, and bootstrap-sample variance values for each row.
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

            # --- Positive samples (member=1) ---
            pos_pool = list(train_indices)
            k_pos    = min(num_per_model, len(pos_pool))
            pos_idx  = rng.choice(pos_pool, size=k_pos, replace=False)
            pos_data = self.unlabelled_data[pos_idx]
            pos_conf = self._padded_predict_proba(model, pos_data)   # (k, C)

            # --- Negative samples (non-member=0) ---
            neg_pool = list(non_train_idx)
            k_neg    = min(num_per_model, len(neg_pool))
            neg_idx  = rng.choice(neg_pool, size=k_neg, replace=False)
            neg_data = self.unlabelled_data[neg_idx]
            neg_conf = self._padded_predict_proba(model, neg_data)   # (k, C)

            # Build this shadow model's max-confidence pool from ALL queries
            all_conf  = np.vstack([pos_conf, neg_conf])              # (k_pos+k_neg, C)
            shadow_max_pool = np.max(all_conf, axis=1)               # (k_pos+k_neg,)

            # Bootstrap variance for positive samples
            pos_var = self._bootstrap_variance(shadow_max_pool, k_pos, rng)
            for j in range(k_pos):
                row = {f"class_{c}": pos_conf[j, c] for c in range(self.num_classes)}
                row["max_confidence"]       = float(np.max(pos_conf[j]))
                row["max_confidence_class"] = int(np.argmax(pos_conf[j]))
                row["variance_of_max"]      = pos_var[j]
                row["is_part_of_dataset"]   = 1
                rows.append(row)

            # Bootstrap variance for negative samples
            neg_var = self._bootstrap_variance(shadow_max_pool, k_neg, rng)
            for j in range(k_neg):
                row = {f"class_{c}": neg_conf[j, c] for c in range(self.num_classes)}
                row["max_confidence"]       = float(np.max(neg_conf[j]))
                row["max_confidence_class"] = int(np.argmax(neg_conf[j]))
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
            f"{self.num_classes + 3} features (confidence + max_conf + max_class + variance).",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Step 3: Train attack model on 16-dim features
    # ------------------------------------------------------------------
    def _train_attack_model(self):
        conf_cols    = [f"class_{c}" for c in range(self.num_classes)]
        feature_cols = conf_cols + ["max_confidence", "max_confidence_class", "variance_of_max"]

        X = self.attack_dataset[feature_cols].values
        y = self.attack_dataset["is_part_of_dataset"].values

        self.attack_model = self.attack_model_parameters.build(
            random_state=self.random_state
        )
        self.attack_model.fit(X, y)
        print(
            f"  Attack model trained ({self.attack_model_parameters.model_type}) "
            f"on {X.shape[1]} features.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # execute() override: use shadow phase split
    # ------------------------------------------------------------------
    def execute(self) -> "VarianceMIA":
        """Full pipeline. Calls execute_shadow_phase then trains one attack model."""
        self.execute_shadow_phase()
        print("[VarianceMIA] Training attack model …", flush=True)
        self._train_attack_model()
        self._is_trained = True
        print("[VarianceMIA] Pipeline complete ✓", flush=True)
        return self

    def execute_shadow_phase(self) -> "VarianceMIA":
        """Steps 1 + 2 only: train shadow models and build attack dataset.

        Call ONCE, then call evaluate_attack_model() for each attack model
        variant without re-training shadows.
        """
        print("[VarianceMIA] Step 1/2: Training shadow models …", flush=True)
        self._train_shadow_models()
        print("[VarianceMIA] Step 2/2: Preparing attack dataset (conf + variance) …", flush=True)
        self._prepare_attack_dataset()
        print("[VarianceMIA] Shadow phase complete ✓", flush=True)
        return self

    def evaluate_attack_model(
        self,
        attack_params: "ModelParameters",
        member_data: np.ndarray,
        non_member_data: np.ndarray,
    ) -> dict:
        """Train ONE attack model on the pre-built 16-dim dataset and evaluate it."""
        if self.attack_dataset is None:
            raise RuntimeError("Call execute_shadow_phase() before evaluate_attack_model().")

        conf_cols    = [f"class_{c}" for c in range(self.num_classes)]
        feature_cols = conf_cols + ["max_confidence", "max_confidence_class", "variance_of_max"]
        X = self.attack_dataset[feature_cols].values
        y = self.attack_dataset["is_part_of_dataset"].values

        model = attack_params.build(random_state=self.random_state)
        model.fit(X, y)
        print(f"  Attack model trained: {attack_params.model_type} (18 features)", flush=True)

        # Temporarily swap in this model for the attack() call
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

    # ------------------------------------------------------------------
    # Attack: victim confidence + victim's bootstrapped variance → membership
    # ------------------------------------------------------------------
    def attack(self, data: np.ndarray, return_confidence: bool = False):
        """Predict membership using 16-dim features.

        The 16th feature (variance_of_max) is bootstrapped from the victim
        model's max-confidence pool, which was built during shadow model
        training from the victim API queries.

        Parameters
        ----------
        data : np.ndarray of str, shape (N,)
            Image file paths.
        return_confidence : bool
            If True, also return attack model probabilities.

        Returns
        -------
        predictions : np.ndarray of {0, 1}
        confidences : np.ndarray (only if return_confidence=True)
        """
        if not self._is_trained:
            raise RuntimeError("Call .execute() before .attack().")

        if self._victim_max_conf_pool is None:
            raise RuntimeError(
                "Victim max-confidence pool is not available. "
                "Make sure _train_shadow_models() was called via VarianceMIA "
                "(not the base MIA class)."
            )

        # Victim confidence scores (N, num_classes)
        victim_scores = np.asarray(self.victim_model_api.predict(data))
        if victim_scores.ndim == 1:
            victim_scores = victim_scores.reshape(1, -1)

        n = len(victim_scores)

        # Add max_confidence and max_confidence_class features
        max_conf  = np.max(victim_scores, axis=1, keepdims=True)   # (N, 1)
        max_class = np.argmax(victim_scores, axis=1).reshape(-1, 1).astype(float)  # (N, 1)

        # Bootstrap variance from the victim's max-confidence pool
        rng = np.random.RandomState(self.random_state + 99)
        variance = self._bootstrap_variance(
            self._victim_max_conf_pool, n, rng
        ).reshape(-1, 1)  # (N, 1)

        # Combine into 18-dim feature vector
        features = np.hstack([victim_scores, max_conf, max_class, variance])

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
        pool_info = (
            f", victim_pool_size={len(self._victim_max_conf_pool)}"
            if self._victim_max_conf_pool is not None else ""
        )
        return (
            f"VarianceMIA(num_classes={self.num_classes}, "
            f"num_shadow_models={self.num_shadow_models}, "
            f"shadow={self.shadow_model_parameters!r}, "
            f"attack={self.attack_model_parameters!r}, "
            f"features=confidence+variance_of_max, "
            f"status={status}{pool_info})"
        )
