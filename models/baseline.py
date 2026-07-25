"""Tier 1 -- the tabular Autoencoder baseline (Deliverable #2).

The brief allows a statistical profile, an Autoencoder or a One-Class SVM. We build exactly one:
a small symmetric Autoencoder over the **entity-relative numeric features**. The reasoning is
recorded in the plan, but in short: OCSVM does not scale to ~100k events and yields one opaque
distance; a bare statistical profile already lives in the feature layer and cannot express "this
*combination* of behaviours never occurs", which is the zero-day case. Reconstruction error is
multivariate and still explains itself per feature.

**What the network sees.** Only the scaled numeric block of the feature vector -- the first
``len(numeric_names)`` columns. Those are the entity-relative signals (likelihoods, z-scores,
novelty flags), already standardized by the pipeline's scaler, which is exactly what an
MSE autoencoder wants. The categorical *codes* that follow are arbitrary integer identities
(``geo_country`` 5 is not "more" than 2); reconstructing them under MSE would be meaningless, so
they are left to the LightGBM classifier in Phase 5, which treats them as categorical.

**Cold start is inherited, not special-cased.** The feature layer already blends a thin-history
entity toward its cohort prior before we ever see the vector, so a brand-new entity produces a
finite, comparable input and therefore a finite, comparable score. There is no separate cold-start
branch here.

**One score, one explanation, from one computation.** The per-sample anomaly score is the mean
squared reconstruction error; the per-feature squared error is the attribution the explainability
layer quotes. Score and explanation can never disagree because they are the same numbers.

**JSON, never pickle.** A pickled ``torch`` module is tied to the library version that wrote it and
would break the serving container after any dependency bump. The network here is a few thousand
floats, so weights persist as plain JSON lists alongside the normalizer -- readable, diffable and
version-independent, consistent with every other artifact in the system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from common.config import settings
from common.seed import set_global_seed
from features.featurize import FeatureVector

logger = logging.getLogger(__name__)

#: Artifact filename for the persisted baseline model.
BASELINE_FILE = "baseline_model.json"

# Network geometry. Small on purpose: the plan specifies input -> 32 -> 16 -> 32 -> input, and a
# larger net would memorize the ~0.9% contamination in the training split rather than learn the
# normal manifold.
DEFAULT_HIDDEN_DIM = 32
DEFAULT_BOTTLENECK_DIM = 16

# The reconstruction error at this quantile of held-out *normal-ish* residuals maps to a score of
# 0.5. Typical events therefore score near zero and only the upper tail climbs toward one, which is
# the behaviour risk fusion (Phase 5) expects from an unsupervised tier.
DEFAULT_NORMALIZER_QUANTILE = 0.95

#: Clip the standardized error before the logistic so one pathological event cannot saturate to an
#: exact 1.0 and collapse the ranking among the most anomalous events (which would cost PR-AUC).
_Z_CLIP = 12.0


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


class TabularAutoencoder(nn.Module):
    """A small symmetric autoencoder: ``input -> 32 -> 16 -> 32 -> input`` with ReLU.

    Deliberately shallow. The bottleneck forces the network to keep only the structure common to
    normal behaviour, so anything anomalous -- which by definition does not share that structure --
    reconstructs poorly and earns a high error.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - standard nn.Module hook
        return self.decoder(self.encoder(x))


# --------------------------------------------------------------------------- #
# Score normalization
# --------------------------------------------------------------------------- #


@dataclass
class ScoreNormalizer:
    """Maps a raw reconstruction error onto a bounded ``[0, 1]`` anomaly score.

    Fitted on **held-out** residuals -- errors the network never trained on -- so the reference
    distribution reflects unseen normal behaviour rather than the training fit. The map is a
    logistic on a robustly standardized error:

        ``score = sigmoid((error - center) / scale)``

    with ``center`` a high quantile of the held-out errors and ``scale`` a robust spread (scaled
    median absolute deviation). It is strictly monotonic in the raw error, so it preserves ranking
    -- and therefore PR-AUC and recall@budget -- exactly, while giving fusion a calibrated,
    comparable input on the same 0-1 footing as the other tiers.
    """

    center: float
    scale: float
    quantile: float = DEFAULT_NORMALIZER_QUANTILE

    @classmethod
    def fit(
        cls, errors: Sequence[float], quantile: float = DEFAULT_NORMALIZER_QUANTILE
    ) -> "ScoreNormalizer":
        values = np.asarray(list(errors), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return cls(center=0.0, scale=1.0, quantile=quantile)

        center = float(np.quantile(values, quantile))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))

        # 1.4826 rescales the MAD to a normal-consistent standard deviation. Fall back through the
        # ordinary std and finally a unit scale so a near-constant error column can never divide by
        # zero.
        scale = 1.4826 * mad
        if scale < 1e-9:
            scale = float(values.std())
        if scale < 1e-9:
            scale = 1.0
        return cls(center=center, scale=float(scale), quantile=quantile)

    def normalize(self, errors: Union[float, Sequence[float], np.ndarray]) -> np.ndarray:
        values = np.asarray(errors, dtype=float)
        z = (values - self.center) / (self.scale if self.scale else 1.0)
        z = np.clip(z, -_Z_CLIP, _Z_CLIP)
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self) -> Dict[str, Any]:
        return {"center": self.center, "scale": self.scale, "quantile": self.quantile}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScoreNormalizer":
        return cls(
            center=float(payload.get("center", 0.0)),
            scale=float(payload.get("scale", 1.0)),
            quantile=float(payload.get("quantile", DEFAULT_NORMALIZER_QUANTILE)),
        )


# --------------------------------------------------------------------------- #
# The model wrapper
# --------------------------------------------------------------------------- #


@dataclass
class BaselineTrainConfig:
    """Training hyper-parameters, kept together so a run is fully described by one object."""

    hidden_dim: int = DEFAULT_HIDDEN_DIM
    bottleneck_dim: int = DEFAULT_BOTTLENECK_DIM
    epochs: int = 120
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    holdout_fraction: float = 0.15
    normalizer_quantile: float = DEFAULT_NORMALIZER_QUANTILE
    seed: Optional[int] = None


class BaselineModel:
    """A trained autoencoder plus its score normalizer and the feature names it consumes.

    Constructed by :meth:`train` offline, or by :meth:`load` in the serving container. Serving
    never trains: it loads weights and scores.
    """

    def __init__(
        self,
        net: TabularAutoencoder,
        normalizer: ScoreNormalizer,
        feature_names: Sequence[str],
    ) -> None:
        self.net = net
        self.net.eval()
        self.normalizer = normalizer
        #: The numeric feature names, in order, that make up the network input.
        self.feature_names: List[str] = list(feature_names)
        if len(self.feature_names) != net.input_dim:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries but the network expects "
                f"{net.input_dim} inputs"
            )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def input_dim(self) -> int:
        return self.net.input_dim

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #

    def _slice(self, matrix: np.ndarray) -> np.ndarray:
        """Take the numeric block the network was trained on.

        Feature vectors carry ``[scaled numeric ..., categorical codes ...]``. The numeric block is
        exactly the leading ``input_dim`` columns, in the same order the model was fitted on. A
        matrix that already has ``input_dim`` columns is passed through untouched.
        """
        array = np.asarray(matrix, dtype=float)
        if array.ndim == 1:
            array = array[None, :]
        width = array.shape[1]
        if width == self.input_dim:
            return array
        if width > self.input_dim:
            return array[:, : self.input_dim]
        raise ValueError(
            f"expected at least {self.input_dim} feature columns, got {width}"
        )

    def _as_matrix(
        self, data: Union[FeatureVector, Sequence[FeatureVector], np.ndarray]
    ) -> Tuple[np.ndarray, bool]:
        """Return ``(matrix, was_single)`` for any accepted input shape."""
        if isinstance(data, FeatureVector):
            return self._slice(data.values), True
        if isinstance(data, np.ndarray):
            single = data.ndim == 1
            return self._slice(data), single
        # A sequence of FeatureVector.
        vectors = list(data)
        if not vectors:
            return np.zeros((0, self.input_dim), dtype=float), False
        stacked = np.vstack([vector.values for vector in vectors])
        return self._slice(stacked), False

    # ------------------------------------------------------------------ #
    # Core computations
    # ------------------------------------------------------------------ #

    def _reconstruct(self, matrix: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(matrix, dtype=np.float32))
            return self.net(tensor).cpu().numpy()

    def per_feature_errors(self, matrix: np.ndarray) -> np.ndarray:
        """Squared reconstruction error per feature, shape ``(n, input_dim)``."""
        data = self._slice(matrix)
        recon = self._reconstruct(data)
        return np.square(data - recon)

    def reconstruction_error(
        self, data: Union[FeatureVector, Sequence[FeatureVector], np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Mean squared reconstruction error per sample (the raw, unnormalized score)."""
        matrix, single = self._as_matrix(data)
        if matrix.shape[0] == 0:
            return np.zeros((0,), dtype=float)
        recon = self._reconstruct(matrix)
        errors = np.square(matrix - recon).mean(axis=1)
        return float(errors[0]) if single else errors

    def score_baseline(
        self, data: Union[FeatureVector, Sequence[FeatureVector], np.ndarray]
    ) -> Union[float, np.ndarray]:
        """The normalized ``baseline_score`` in ``[0, 1]``; higher means more anomalous."""
        matrix, single = self._as_matrix(data)
        if matrix.shape[0] == 0:
            return np.zeros((0,), dtype=float)
        recon = self._reconstruct(matrix)
        errors = np.square(matrix - recon).mean(axis=1)
        scores = self.normalizer.normalize(errors)
        return float(scores[0]) if single else scores

    def reconstruction_errors(self, vector: FeatureVector) -> Dict[str, float]:
        """Per-feature squared error for one event, keyed by feature name (for explanations)."""
        matrix = self._slice(vector.values)
        errors = self.per_feature_errors(matrix)[0]
        return {name: float(err) for name, err in zip(self.feature_names, errors)}

    def top_reconstruction_errors(
        self, vector: FeatureVector, k: int = 5
    ) -> List[Tuple[str, float]]:
        """The ``k`` features that reconstructed worst -- the "why" behind a baseline score."""
        errors = self.reconstruction_errors(vector)
        ranked = sorted(errors.items(), key=lambda item: item[1], reverse=True)
        return ranked[: max(0, k)]

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    @classmethod
    def train(
        cls,
        matrix: np.ndarray,
        feature_names: Sequence[str],
        config: Optional[BaselineTrainConfig] = None,
    ) -> "BaselineModel":
        """Fit the autoencoder on a matrix of (mostly-normal) training feature vectors.

        A held-out slice is carved off first. It is used for two things: early stopping (so the net
        stops before it starts memorizing) and fitting the score normalizer (so the reference error
        distribution comes from data the weights never saw). The remaining rows train the weights.

        The training matrix may be the full feature space or just the numeric block; either way the
        leading ``len(feature_names)`` columns are used.
        """
        config = config or BaselineTrainConfig()
        seed = set_global_seed(config.seed)

        names = list(feature_names)
        input_dim = len(names)
        data = np.asarray(matrix, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {data.shape}")
        if data.shape[1] < input_dim:
            raise ValueError(
                f"matrix has {data.shape[1]} columns but {input_dim} feature names were given"
            )
        data = data[:, :input_dim]
        if data.shape[0] < 2:
            raise ValueError("need at least 2 training rows to fit the baseline")

        # --- deterministic held-out split ---
        rng = np.random.default_rng(seed)
        order = rng.permutation(data.shape[0])
        n_holdout = max(1, int(round(data.shape[0] * config.holdout_fraction)))
        # Guarantee at least one row remains for fitting the weights.
        n_holdout = min(n_holdout, data.shape[0] - 1)
        holdout_idx = order[:n_holdout]
        fit_idx = order[n_holdout:]
        x_fit = torch.from_numpy(data[fit_idx])
        x_holdout = data[holdout_idx]

        net = TabularAutoencoder(input_dim, config.hidden_dim, config.bottleneck_dim)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        loss_fn = nn.MSELoss()

        best_holdout = float("inf")
        best_state: Dict[str, torch.Tensor] = {
            key: value.detach().clone() for key, value in net.state_dict().items()
        }
        epochs_without_improvement = 0
        n_fit = x_fit.shape[0]

        for epoch in range(config.epochs):
            net.train()
            epoch_order = torch.from_numpy(rng.permutation(n_fit))
            for start in range(0, n_fit, config.batch_size):
                batch_idx = epoch_order[start : start + config.batch_size]
                batch = x_fit[batch_idx]
                optimizer.zero_grad()
                reconstruction = net(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()

            # Early-stopping signal: mean reconstruction error on the held-out slice.
            net.eval()
            with torch.no_grad():
                holdout_tensor = torch.from_numpy(x_holdout)
                holdout_loss = float(loss_fn(net(holdout_tensor), holdout_tensor).item())

            if holdout_loss < best_holdout - 1e-7:
                best_holdout = holdout_loss
                best_state = {
                    key: value.detach().clone() for key, value in net.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    logger.info(
                        "Early stopping at epoch %d (best holdout MSE %.6f)",
                        epoch + 1,
                        best_holdout,
                    )
                    break

        net.load_state_dict(best_state)
        net.eval()

        model = cls(net, ScoreNormalizer(center=0.0, scale=1.0), names)

        # Fit the normalizer on held-out residuals -- errors from rows the weights never saw.
        holdout_errors = model.reconstruction_error(x_holdout)
        model.normalizer = ScoreNormalizer.fit(
            np.atleast_1d(holdout_errors), quantile=config.normalizer_quantile
        )
        logger.info(
            "Baseline trained: input_dim=%d, holdout MSE=%.6f, normalizer(center=%.5f, scale=%.5f)",
            input_dim,
            best_holdout,
            model.normalizer.center,
            model.normalizer.scale,
        )
        return model

    # ------------------------------------------------------------------ #
    # Persistence (JSON, never pickle)
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        state = {
            key: value.detach().cpu().numpy().tolist()
            for key, value in self.net.state_dict().items()
        }
        return {
            "model_type": "tabular_autoencoder",
            "input_dim": self.net.input_dim,
            "hidden_dim": self.net.hidden_dim,
            "bottleneck_dim": self.net.bottleneck_dim,
            "feature_names": list(self.feature_names),
            "normalizer": self.normalizer.to_dict(),
            "state_dict": state,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BaselineModel":
        net = TabularAutoencoder(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload.get("hidden_dim", DEFAULT_HIDDEN_DIM)),
            bottleneck_dim=int(payload.get("bottleneck_dim", DEFAULT_BOTTLENECK_DIM)),
        )
        state = {
            key: torch.tensor(np.asarray(value, dtype=np.float32))
            for key, value in payload["state_dict"].items()
        }
        net.load_state_dict(state)
        net.eval()
        normalizer = ScoreNormalizer.from_dict(payload.get("normalizer", {}))
        return cls(net, normalizer, list(payload.get("feature_names", [])))

    def save(self, path: Optional[Path] = None) -> Path:
        """Write the model to JSON and return the path."""
        target = Path(path) if path else Path(settings.artifacts_dir) / BASELINE_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        logger.info("Wrote baseline model to %s", target)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BaselineModel":
        """Load a model previously written by :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If the artifact is missing, with the command needed to build it. Serving must fail
            loudly rather than score with an untrained network.
        """
        source = Path(path) if path else Path(settings.artifacts_dir) / BASELINE_FILE
        if not source.exists():
            raise FileNotFoundError(
                f"No baseline model at {source}. Run: python -m training.train_baseline"
            )
        with source.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


__all__ = [
    "BASELINE_FILE",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_BOTTLENECK_DIM",
    "DEFAULT_NORMALIZER_QUANTILE",
    "TabularAutoencoder",
    "ScoreNormalizer",
    "BaselineTrainConfig",
    "BaselineModel",
]
