"""Categorical encoders and the numeric scaler.

Two deliberate choices here.

**Unseen categories are a feature, not an error.** A country, resource or device the model
has never seen is exactly what an intrusion looks like. So every encoder reserves code ``0``
for "unseen" and reports a novelty flag alongside the code. Nothing raises, and the novelty
itself becomes signal.

**Everything persists as JSON, not pickle.** A pickled scikit-learn scaler is tied to the
library version that created it, and would break the serving container after any dependency
bump. These are a handful of means and category lists; JSON keeps artifacts readable,
diffable and version-independent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Reserved code for a value the encoder never saw while fitting.
UNSEEN_CODE = 0

#: Smallest standard deviation used when scaling. A constant feature would otherwise divide
#: by zero; treating it as unit-variance leaves it at zero after centring, which is correct --
#: a feature that never varies carries no information.
MIN_STD = 1e-6


@dataclass
class CategoricalEncoder:
    """Maps a categorical value to a stable integer code.

    Codes start at 1; ``0`` always means "not seen during fitting".
    """

    name: str
    categories: List[str] = field(default_factory=list)
    max_categories: Optional[int] = None

    def __post_init__(self) -> None:
        self._index: Dict[str, int] = {
            value: position + 1 for position, value in enumerate(self.categories)
        }

    @classmethod
    def fit(
        cls,
        name: str,
        values: Iterable[Any],
        max_categories: Optional[int] = None,
        min_count: int = 1,
    ) -> "CategoricalEncoder":
        """Learn the category set from observed values.

        Parameters
        ----------
        max_categories:
            Keep only the most frequent N. High-cardinality fields (resources) would
            otherwise produce a code space larger than the training data can support.
        min_count:
            Drop categories seen fewer than this many times; they behave more like novelty
            than like a category.
        """
        counts: Dict[str, int] = {}
        for value in values:
            key = cls.normalize(value)
            if key is None:
                continue
            counts[key] = counts.get(key, 0) + 1

        kept = [value for value, count in counts.items() if count >= min_count]
        # Sort by frequency then name: frequency makes truncation sensible, and the name
        # tiebreak keeps the code assignment deterministic.
        kept.sort(key=lambda value: (-counts[value], value))
        if max_categories is not None:
            kept = kept[:max_categories]

        return cls(name=name, categories=kept, max_categories=max_categories)

    @staticmethod
    def normalize(value: Any) -> Optional[str]:
        """Canonical string form of a category value, or ``None`` for missing."""
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        text = str(value).strip()
        return text or None

    def transform(self, value: Any) -> Tuple[int, bool]:
        """Encode one value.

        Returns
        -------
        (code, is_novel)
            ``code`` is ``UNSEEN_CODE`` for unknown or missing values, and ``is_novel`` is
            True in that case so the caller can surface it as a feature.
        """
        key = self.normalize(value)
        if key is None:
            return UNSEEN_CODE, True
        code = self._index.get(key)
        if code is None:
            return UNSEEN_CODE, True
        return code, False

    def inverse(self, code: int) -> Optional[str]:
        """Recover the category for a code, or ``None`` for the unseen code."""
        if code <= 0 or code > len(self.categories):
            return None
        return self.categories[code - 1]

    @property
    def cardinality(self) -> int:
        """Number of distinct codes, including the reserved unseen code."""
        return len(self.categories) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "categories": list(self.categories),
            "max_categories": self.max_categories,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CategoricalEncoder":
        return cls(
            name=payload["name"],
            categories=list(payload.get("categories", [])),
            max_categories=payload.get("max_categories"),
        )


@dataclass
class NumericScaler:
    """Standardizes a numeric feature vector using fitted means and standard deviations.

    Hand-rolled rather than ``sklearn.StandardScaler`` so it round-trips through JSON and
    behaves identically in the training process and the serving container.
    """

    names: List[str] = field(default_factory=list)
    means: List[float] = field(default_factory=list)
    stds: List[float] = field(default_factory=list)

    @classmethod
    def fit(cls, names: Sequence[str], matrix: np.ndarray) -> "NumericScaler":
        """Learn per-column means and standard deviations."""
        data = np.asarray(matrix, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {data.shape}")
        if data.shape[1] != len(names):
            raise ValueError(
                f"matrix has {data.shape[1]} columns but {len(names)} names were given"
            )

        # NaNs would poison the statistics; they are treated as missing rather than zero so a
        # single bad row cannot shift a whole column's mean.
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        stds = np.nan_to_num(stds, nan=1.0, posinf=1.0, neginf=1.0)
        stds = np.where(stds < MIN_STD, 1.0, stds)

        return cls(names=list(names), means=means.tolist(), stds=stds.tolist())

    def transform(self, vector: np.ndarray) -> np.ndarray:
        """Standardize one vector or a matrix of vectors."""
        data = np.asarray(vector, dtype=float)
        means = np.asarray(self.means, dtype=float)
        stds = np.asarray(self.stds, dtype=float)

        expected = len(self.names)
        if data.shape[-1] != expected:
            raise ValueError(
                f"expected {expected} features, got {data.shape[-1]}"
            )

        scaled = (np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0) - means) / stds
        return np.clip(scaled, -12.0, 12.0)

    def inverse_transform(self, vector: np.ndarray) -> np.ndarray:
        """Recover approximate raw values from standardized ones (for explanations)."""
        data = np.asarray(vector, dtype=float)
        return data * np.asarray(self.stds, dtype=float) + np.asarray(self.means, dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {"names": list(self.names), "means": list(self.means), "stds": list(self.stds)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NumericScaler":
        return cls(
            names=list(payload.get("names", [])),
            means=list(payload.get("means", [])),
            stds=list(payload.get("stds", [])),
        )


@dataclass
class EncoderBundle:
    """Every fitted transform the feature pipeline needs, in one saveable object."""

    categorical: Dict[str, CategoricalEncoder] = field(default_factory=dict)
    scaler: Optional[NumericScaler] = None
    numeric_names: List[str] = field(default_factory=list)
    categorical_names: List[str] = field(default_factory=list)

    @property
    def feature_names(self) -> List[str]:
        """Full ordered feature list: numeric columns first, then categorical codes."""
        return list(self.numeric_names) + [f"{name}_code" for name in self.categorical_names]

    @property
    def categorical_indices(self) -> List[int]:
        """Column positions of the categorical codes, for LightGBM's ``categorical_feature``."""
        offset = len(self.numeric_names)
        return [offset + position for position in range(len(self.categorical_names))]

    def encode_categoricals(self, values: Dict[str, Any]) -> Tuple[List[int], Dict[str, bool]]:
        """Encode every categorical field, returning codes in order plus novelty flags."""
        codes: List[int] = []
        novelty: Dict[str, bool] = {}
        for name in self.categorical_names:
            encoder = self.categorical.get(name)
            if encoder is None:
                codes.append(UNSEEN_CODE)
                novelty[name] = True
                continue
            code, is_novel = encoder.transform(values.get(name))
            codes.append(code)
            novelty[name] = is_novel
        return codes, novelty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numeric_names": list(self.numeric_names),
            "categorical_names": list(self.categorical_names),
            "categorical": {
                name: encoder.to_dict() for name, encoder in self.categorical.items()
            },
            "scaler": self.scaler.to_dict() if self.scaler else None,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EncoderBundle":
        scaler_payload = payload.get("scaler")
        return cls(
            numeric_names=list(payload.get("numeric_names", [])),
            categorical_names=list(payload.get("categorical_names", [])),
            categorical={
                name: CategoricalEncoder.from_dict(value)
                for name, value in (payload.get("categorical") or {}).items()
            },
            scaler=NumericScaler.from_dict(scaler_payload) if scaler_payload else None,
        )

    def save(self, path: Path) -> Path:
        """Write the bundle to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "EncoderBundle":
        """Read a bundle previously written by :meth:`save`."""
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


__all__ = [
    "UNSEEN_CODE",
    "MIN_STD",
    "CategoricalEncoder",
    "NumericScaler",
    "EncoderBundle",
]
