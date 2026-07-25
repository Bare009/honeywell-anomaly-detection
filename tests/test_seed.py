"""Determinism tests.

Every metric this project reports, and the scripted demo itself, assume the whole pipeline
is reproducible under seed 42. If seeding breaks, results stop being comparable across
runs -- so these tests guard the guarantee directly rather than trusting it.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from common.config import settings
from common.seed import lightgbm_params, new_rng, set_global_seed


class TestSetGlobalSeed:
    """The global seeding call must make all libraries reproducible together."""

    def test_returns_applied_seed(self) -> None:
        assert set_global_seed() == settings.random_seed

    def test_explicit_seed_overrides_default(self) -> None:
        assert set_global_seed(1234) == 1234

    def test_sets_pythonhashseed(self) -> None:
        """Child processes need stable hashing for dict/set iteration order."""
        set_global_seed(7)
        assert os.environ["PYTHONHASHSEED"] == "7"

    def test_python_random_is_reproducible(self) -> None:
        set_global_seed(42)
        first = [random.random() for _ in range(5)]
        set_global_seed(42)
        assert [random.random() for _ in range(5)] == first

    def test_numpy_global_state_is_reproducible(self) -> None:
        set_global_seed(42)
        first = np.random.rand(5).tolist()
        set_global_seed(42)
        assert np.random.rand(5).tolist() == first

    def test_different_seeds_produce_different_streams(self) -> None:
        """Sanity check that seeding is actually taking effect."""
        set_global_seed(1)
        first = np.random.rand(5).tolist()
        set_global_seed(2)
        assert np.random.rand(5).tolist() != first

    def test_seeds_torch_when_available(self) -> None:
        torch = pytest.importorskip("torch")
        set_global_seed(42)
        first = torch.rand(4).tolist()
        set_global_seed(42)
        assert torch.rand(4).tolist() == first


class TestNewRng:
    """Isolated generators let a component be reproducible without global state."""

    def test_default_seed_matches_settings(self) -> None:
        assert new_rng().random() == new_rng(settings.random_seed).random()

    def test_two_generators_with_same_seed_agree(self) -> None:
        assert new_rng(11).integers(0, 100, size=6).tolist() == (
            new_rng(11).integers(0, 100, size=6).tolist()
        )

    def test_generator_is_isolated_from_global_state(self) -> None:
        """Perturbing the global NumPy state must not shift an isolated generator."""
        rng = new_rng(5)
        expected = rng.random()

        np.random.seed(999)
        _ = np.random.rand(100)

        assert new_rng(5).random() == expected


class TestLightGbmParams:
    """LightGBM needs explicit determinism flags, not just a seed."""

    def test_all_seed_fields_are_set(self) -> None:
        params = lightgbm_params()
        for key in (
            "seed",
            "random_state",
            "bagging_seed",
            "feature_fraction_seed",
            "data_random_seed",
        ):
            assert params[key] == settings.random_seed

    def test_determinism_flags(self) -> None:
        params = lightgbm_params()
        assert params["deterministic"] is True
        assert params["force_row_wise"] is True
        assert params["num_threads"] == 1

    def test_overrides_are_merged(self) -> None:
        params = lightgbm_params(objective="multiclass", num_class=9)
        assert params["objective"] == "multiclass"
        assert params["num_class"] == 9
        assert params["deterministic"] is True

    def test_overrides_can_replace_defaults(self) -> None:
        assert lightgbm_params(num_threads=4)["num_threads"] == 4

    def test_explicit_seed_is_used(self) -> None:
        assert lightgbm_params(seed=99)["seed"] == 99
