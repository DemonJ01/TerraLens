from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def prepare_xgboost_runtime() -> None:
    """Load an environment-local OpenMP runtime when one is supplied on macOS."""
    if sys.platform == "darwin":
        runtime = Path(sys.prefix) / "lib" / "libomp.dylib"
        if runtime.exists():
            ctypes.CDLL(str(runtime))


def get_model(name: str, seed: int = 42):
    name = name.lower()

    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0, random_state=seed)

    if name == "elasticnet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=seed)

    if name == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_depth=6,
            max_iter=500,
            random_state=seed,
        )

    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=900,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    if name == "mlp":
        from sklearn.neural_network import MLPRegressor

        # Feed-forward network for numeric tract features.
        return MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=25,
            verbose=False,
        )

    if name == "xgboost":
        try:
            prepare_xgboost_runtime()
            from xgboost import XGBRegressor
        except (ImportError, OSError, ValueError) as exc:
            raise RuntimeError(
                "XGBoost could not load. Install xgboost and, on macOS, the libomp "
                "OpenMP runtime (brew install libomp), or select a different model."
            ) from exc
        return XGBRegressor(
            n_estimators=1200,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {name}")
