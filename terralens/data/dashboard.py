"""Prediction loading shared by the TerraLens dashboard and its checks."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_COLUMNS = ["city", "model", "feature_set", "target", "path"]


def load_prediction_metadata(path: str | Path) -> dict:
    """Ignore malformed sidecars and sidecars belonging to replaced predictions."""
    path = Path(path)
    try:
        details = json.loads(path.with_suffix(".metrics.json").read_text(encoding="utf-8"))
        if not isinstance(details, dict):
            return {}
        expected_hash = details.get("prediction_sha256")
        if expected_hash and expected_hash != sha256(path.read_bytes()).hexdigest():
            return {}
        return details
    except (OSError, ValueError):
        return {}


def load_spatial_metadata(path: str | Path, prediction_metadata: dict) -> dict:
    """Show separate geographic validation only for the matching input dataset."""
    path = Path(path)
    dataset_hash = prediction_metadata.get("dataset_sha256")
    if not dataset_hash:
        return {}
    sidecar = path.parent.parent / "spatial_validation" / f"{path.stem}.spatial.json"
    try:
        details = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(details, dict) or details.get("evaluation") != "nested_spatial_block_cv":
            return {}
        if details.get("dataset_sha256") != dataset_hash:
            return {}
        for key in ("city", "model", "feature_set", "target"):
            if details.get(key) != prediction_metadata.get(key):
                return {}
        expected_hash = details.get("prediction_sha256")
        if (
            not expected_hash
            or expected_hash != sha256(sidecar.with_suffix(".parquet").read_bytes()).hexdigest()
        ):
            return {}
        return details
    except (OSError, ValueError):
        return {}


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Validate coordinates and compute errors from the current prediction values."""
    df = pd.read_parquet(path).copy()
    required = {"lat", "lon", "y_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(missing))}")
    for col in ["lat", "lon", "y_true", "y_pred"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    df = df.loc[valid & df["y_pred"].notna()].copy()
    if "tract_id" not in df:
        df["tract_id"] = [f"area-{i + 1}" for i in range(len(df))]
    df["error"] = df["y_pred"] - df["y_true"]
    return df.reset_index(drop=True)


def prediction_metrics(df: pd.DataFrame) -> dict:
    """Use the same finite truth/prediction pairs for every metric."""
    empty = {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "n": 0}
    if not {"y_true", "y_pred"}.issubset(df):
        return empty
    pairs = df[["y_true", "y_pred"]].apply(pd.to_numeric, errors="coerce")
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
    if pairs.empty:
        return empty
    y, pred = pairs["y_true"].to_numpy(), pairs["y_pred"].to_numpy()
    errors = pred - y
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "r2": 1 - float(np.sum(errors**2)) / total if len(y) > 1 and total > 0 else float("nan"),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "n": len(pairs),
    }


def _resolve_path(value: str, root: Path, pred_dir: Path) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_file():
        return candidate
    # Cached manifests copied from another machine can contain old absolute paths.
    return pred_dir / path.name


def discover_predictions(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Recover missing/stale manifests and prefer real artifacts over demo files."""
    root = Path(root)
    pred_dir = root / "outputs" / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pred_dir / "manifest.csv"
    rows = []
    if manifest_path.is_file():
        try:
            manifest = pd.read_csv(manifest_path)
            if set(MANIFEST_COLUMNS).issubset(manifest):
                for row in manifest[MANIFEST_COLUMNS].dropna().to_dict("records"):
                    if not all(
                        isinstance(value, str) and value and value == value.strip()
                        for value in row.values()
                    ):
                        continue
                    row["path"] = str(_resolve_path(str(row["path"]), root, pred_dir))
                    rows.append(row)
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    for path in sorted(pred_dir.glob("*.parquet")):
        parts = path.stem.split("__", 3)
        if len(parts) == 4:
            rows.append(dict(zip(MANIFEST_COLUMNS, [*parts, str(path)])))

    valid_rows = []
    checked = {}
    for row in rows:
        path = str(row["path"])
        selection = [row[field] for field in MANIFEST_COLUMNS[:4]]
        if not all(
            isinstance(value, str) and value and value == value.strip() for value in selection
        ):
            continue
        # Canonical filenames identify legacy artifacts without embedded labels.
        # A stale manifest must never relabel one city's data as another city's.
        filename_selection = Path(path).stem.split("__", 3)
        if len(filename_selection) == 4 and filename_selection != selection:
            continue
        if path not in checked:
            try:
                checked[path] = load_predictions(path)
            except (OSError, ValueError, KeyError):
                checked[path] = None
        predictions = checked[path]
        if predictions is None or predictions.empty:
            continue
        if any(
            field in predictions and not predictions[field].eq(row[field]).all()
            for field in MANIFEST_COLUMNS[:4]
        ):
            continue
        valid_rows.append(row)
    if not valid_rows:
        return create_sample_predictions(root)
    manifest = pd.DataFrame(valid_rows).drop_duplicates(MANIFEST_COLUMNS[:4], keep="first")
    is_sample = manifest["city"].astype(str).str.endswith("_sample")
    if (~is_sample).any():
        manifest = manifest.loc[~is_sample]
    return manifest.reset_index(drop=True)


def create_sample_predictions(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Create deterministic synthetic display data; these are not trained models."""
    pred_dir = Path(root) / "outputs" / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    count = 60
    idx = np.arange(count)
    lat = 40.60 + (idx % 10) * 0.027 + rng.normal(0, 0.003, count)
    lon = -74.05 + (idx // 10) * 0.045 + rng.normal(0, 0.003, count)
    truth = 62_000 + (lat - lat.min()) * 220_000 + (lon - lon.min()) * 90_000
    truth = np.clip(truth + rng.normal(0, 5_500, count), 30_000, 180_000)
    rows = []
    for model, noise in {"xgboost": 9_000, "extra_trees": 11_000, "hist_gb": 13_000}.items():
        sample = pd.DataFrame(
            {
                "tract_id": [f"sample-{n:03d}" for n in range(1, count + 1)],
                "lat": lat,
                "lon": lon,
                "y_true": truth,
                "y_pred": np.clip(truth + rng.normal(0, noise, count), 20_000, 200_000),
            }
        )
        sample["error"] = sample["y_pred"] - sample["y_true"]
        row = {
            "city": "nyc_sample",
            "model": model,
            "feature_set": "osm_only",
            "target": "median_income",
        }
        for col, value in row.items():
            sample[col] = value
        path = pred_dir / f"nyc_sample__{model}__osm_only__median_income.parquet"
        sample.to_parquet(path, index=False)
        rows.append({**row, "path": str(path)})
    # Discovery always scans the artifacts; an existing user manifest is preserved.
    return pd.DataFrame(rows)
