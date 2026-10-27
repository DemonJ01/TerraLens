#!/usr/bin/env python3
"""Render the README figures from verified, saved results; never train a model.

Run after preparing all three cities and their spatial validation artifacts:
    python scripts/make_readme_figures.py
Only aggregate scores, figures, and source fingerprints are exported to docs/assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "terralens-mpl"))

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terralens.data.dashboard import prediction_metrics  # noqa: E402

CITIES = {
    "nyc": "New York City",
    "chicago": "Chicago / Cook County",
    "san_francisco": "San Francisco",
}
MODELS = {
    "ridge": "Ridge",
    "extra_trees": "Extra Trees",
    "hist_gb": "Hist. Gradient Boosting",
    "xgboost": "XGBoost",
}
# title, filename, symmetric colour cap in display units, unit, multiplier
TARGETS = {
    "median_income": ("Household income", "income", 50000, "USD / year", 1),
    "median_rent": ("Monthly gross rent", "rent", 1000, "USD / month", 1),
    "median_home_value": ("Home value", "home-value", 500000, "USD", 1),
    "poverty_rate": ("Poverty rate", "poverty", 20, "percentage points", 100),
    "unemployment_rate": ("Unemployment rate", "unemployment", 10, "percentage points", 100),
    "bachelors_plus_rate": (
        "Bachelor's degree or higher",
        "education",
        25,
        "percentage points",
        100,
    ),
}
PAPER = "#fffefa"
BLUE = "#315b88"
RUST = "#b36b54"
INK = "#34454b"
MUTED = "#67767d"
SAGE = "#e5eadd"
CONTEXT = "#f8f7ef"
LINE = "#bac7c7"
CMAP = LinearSegmentedColormap.from_list("residual", [BLUE, CONTEXT, RUST])


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def money_or_points(value: float, target: str) -> str:
    return f"{value:.1f} pp" if TARGETS[target][4] == 100 else f"${value:,.0f}"


def verify_hash(path: Path, expected: str) -> None:
    if not expected or fingerprint(path) != expected:
        raise ValueError(f"Missing or stale source fingerprint: {path}")


def source_caption(provenance: dict) -> str:
    periods = sorted({source["acs_period"] for source in provenance.values()})
    dates = sorted(
        {
            (source["osm_source"].get("retrieved_on") or source["osm_source"]["retrieved_at"])[:10]
            for source in provenance.values()
        }
    )
    date_label = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
    return f"ACS {' / '.join(periods)} / OSM retrieved {date_label}."


def load_results(root: Path):
    """Reject demos, stale scores, mismatched observations, and unpaired models."""
    scores, spatial, coverage, sources = [], [], [], []
    predictions, geometries, provenance = {}, {}, {}
    for city in CITIES:
        dataset_path = root / "data" / "processed" / f"{city}_dataset.parquet"
        dataset = pd.read_parquet(dataset_path).set_index("tract_id", verify_integrity=True)
        dataset.index = dataset.index.astype(str)
        dataset_hash = fingerprint(dataset_path)
        source_path = dataset_path.with_suffix(".source.json")
        provenance[city] = json.loads(source_path.read_text())
        year = int(provenance[city]["acs_year"])
        geometry_path = root / "data" / "raw" / "census" / f"{city}_tracts_{year}.gpkg"
        geo = gpd.read_file(geometry_path)
        geo["tract_id"] = geo["tract_id"].astype(str)
        if geo["tract_id"].duplicated().any():
            raise ValueError(f"Duplicate tract boundaries for {city}")
        # Unobserved water-only tracts can extend far offshore and obscure the city.
        context = geo.loc[geo.tract_id.isin(dataset.index)]
        geometries[city] = context.to_crs(
            32618 if city == "nyc" else 32616 if city == "chicago" else 32610
        )
        sources.extend(
            [
                {"path": str(dataset_path.relative_to(root)), "sha256": dataset_hash},
                {"path": str(source_path.relative_to(root)), "sha256": fingerprint(source_path)},
                {
                    "path": str(geometry_path.relative_to(root)),
                    "sha256": fingerprint(geometry_path),
                },
            ]
        )
        for target in TARGETS:
            reference = None
            valid = pd.to_numeric(dataset[target], errors="coerce")
            valid = dataset.loc[np.isfinite(valid) & valid.ge(0), target]
            for model in MODELS:
                stem = f"{city}__{model}__osm_only__{target}"
                path = root / "outputs" / "preds" / f"{stem}.parquet"
                meta_path = path.with_suffix(".metrics.json")
                meta = json.loads(meta_path.read_text())
                if meta.get("evaluation") != "random_holdout":
                    raise ValueError(f"Expected real holdout metadata: {path}")
                verify_hash(path, meta.get("prediction_sha256"))
                if meta.get("dataset_sha256") != dataset_hash:
                    raise ValueError(f"Dataset changed since predictions were saved: {path}")
                if not meta.get("data_source", {}).get("measured_features"):
                    raise ValueError(f"Measured-feature provenance missing: {path}")
                if meta["data_source"] != provenance[city]:
                    raise ValueError(f"Source provenance disagrees with dataset sidecar: {path}")
                df = pd.read_parquet(path)
                df["tract_id"] = df["tract_id"].astype(str)
                if df["tract_id"].duplicated().any():
                    raise ValueError(f"Repeated test tracts: {path}")
                for field, expected in (
                    ("city", city),
                    ("model", model),
                    ("target", target),
                    ("feature_set", "osm_only"),
                ):
                    if meta.get(field) != expected or not df[field].eq(expected).all():
                        raise ValueError(f"Mislabeled {field}: {path}")
                truth = df.set_index("tract_id")["y_true"].sort_index()
                if not truth.index.isin(valid.index).all() or not np.allclose(
                    truth.to_numpy(), valid.reindex(truth.index).to_numpy()
                ):
                    raise ValueError(f"Observed values do not match the dataset: {path}")
                if reference is not None and not truth.equals(reference):
                    raise ValueError(f"Models do not share the same observed test tracts: {path}")
                reference = truth
                train_ids, test_ids = set(meta["train_tract_ids"]), set(meta["test_tract_ids"])
                if train_ids & test_ids or test_ids != set(truth.index):
                    raise ValueError(f"Invalid training/test separation: {path}")
                metrics = prediction_metrics(df)
                if metrics["n"] != len(df) or any(
                    not np.isclose(metrics[key], meta[key]) for key in ("r2", "mae", "rmse")
                ):
                    raise ValueError(f"Saved scores disagree with prediction values: {path}")
                scale = TARGETS[target][4]
                scores.append(
                    dict(
                        city=city,
                        target=target,
                        model=model,
                        n=len(df),
                        n_features=meta["n_features"],
                        r2=metrics["r2"],
                        mae=metrics["mae"] * scale,
                        rmse=metrics["rmse"] * scale,
                        unit=TARGETS[target][3],
                    )
                )
                sources.append(
                    {
                        "path": str(path.relative_to(root)),
                        "sha256": meta["prediction_sha256"],
                        "metadata_sha256": fingerprint(meta_path),
                    }
                )
                predictions[city, target, model] = df
                if not df["tract_id"].isin(geo["tract_id"]).all():
                    raise ValueError(f"Test tracts missing geometry: {path}")
                if meta["valid_label_rows"] != len(valid):
                    raise ValueError(f"Observed-tract count disagrees with metadata: {path}")
                if model == "xgboost":
                    coverage.append(
                        dict(
                            city=city,
                            target=target,
                            observed=len(valid),
                            held_out=len(df),
                            boundary_tracts=len(geo),
                        )
                    )
                sp = root / "outputs" / "spatial_validation" / f"{stem}.spatial.json"
                sm = json.loads(sp.read_text())
                if (
                    sm.get("evaluation") != "nested_spatial_block_cv"
                    or sm.get("dataset_sha256") != dataset_hash
                ):
                    raise ValueError(f"Spatial validation does not match the dataset: {sp}")
                verify_hash(sp.with_suffix(".parquet"), sm.get("prediction_sha256"))
                for field in ("city", "model", "target", "feature_set"):
                    if sm.get(field) != meta.get(field):
                        raise ValueError(f"Mismatched spatial {field}: {sp}")
                spatial_df = pd.read_parquet(sp.with_suffix(".parquet"))
                spatial_truth = spatial_df.set_index("tract_id", verify_integrity=True)["y_true"]
                spatial_truth.index = spatial_truth.index.astype(str)
                if set(spatial_truth.index) != set(valid.index) or not np.allclose(
                    spatial_truth.to_numpy(), valid.reindex(spatial_truth.index).to_numpy()
                ):
                    raise ValueError(f"Spatial observations disagree with the dataset: {sp}")
                spatial_metrics = prediction_metrics(spatial_df)
                if spatial_metrics["n"] != len(valid) or any(
                    not np.isclose(spatial_metrics[key], sm["model_metrics"][key])
                    for key in ("r2", "mae", "rmse")
                ):
                    raise ValueError(f"Spatial scores disagree with saved predictions: {sp}")
                spatial.append(
                    dict(
                        city=city,
                        target=target,
                        model=model,
                        n=sm["rows"],
                        r2=spatial_metrics["r2"],
                        mae=spatial_metrics["mae"] * scale,
                        rmse=spatial_metrics["rmse"] * scale,
                        unit=TARGETS[target][3],
                    )
                )
                sources.append(
                    {
                        "path": str(sp.relative_to(root)),
                        "sha256": fingerprint(sp),
                        "prediction_sha256": sm["prediction_sha256"],
                    }
                )
    return (
        pd.DataFrame(scores),
        pd.DataFrame(spatial),
        pd.DataFrame(coverage),
        predictions,
        geometries,
        sources,
        provenance,
    )


def canvas(title: str, subtitle: str, height: float = 8):
    fig = plt.figure(figsize=(15, height), facecolor=PAPER)
    fig.text(0.04, 0.944, title, fontsize=25, color=BLUE, family="STIXGeneral", style="italic")
    fig.text(0.04, 0.895, subtitle, fontsize=11.5, color=INK)
    return fig


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=160, facecolor=PAPER)
    plt.close(fig)


def residual_maps(out, scores, coverage, predictions, geometries, caption):
    for target, (label, slug, cap, unit, scale) in TARGETS.items():
        title = (
            "Location makes a prediction error specific."
            if target == "median_income"
            else f"Where the estimates miss: {label.lower()}."
        )
        fig = canvas(title, f"{label}  /  XGBoost  /  measured OpenStreetMap features")
        for i, (city, city_name) in enumerate(CITIES.items()):
            x = 0.04 + i * 0.316
            counts = coverage.loc[(coverage.city == city) & (coverage.target == target)].iloc[0]
            fig.text(x, 0.825, city_name, fontsize=20, color=BLUE, family="STIXGeneral")
            fig.text(
                x,
                0.786,
                f"{counts.held_out:,} held-out / {counts.observed:,} observed tracts",
                fontsize=10,
                color=MUTED,
                family="DejaVu Sans Mono",
            )
            ax = fig.add_axes([x, 0.25, 0.292, 0.50], facecolor=SAGE)
            geo = geometries[city]
            df = predictions[city, target, "xgboost"].copy()
            df["residual"] = (df["y_pred"] - df["y_true"]) * scale
            merged = geo.merge(
                df[["tract_id", "residual"]], on="tract_id", how="left", validate="one_to_one"
            )
            merged.plot(ax=ax, color=CONTEXT, edgecolor=LINE, linewidth=0.22)
            merged.loc[merged.residual.notna()].plot(
                ax=ax,
                column="residual",
                cmap=CMAP,
                vmin=-cap,
                vmax=cap,
                edgecolor=LINE,
                linewidth=0.22,
            )
            # SF's offshore Farallon Islands are excluded from this mainland view.
            view = geo
            if city == "san_francisco":
                view = geo.loc[geo.geometry.centroid.to_crs(4326).x > -122.6]
            xmin, ymin, xmax, ymax = view.total_bounds
            width, height = (xmax - xmin) * 1.13, (ymax - ymin) * 1.13
            panel_ratio = (0.292 * 15) / (0.50 * 8)
            width, height = max(width, height * panel_ratio), max(height, width / panel_ratio)
            ax.set_xlim((xmin + xmax - width) / 2, (xmin + xmax + width) / 2)
            ax.set_ylim((ymin + ymax - height) / 2, (ymin + ymax + height) / 2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_visible(False)
            row = scores.loc[
                (scores.city == city) & (scores.target == target) & (scores.model == "xgboost")
            ].iloc[0]
            fig.text(
                x, 0.197, f"R²  {row.r2:.3f}", color=BLUE, fontsize=17, family="DejaVu Sans Mono"
            )
            fig.text(x, 0.158, f"MAE  {money_or_points(row.mae, target)}", color=MUTED, fontsize=12)
        fig.text(
            0.04,
            0.099,
            f"RESIDUAL = PREDICTION − OBSERVED  /  {unit.upper()}",
            fontsize=10,
            color=MUTED,
            family="DejaVu Sans Mono",
        )
        for x, color, text in ((0.58, BLUE, "Underestimate"), (0.765, RUST, "Overestimate")):
            fig.add_artist(
                Rectangle((x, 0.095), 0.017, 0.022, transform=fig.transFigure, color=color)
            )
            fig.text(x + 0.025, 0.097, text, fontsize=11, color=MUTED)
        cap_text = f"{cap:g} percentage points" if scale == 100 else f"${cap:,.0f}"
        fig.text(
            0.04,
            0.051,
            f"Colour saturates at ±{cap_text}; scores use uncapped errors.",
            fontsize=9.5,
            color=MUTED,
        )
        fig.text(
            0.04,
            0.023,
            "Pale tracts provide context. Maps scaled individually; SF mainland shown.  " + caption,
            fontsize=9,
            color=MUTED,
        )
        save(fig, out, f"{slug}-residuals.png")


def model_comparison(out, scores, caption):
    fig = canvas(
        "One score cannot tell the whole story.",
        "Random 20% holdout R²  /  four model families  /  six Census measures",
        8.8,
    )
    palette = LinearSegmentedColormap.from_list("scores", [RUST, CONTEXT, BLUE])
    for i, (city, name) in enumerate(CITIES.items()):
        x = 0.196 + i * 0.258
        fig.text(x, 0.822, name, fontsize=18, color=BLUE, family="STIXGeneral")
        matrix = scores.loc[scores.city == city].pivot(index="target", columns="model", values="r2")
        matrix = matrix.reindex(index=TARGETS, columns=MODELS)
        ax = fig.add_axes([x, 0.26, 0.235, 0.49])
        ax.imshow(
            matrix,
            cmap=palette,
            norm=TwoSlopeNorm(vmin=min(-0.3, scores.r2.min()), vcenter=0, vmax=1),
            aspect="auto",
        )
        ax.set_xticks(range(4), ["Ridge", "Extra\nTrees", "Hist.\nGB", "XGBoost"], fontsize=9.5)
        ax.tick_params(axis="both", length=0, pad=10, labelcolor=INK)
        ax.set_yticks(
            range(6), [TARGETS[t][0] for t in TARGETS] if i == 0 else [""] * 6, fontsize=10
        )
        for row in range(6):
            for col in range(4):
                value = matrix.iloc[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color=PAPER if value > 0.65 else INK,
                    family="DejaVu Sans Mono",
                    weight="bold" if col == np.argmax(matrix.iloc[row].to_numpy()) else "normal",
                )
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
        ax.grid(which="minor", color=PAPER, linewidth=5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(
        0.04,
        0.139,
        "Higher R² is better. Bold marks the highest score within each city and measure.",
        fontsize=11,
        color=INK,
    )
    fig.text(
        0.04,
        0.092,
        "Scores are recomputed from matching held-out tracts. R² can be negative"
        " and is not an accuracy percentage.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.047,
        f"{' / '.join(map(str, sorted(scores.n_features.unique())))} geographic predictors. "
        + caption
        + " Full scores: model-scores.csv.",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, out, "model-comparison.png")


def validation_comparison(out, scores, spatial):
    spatial = spatial.loc[spatial.model == "xgboost"]
    fig = canvas(
        "A harder test holds whole places out.",
        "XGBoost R²  /  random holdout and nested geographic block validation",
        8.2,
    )
    low = min(-0.1, np.floor(spatial.r2.min() * 10) / 10 - 0.08)
    for i, (city, name) in enumerate(CITIES.items()):
        x = 0.20 + i * 0.258
        fig.text(x, 0.811, name, fontsize=18, color=BLUE, family="STIXGeneral")
        ax = fig.add_axes([x, 0.26, 0.223, 0.49], facecolor=PAPER)
        hold = scores.loc[(scores.city == city) & (scores.model == "xgboost")].set_index("target")
        block = spatial.loc[spatial.city == city].set_index("target")
        for j, target in enumerate(TARGETS):
            a, b = hold.loc[target, "r2"], block.loc[target, "r2"]
            ax.plot([a, b], [j, j], color=LINE, linewidth=2, zorder=1)
            ax.scatter(a, j, color=BLUE, s=48, zorder=3)
            ax.scatter(b, j, color=RUST, s=48, marker="D", zorder=3)
        ax.set_yticks(
            range(6), [TARGETS[t][0] for t in TARGETS] if i == 0 else [""] * 6, fontsize=10
        )
        ax.set_ylim(5.6, -0.6)
        ax.set_xlim(low, 0.9)
        ax.axvline(0, color=LINE, linewidth=1, linestyle="--")
        ax.grid(axis="x", color=LINE, alpha=0.45, linewidth=0.5)
        ax.set_xlabel("R²", color=MUTED, fontsize=10)
        ax.tick_params(axis="both", length=0, pad=9, labelcolor=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.20, 0.16, "●  Random 20% holdout", color=BLUE, fontsize=12)
    fig.text(0.48, 0.16, "◆  Geographic block validation", color=RUST, fontsize=12)
    fig.text(
        0.04,
        0.094,
        "Same source dataset; different test tracts. Geographic scores pool all"
        " out-of-fold predictions (3 outer / 3 inner folds).",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.049,
        "4 × 4 coordinate grid; no buffer between blocks. Neither scheme tests"
        " transfer to a new city or future year.",
        fontsize=10,
        color=MUTED,
    )
    save(fig, out, "validation-comparison.png")


def income_tables(out, scores, spatial):
    fig = canvas(
        "Four models. Three cities. Two ways to test.",
        "Median household income  /  identical holdout tracts within each city"
        "  /  errors in USD per year",
        7.2,
    )
    names = ["Ridge", "Extra Trees", "Histogram GB", "XGBoost"]
    for i, (city, name) in enumerate(CITIES.items()):
        x = 0.04 + i * 0.316
        fig.text(x, 0.802, name, fontsize=20, color=BLUE, family="STIXGeneral")
        ax = fig.add_axes([x, 0.27, 0.292, 0.47])
        ax.set_axis_off()
        columns = [0.025, 0.48, 0.70, 0.93]
        for column, label in zip(columns, ["MODEL", "HOLDOUT\nR²", "MAE\nUSD", "SPATIAL\nR²"]):
            ax.text(
                column,
                0.92,
                label,
                ha="left" if column == columns[0] else "right",
                va="center",
                fontsize=9,
                color=MUTED,
                linespacing=1.6,
            )
        hold = scores.loc[(scores.city == city) & (scores.target == "median_income")].set_index(
            "model"
        )
        block = spatial.loc[(spatial.city == city) & (spatial.target == "median_income")].set_index(
            "model"
        )
        for j, (model, label) in enumerate(zip(MODELS, names)):
            y = 0.70 - j * 0.19
            if j % 2 == 0:
                ax.add_patch(Rectangle((0, y - 0.082), 1, 0.164, color=SAGE, alpha=0.7))
            values = [
                label,
                f"{hold.loc[model, 'r2']:.3f}",
                f"{hold.loc[model, 'mae']:,.0f}",
                f"{block.loc[model, 'r2']:.3f}",
            ]
            for k, (column, value) in enumerate(zip(columns, values)):
                ax.text(
                    column,
                    y,
                    value,
                    ha="left" if k == 0 else "right",
                    va="center",
                    fontsize=10,
                    color=INK if k != 3 else RUST,
                    family="DejaVu Sans" if k == 0 else "DejaVu Sans Mono",
                )
    fig.text(
        0.04,
        0.178,
        "Holdout R² and MAE use a random 20% test sample. Spatial R² uses nested"
        " geographic block validation.",
        fontsize=11,
        color=INK,
    )
    fig.text(
        0.04,
        0.113,
        "Higher R² and lower MAE are better. Different evaluation schemes use"
        " different test observations.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.053,
        "Values rounded for display; full precision for all six measures:"
        " model-scores.csv and spatial-scores.csv.",
        fontsize=10,
        color=MUTED,
    )
    save(fig, out, "income-models.png")


def coverage_figure(out, coverage):
    fig = canvas(
        "Every result begins with observed tracts.",
        "Valid ACS labels by measure  /  held-out tracts are a subset of observations",
        7.6,
    )
    for i, (city, name) in enumerate(CITIES.items()):
        x = 0.20 + i * 0.258
        fig.text(x, 0.81, name, fontsize=18, color=BLUE, family="STIXGeneral")
        ax = fig.add_axes([x, 0.27, 0.23, 0.47], facecolor=PAPER)
        rows = coverage.loc[coverage.city == city].set_index("target").reindex(TARGETS)
        y = np.arange(6)
        ax.barh(y, rows.observed, color=SAGE, height=0.59)
        ax.barh(y, rows.held_out, color=BLUE, height=0.59)
        for j, row in enumerate(rows.itertuples()):
            ax.text(
                row.observed + rows.observed.max() * 0.035,
                j,
                f"{row.held_out:,} / {row.observed:,}",
                va="center",
                fontsize=9.5,
                color=INK,
                family="DejaVu Sans Mono",
            )
        ax.set_yticks(y, [TARGETS[t][0] for t in TARGETS] if i == 0 else [""] * 6, fontsize=10)
        ax.set_xlim(0, rows.observed.max() * 1.78)
        ax.set_ylim(5.6, -0.6)
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0, pad=10, labelcolor=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(0.20, 0.16, "BLUE  Held out", color=BLUE, fontsize=12)
    fig.text(0.46, 0.16, "SAGE  All observed tracts", color=MUTED, fontsize=12)
    fig.text(
        0.04,
        0.09,
        "Counts vary by label availability. Missing or invalid labels are excluded"
        " separately for each measure.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.044,
        "Bar lengths are scaled within each city; count labels allow comparison"
        " across cities. Data: coverage.csv.",
        fontsize=10,
        color=MUTED,
    )
    save(fig, out, "data-coverage.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Prepared TerraLens checkout")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "assets")
    args = parser.parse_args()
    try:
        scores, spatial, coverage, predictions, geometries, sources, provenance = load_results(
            args.root
        )
        caption = source_caption(provenance)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(
            1,
            f"Cannot render README figures: {error}\n"
            "Prepare the three cities and run scripts/validate_models.py for each first.\n",
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": True})
    residual_maps(args.output_dir, scores, coverage, predictions, geometries, caption)
    model_comparison(args.output_dir, scores, caption)
    income_tables(args.output_dir, scores, spatial)
    validation_comparison(args.output_dir, scores, spatial)
    coverage_figure(args.output_dir, coverage)
    for name, data in (
        ("model-scores", scores),
        ("spatial-scores", spatial),
        ("coverage", coverage),
    ):
        data.to_csv(args.output_dir / f"{name}.csv", index=False)
    manifest = {
        "generator": "scripts/make_readme_figures.py",
        "data_sources": {
            city: {key: source[key] for key in ("acs_year", "acs_period", "osm_source")}
            for city, source in provenance.items()
        },
        "evaluation": "Maps and model scores: random 20% holdout. Spatial scores: nested block CV.",
        "rates": "MAE, RMSE and residuals exported in percentage points; source fractions × 100.",
        "maps": "Context: prepared dataset tracts. Individually scaled; SF mainland view.",
        "sources": sources,
        "figures": {p.name: fingerprint(p) for p in sorted(args.output_dir.glob("*.png"))},
    }
    (args.output_dir / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote 10 figures, 3 CSVs and a source manifest to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
