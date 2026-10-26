from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from terralens.data.comparison import align_model_predictions
from terralens.data.dashboard import (
    discover_predictions,
    load_prediction_metadata,
    load_predictions,
    load_spatial_metadata,
    prediction_metrics,
)
from terralens.utils import maps as map_utils
from terralens.ui.comparison import render_model_comparison

MAP_RENDERER_VERSION = sha256(Path(map_utils.__file__).read_bytes()).hexdigest()

st.set_page_config(
    page_title="TerraLens",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
hr { margin: 1.2rem 0; }
.smallcap { color: rgba(230,237,243,0.78); font-size: 0.95rem; line-height: 1.35; }
.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 0.95rem 1.05rem;
  background: rgba(255,255,255,0.03);
}
.badge {
  display:inline-block;
  padding: 0.16rem 0.55rem;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 999px;
  margin-right: .35rem;
  margin-top: .25rem;
  background: rgba(255,255,255,0.04);
  font-size: 0.85rem;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(105px, 1fr));
  gap: 0.65rem;
  margin: 0.85rem 0;
}
.metric-card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 0.65rem 0.8rem;
  background: rgba(255,255,255,0.025);
}
.metric-label { color: rgba(230,237,243,0.70); font-size: 0.78rem; }
.metric-value { font-size: 1.45rem; font-weight: 700; margin-top: 0.1rem; }
@media (max-width: 700px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
""",
    unsafe_allow_html=True,
)

TARGET_LABELS = {
    "median_income": "Median household income (USD)",
    "median_rent": "Median rent (USD)",
    "median_home_value": "Median home value (USD)",
    "poverty_rate": "Poverty rate (0–1)",
    "unemployment_rate": "Unemployment rate (0–1)",
    "bachelors_plus_rate": "Bachelor’s+ rate (0–1)",
}

CITY_LABELS = {
    "nyc": "NYC",
    "chicago": "Chicago / Cook County",
    "san_francisco": "San Francisco",
    "nyc_sample": "NYC (sample)",
}

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "extra_trees": "Extra Trees",
    "hist_gb": "Histogram Gradient Boosting",
    "ridge": "Ridge Regression",
    "mlp": "Neural Network (MLP)",
}

FEATURESET_LABELS = {
    "osm_only": "OSM-only",
    "osm_311": "OSM + 311 (NYC)",
}


def fmt_target(t: str) -> str:
    return TARGET_LABELS.get(t, t)


def fmt_city(city: str) -> str:
    return CITY_LABELS.get(city, city.upper())


def fmt_model(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def fmt_featureset(fs: str) -> str:
    return FEATURESET_LABELS.get(fs, fs)


def load_manifest() -> pd.DataFrame:
    return discover_predictions()


@st.cache_data(show_spinner=False)
def _load_preds_cached(path: str, version: tuple[int, int]) -> pd.DataFrame:
    return load_predictions(path)


def load_preds(path: str) -> pd.DataFrame:
    stat = Path(path).stat()
    return _load_preds_cached(path, (stat.st_mtime_ns, stat.st_size))


@st.cache_data(show_spinner=False, max_entries=32)
def render_map_html(
    data: pd.DataFrame, value_col: str, mode: str, clip: bool, renderer_version: str
) -> str:
    """Reuse map HTML until the data, settings, or renderer source changes."""
    return map_utils.build_prediction_map(data, value_col, mode, clip=clip).get_root().render()


def reload_results() -> None:
    """Reread prediction files and rebuild maps on an explicit refresh."""
    _load_preds_cached.clear()
    render_map_html.clear()


def compute_metrics(df: pd.DataFrame) -> dict:
    return prediction_metrics(df)


def format_value(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:,.0f}" if abs(value) >= 1 else f"{value:.4f}"


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "⬇️ Download CSV"):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        width="stretch",
        on_click="ignore",
    )


def local_explain(city: str, target: str, feature_set: str, model: str, df: pd.DataFrame) -> str:
    """Return a short, plain-language explanation for the selected result."""
    met = compute_metrics(df)
    description = (
        "This view contains synthetic locations and values to try the controls, "
        "not predictions from a trained model."
        if city.endswith("_sample")
        else f"{fmt_model(model)} estimates {fmt_target(target).lower()} for "
        f"census tracts in {fmt_city(city)}, using {fmt_featureset(feature_set)} data."
    )
    return "\n".join(
        [
            "### About this result",
            description,
            "",
            "### What the map means",
            "- **Prediction:** TerraLens' estimate.",
            "- **Ground truth:** the value in the dataset.",
            "- **Error:** prediction minus the real value. "
            "Positive means the estimate was too high.",
            "",
            "### Accuracy",
            (
                f"- **R²: {met['r2']:.3f}** — higher is better."
                if np.isfinite(met["r2"])
                else "- **R²: —** — requires multiple different observed values."
            ),
            f"- **MAE: {format_value(met['mae'])}** — the average size of an error.",
            f"- **Tracts checked: {met['n']:,}**.",
            "",
            "These scores describe the loaded predictions. Sample scores do not measure "
            "real-world accuracy; other scores depend on how the data was split for testing.",
        ]
    )


st.markdown(
    """
<div class="card">
  <h1 style="margin:0;">🌆 TerraLens</h1>
  <div class="smallcap">
    Explore neighborhood estimates, compare models, and check results on the map.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

m = load_manifest()
if m.empty:
    st.error("No prediction data is available. See the README for data preparation steps.")
    st.stop()

if set(m["city"].dropna()) == {"nyc_sample"}:
    st.info(
        "Sample mode: these locations, values, and model scores are synthetic examples. "
        "Real project results appear when prediction data is available."
    )

with st.sidebar:
    st.header("Selections")
    st.caption("Options reflect the prediction data available on this computer.")
    st.button("Reload results", on_click=reload_results, width="stretch")

    cities = sorted(m["city"].dropna().unique().tolist())
    city = st.selectbox("City", cities, index=0, format_func=fmt_city)

    m_city = m[m["city"] == city].copy()
    feature_sets = sorted(m_city["feature_set"].dropna().unique().tolist())
    feature_set = st.selectbox("Feature set", feature_sets, index=0, format_func=fmt_featureset)

    m_fs = m_city[m_city["feature_set"] == feature_set].copy()
    targets = sorted(m_fs["target"].dropna().unique().tolist())
    target = st.selectbox("Target", targets, index=0, format_func=fmt_target)

    m_t = m_fs[m_fs["target"] == target].copy()
    models = sorted(m_t["model"].dropna().unique().tolist())
    model = st.selectbox("Model", models, index=0, format_func=fmt_model)

    st.divider()
    st.subheader("Map settings")
    map_mode = st.radio("Map mode", ["Cluster", "Points", "Heatmap"], index=0)
    layer = st.radio("Color by", ["Prediction", "Error", "Ground truth"], index=0)
    clip_outliers = st.toggle("Clip outliers (2–98%)", value=True)

sel = m[
    (m.city == city) & (m.feature_set == feature_set) & (m.target == target) & (m.model == model)
]
if sel.empty:
    st.error("This result is no longer available. Reload results to refresh the choices.")
    st.stop()

path = str(sel.iloc[0]["path"])
df = load_preds(path)
details = load_prediction_metadata(path)
if details.get("evaluation") == "random_holdout":
    st.caption("Map and accuracy scores use held-out tracts, not the rows used to fit the model.")
    source = details.get("data_source", {})
    if isinstance(source, dict) and source.get("acs_period"):
        st.caption(
            f"Observed labels: ACS {source['acs_period']}. OSM and ACS dates differ; "
            "these are retrospective estimates, not forecasts."
        )

met = compute_metrics(df)
st.markdown(
    f"""
<div class="card">
  <div class="badge">{fmt_city(city)}</div>
  <div class="badge">{fmt_target(target)}</div>
  <div class="badge">{fmt_featureset(feature_set)}</div>
  <div class="badge">{fmt_model(model)}</div>
</div>
""",
    unsafe_allow_html=True,
)

r2_display = f"{met['r2']:.3f}" if np.isfinite(met["r2"]) else "—"
mae_display = format_value(met["mae"])
rmse_display = format_value(met["rmse"])
st.markdown(
    f"""
<div class="metrics">
  <div class="metric-card">
    <div class="metric-label">R²</div><div class="metric-value">{r2_display}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">MAE</div><div class="metric-value">{mae_display}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">RMSE</div><div class="metric-value">{rmse_display}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Tracts</div><div class="metric-value">{len(df):,}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
if met["n"] < len(df):
    st.caption(
        f"{len(df):,} tracts have predictions; "
        f"{met['n']:,} have observed values for accuracy scores."
    )
download_csv_button(
    df, f"{city}_{model}_{feature_set}_{target}.csv", label="⬇️ Download current view (CSV)"
)

tab_map, tab_diag, tab_lead, tab_compare, tab_explain = st.tabs(
    ["Map", "Quick check", "Leaderboard", "Compare", "Explain"]
)

with tab_map:
    if layer == "Prediction":
        value_col = "y_pred"
    elif layer == "Error":
        value_col = "error"
    else:
        value_col = "y_true"

    dmap = df.assign(target=target)
    if value_col not in dmap.columns:
        st.error(f"Missing column: {value_col}")
    else:
        dmap[value_col] = pd.to_numeric(dmap[value_col], errors="coerce")
        dmap = dmap.dropna(subset=[value_col, "lat", "lon"]).copy()

        if dmap.empty:
            st.info(
                "This layer has no values yet. Choose Prediction to view the available estimates."
            )
        else:
            components.html(
                render_map_html(dmap, value_col, map_mode, clip_outliers, MAP_RENDERER_VERSION),
                height=480,
                scrolling=False,
            )
            st.caption("Drag to move, scroll or use +/− to zoom. Reset view shows all areas.")
            if map_mode == "Heatmap":
                st.caption(
                    "Brighter areas show larger errors."
                    if layer == "Error"
                    else "Brighter areas show higher values."
                )
            else:
                st.caption(
                    "Blue = estimate too low; red = estimate too high."
                    if layer == "Error"
                    else "Blue = lower values; orange = higher values. Click a point to explore."
                )

with tab_diag:
    st.markdown("### Quick check")
    st.caption("A simple look at how close the estimates are to the values in the data.")
    need = {"y_true", "y_pred"}
    if not need.issubset(df.columns) or not compute_metrics(df)["n"]:
        st.info(
            "Observed values are needed for an accuracy check. "
            "You can still view predictions on the map."
        )
    else:
        d = df[["y_true", "y_pred", "error", "tract_id", "lat", "lon"]].copy()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"]).copy()
        if "error" not in d.columns or d["error"].isna().all():
            d["error"] = d["y_pred"] - d["y_true"]
        st.markdown("#### Estimated value vs. data value")
        st.scatter_chart(
            d, x="y_true", y="y_pred", x_label="Data value", y_label="Estimate", width="stretch"
        )
        st.caption("Dots closer to a diagonal pattern mean the estimates are closer to the data.")

        average_error = float(np.abs(d["error"]).mean())
        st.metric("Average error", format_value(average_error))
        st.caption("Use the Error layer on the map if you want to see where the larger misses are.")

        baseline = details.get("baseline", {})
        if isinstance(baseline, dict) and baseline.get("strategy") == "training_mean":
            try:
                training_mean = float(baseline["value"])
            except (KeyError, TypeError, ValueError):
                training_mean = float("nan")
            if np.isfinite(training_mean):
                baseline_scores = compute_metrics(d.assign(y_pred=training_mean))
                st.markdown("#### Compare with a simple baseline")
                st.caption(
                    "The baseline predicts the training-data average for every test tract. "
                    "It uses no test values to choose that average."
                )
                st.metric("Baseline RMSE", format_value(baseline_scores["rmse"]))
                if baseline_scores["rmse"] > 0:
                    reduction = 1 - (met["rmse"] / baseline_scores["rmse"]) ** 2
                    if reduction > 0:
                        st.write(
                            f"This model has **{reduction:.1%} lower squared error** "
                            "than the baseline."
                        )
                    else:
                        st.warning(
                            f"This model has {abs(reduction):.1%} higher squared error than "
                            "the training-average baseline on these test tracts."
                        )

        selection = details.get("model_selection", {})
        if isinstance(selection, dict) and selection.get("method") == "cross_validation":
            folds = selection.get("folds")
            st.caption(
                f"Model settings were selected with {folds}-fold cross-validation "
                "within the training data. These test tracts were excluded from that selection."
            )
        if not city.endswith("_sample"):
            st.caption(
                "A random tract split measures nearby-area estimation. "
                "It does not establish accuracy in a different city or future year."
            )

        spatial = load_spatial_metadata(path, details)
        spatial_metrics = spatial.get("model_metrics", {})
        if isinstance(spatial_metrics, dict):
            try:
                spatial_r2 = float(spatial_metrics["r2"])
                spatial_mae = float(spatial_metrics["mae"])
                spatial_rmse = float(spatial_metrics["rmse"])
                spatial_skill = float(spatial["mse_skill_vs_training_mean"])
                spatial_rows = int(spatial["rows"])
            except (KeyError, TypeError, ValueError):
                spatial_r2 = spatial_mae = spatial_rmse = spatial_skill = float("nan")
                spatial_rows = 0
            if (
                spatial_rows >= 2
                and np.isfinite([spatial_r2, spatial_mae, spatial_rmse, spatial_skill]).all()
            ):
                st.markdown("#### Separate geographic test")
                st.caption(
                    "Whole map areas are held out, with model settings selected only in "
                    "the remaining areas. These scores use a different test design from "
                    "the map and leaderboard above. Adjacent areas have no geographic buffer."
                )
                st.metric("Geographic R²", f"{spatial_r2:.3f}")
                st.write(
                    f"MAE: **{format_value(spatial_mae)}** · "
                    f"RMSE: **{format_value(spatial_rmse)}** · "
                    f"Tracts checked: **{spatial_rows:,}**"
                )
                if spatial_skill > 0:
                    st.write(
                        f"Squared error is **{spatial_skill:.1%} lower** than predicting "
                        "each training fold's average."
                    )
                else:
                    st.warning(
                        "In this geographic test, the model does not beat the training-average "
                        "baseline. Its map should be treated as exploratory."
                    )

with tab_lead:
    st.markdown("### Model scores")
    sub = m[(m.city == city) & (m.feature_set == feature_set) & (m.target == target)].copy()

    frames = {}
    comparison = None
    comparison_issue = None
    for _, r in sub.iterrows():
        try:
            dd = load_preds(str(r["path"]))
            for field in ("city", "model", "feature_set", "target"):
                if field in dd and not dd[field].eq(r[field]).all():
                    raise ValueError(f"Saved predictions disagree with the {field} selection.")
            frames[str(r["model"])] = dd
        except (OSError, ValueError, KeyError) as exc:
            comparison_issue = f"Cannot compare these saved results: {exc}"
            break

    lb = pd.DataFrame()
    if frames and not comparison_issue:
        try:
            comparison = align_model_predictions(frames)
        except ValueError as exc:
            comparison_issue = str(exc)
        else:
            lb = comparison.scores.rename(
                columns={"model": "Model", "r2": "R²", "mae": "MAE", "rmse": "RMSE", "n": "N"}
            )
            lb["Model"] = lb["Model"].map(fmt_model)
    if comparison_issue:
        st.warning(comparison_issue)
    if lb.empty:
        st.info("Model scores need predictions with observed values to compare against.")
    else:
        lb = lb.sort_values(["R²", "MAE"], ascending=[False, True])
        st.caption(
            f"Scores use the same {comparison.common_n:,} observed test tracts for every model."
        )
        if comparison.counts["discarded_rows"].sum():
            st.caption("Only observed tracts shared by every model are included in these scores.")
        st.dataframe(
            lb,
            width="stretch",
            hide_index=True,
            column_config={
                "R²": st.column_config.NumberColumn(format="%.6f"),
                "MAE": st.column_config.NumberColumn(
                    format="%.6f" if target.endswith("_rate") else "%.2f"
                ),
                "RMSE": st.column_config.NumberColumn(
                    format="%.6f" if target.endswith("_rate") else "%.2f"
                ),
            },
        )
        download_csv_button(
            lb,
            f"{city}_{feature_set}_{target}_leaderboard.csv",
            label="⬇️ Download leaderboard (CSV)",
        )

with tab_compare:
    st.markdown("### Compare models")
    if comparison is None or lb.empty:
        if comparison_issue:
            st.warning(comparison_issue)
        st.info("No model scores are available for this selection yet.")
    else:
        render_model_comparison(
            comparison,
            city=city,
            city_label=fmt_city(city),
            target=target,
            target_label=fmt_target(target),
            feature_set=feature_set,
            selected_model=model,
            model_labels=MODEL_LABELS,
        )

with tab_explain:
    st.markdown(local_explain(city, target, feature_set, model, df))

    st.markdown("---")
    st.markdown("### Common questions")
    with st.expander("What am I looking at?"):
        st.write(
            "Each dot is a small neighborhood area called a census tract. "
            "Choose Prediction, Ground truth, or Error to see an estimate, "
            "the data value, or the difference between them."
        )
    with st.expander("What does Error (pred − true) mean?"):
        st.write(
            "A positive error means the estimate is too high. A negative error means it is too low."
        )
    with st.expander("Which model should I trust?"):
        st.write(
            "Compare models tested on the same areas and data split. "
            "Compare their errors against the training-average baseline in Quick check. "
            "R² is not an accuracy percentage. Performance can vary by target and area; "
            "sample scores do not establish accuracy on real data."
        )
    with st.expander("Why do I only see dots and not polygons?"):
        st.write(
            "TerraLens uses one point at the center of each census tract "
            "to keep the map fast and easy to read."
        )
