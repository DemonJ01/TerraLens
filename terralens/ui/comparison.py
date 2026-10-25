"""Display model scores and actual prediction differences on matched tracts."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from terralens.data.comparison import ModelComparison, paired_predictions

MODEL_COLORS = {
    "ridge": "#a78bfa",
    "extra_trees": "#34d399",
    "hist_gb": "#fbbf24",
    "xgboost": "#60a5fa",
}


def score_chart(scores: pd.DataFrame, metric: str, model_labels: dict, rate: bool):
    """Keep a zero reference and print the measured values beside each model."""
    table = scores.copy()
    key = {"R²": "r2", "MAE": "mae", "RMSE": "rmse"}[metric]
    table = table.loc[np.isfinite(table[key])].copy()
    ascending = metric != "R²"
    table = table.sort_values(key, ascending=ascending)
    table["Model"] = table["model"].map(lambda name: model_labels.get(name, name))
    multiplier = 100 if rate and metric != "R²" else 1
    table["Value"] = table[key] * multiplier
    table["Label"] = table["Value"].map(
        lambda value: f"{value:.6f}" if metric == "R²" else f"{value:,.4f}"
    )
    unit = " (percentage points)" if multiplier == 100 else " (USD)" if metric != "R²" else ""
    order = table["Model"].tolist()
    colors = [MODEL_COLORS.get(name, "#f472b6") for name in table["model"]]
    low, high = min(0.0, table["Value"].min()), max(0.0, table["Value"].max())
    label_room = max(high - low, abs(high), 0.001) * 0.35
    base = alt.Chart(table).encode(
        x=alt.X(
            "Value:Q",
            title=metric + unit,
            scale=alt.Scale(zero=True, domain=[low, high + label_room]),
        ),
        y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=190, labelFontSize=11)),
        tooltip=[
            alt.Tooltip("Model:N"),
            alt.Tooltip("Value:Q", title=metric + unit, format=".6f"),
            alt.Tooltip("n:Q", title="Same test tracts", format=",d"),
        ],
    )
    points = base.mark_point(filled=True, size=150).encode(
        color=alt.Color("Model:N", scale=alt.Scale(domain=order, range=colors), legend=None)
    )
    labels = base.mark_text(align="left", dx=12, fontSize=12).encode(text="Label:N")
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#64748b").encode(x="zero:Q")
    return (
        (zero + points + labels)
        .properties(height=max(150, 48 * len(table)))
        .configure_view(stroke=None)
    )


def difference_chart(pair: pd.DataFrame, *, rate: bool, model_a: str, model_b: str):
    """Show signed prediction gaps; the zero line means the models agree."""
    table = pair.copy()
    table["Gap"] = table["prediction_difference"] * (100 if rate else 1)
    table["Magnitude"] = table["Gap"].abs()
    table = table.sort_values(["Magnitude", "tract_id"], ascending=[False, True]).head(12)
    table["Direction"] = np.where(table["Gap"] >= 0, "A predicts higher", "B predicts higher")
    unit = "percentage points" if rate else "USD"
    base = alt.Chart(table).encode(
        y=alt.Y("tract_id:N", title="Tract", sort=table["tract_id"].tolist()),
        x=alt.X("Gap:Q", title=f"A − B ({unit})", scale=alt.Scale(zero=True)),
        tooltip=[
            alt.Tooltip("tract_id:N", title="Tract"),
            alt.Tooltip("observed:Q", title="Observed", format=",.6f"),
            alt.Tooltip("prediction_a:Q", title=model_a, format=",.6f"),
            alt.Tooltip("prediction_b:Q", title=model_b, format=",.6f"),
            alt.Tooltip("Gap:Q", title=f"A − B ({unit})", format=",.4f"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color(
            "Direction:N",
            scale=alt.Scale(
                domain=["A predicts higher", "B predicts higher"], range=["#60a5fa", "#fbbf24"]
            ),
            legend=alt.Legend(title=None, orient="top"),
        )
    )
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#cbd5e1").encode(x="zero:Q")
    return (bars + zero).properties(height=max(100, 27 * len(table))).configure_view(stroke=None)


def render_model_comparison(
    comparison: ModelComparison,
    *,
    city: str,
    city_label: str,
    target: str,
    target_label: str,
    feature_set: str,
    selected_model: str,
    model_labels: dict,
):
    st.markdown(f"**{city_label} · {target_label}**")
    st.caption(
        f"All {len(comparison.scores)} models are scored on the same {comparison.common_n:,} test tracts. "
        "The model selected above controls Map and Quick check. Choose two models below "
        "to inspect their predictions directly."
    )
    if comparison.counts["discarded_rows"].sum():
        st.info(
            "Comparison uses only test tracts with observed values and predictions shared by "
            "every model. Scores here may differ from the individual map's scores."
        )
    rate = target.endswith("_rate")
    metric = st.selectbox("Score to compare", ["R²", "MAE", "RMSE"], key="comparison_metric")
    key = {"R²": "r2", "MAE": "mae", "RMSE": "rmse"}[metric]
    scores = comparison.scores.sort_values(key, ascending=metric != "R²", na_position="last").copy()
    finite = scores.loc[np.isfinite(scores[key])]
    if finite.empty:
        st.info(
            "R² requires multiple different observed values. Choose MAE or RMSE to compare errors."
        )
    else:
        leader = finite.iloc[0]
        value = float(leader[key])
        formatted = (
            f"{value:.6f}"
            if metric == "R²"
            else f"{value * 100:,.4f} pp" if rate else f"${value:,.2f}"
        )
        direction = "Higher is better" if metric == "R²" else "Lower is better"
        st.markdown(
            f"**Lowest error: {model_labels.get(leader['model'], leader['model'])}**"
            if metric != "R²"
            else f"**Highest R²: {model_labels.get(leader['model'], leader['model'])}**"
        )
        st.caption(f"{metric} = {formatted}. {direction}. R² is not an accuracy percentage.")
        st.altair_chart(score_chart(scores, metric, model_labels, rate), width="stretch")
    table = scores.rename(
        columns={"model": "Model", "r2": "R²", "mae": "MAE", "rmse": "RMSE", "n": "N"}
    )
    table["Model"] = table["Model"].map(lambda name: model_labels.get(name, name))
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "R²": st.column_config.NumberColumn(format="%.6f"),
            "MAE": st.column_config.NumberColumn(format="%.6f" if rate else "%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.6f" if rate else "%.2f"),
        },
    )
    if rate:
        st.caption(
            "The table keeps rates on the 0–1 scale. Error charts use percentage points (0.01 = 1 pp)."
        )
    st.caption(
        "Close scores can come from different predictions. The comparison below shows their actual differences."
    )
    if len(scores) < 2:
        st.info("At least two models are needed for a tract-by-tract comparison.")
        return

    st.markdown("#### Compare two models")
    available = sorted(comparison.frames)
    default_a = selected_model if selected_model in available else available[0]
    scope = f"{city_label}:{feature_set}:{target}:{selected_model}"
    a = st.selectbox(
        "Model A",
        available,
        index=available.index(default_a),
        format_func=lambda name: model_labels.get(name, name),
        key=f"comparison_a:{scope}",
    )
    others = [name for name in available if name != a]
    ranked = comparison.scores.sort_values(
        ["r2", "mae"], ascending=[False, True], na_position="last"
    )
    default_b = next(name for name in ranked["model"] if name != a)
    b = st.selectbox(
        "Model B",
        others,
        index=others.index(default_b),
        format_func=lambda name: model_labels.get(name, name),
        key=f"comparison_b:{scope}:{a}",
    )
    pair = paired_predictions(comparison, a, b)
    mean_gap = float(pair["prediction_difference"].abs().mean())
    equal = bool(np.array_equal(pair["prediction_a"].to_numpy(), pair["prediction_b"].to_numpy()))
    if equal:
        st.warning("These two saved models have identical predictions on the compared tracts.")
    else:
        gap = f"{mean_gap * 100:.4f} percentage points" if rate else f"${mean_gap:,.2f}"
        st.markdown(f"**Average prediction difference: {gap}** across {len(pair):,} tracts.")
    error_a, error_b = pair["error_a"].abs(), pair["error_b"].abs()
    ties = np.isclose(error_a, error_b, rtol=1e-10, atol=1e-12)
    wins_a = int(((error_a < error_b) & ~ties).sum())
    wins_b = int(((error_b < error_a) & ~ties).sum())
    st.caption(
        f"A has a smaller absolute error on {wins_a:,} tracts; B on {wins_b:,}; tied on {int(ties.sum()):,}."
    )
    if not equal:
        st.altair_chart(
            difference_chart(
                pair, rate=rate, model_a=model_labels.get(a, a), model_b=model_labels.get(b, b)
            ),
            width="stretch",
        )
        st.caption(
            f"The {min(12, len(pair))} largest prediction gaps are shown. Positive means A predicts higher than B."
        )
    with st.expander("Observed values and predictions for every compared tract"):
        display = pair[
            [
                "tract_id",
                "observed",
                "prediction_a",
                "prediction_b",
                "prediction_difference",
                "error_a",
                "error_b",
            ]
        ].rename(
            columns={
                "tract_id": "Tract",
                "observed": "Observed",
                "prediction_a": "A prediction",
                "prediction_b": "B prediction",
                "prediction_difference": "A − B",
                "error_a": "A error",
                "error_b": "B error",
            }
        )
        st.caption(
            f"A: {model_labels.get(a, a)} · B: {model_labels.get(b, b)}. Values are in the target's original units."
        )
        st.caption(
            "Observed values and tract locations stay the same across models; predictions and errors can differ."
        )
        st.dataframe(display, hide_index=True, width="stretch")
        st.download_button(
            "Download paired predictions (CSV)",
            pair.assign(city=city, feature_set=feature_set, target=target, model_a=a, model_b=b)
            .to_csv(index=False)
            .encode("utf-8"),
            file_name=f"{city}_{feature_set}_{target}_{a}_vs_{b}.csv",
            mime="text/csv",
            on_click="ignore",
        )
