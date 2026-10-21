"""Interactive maps for validated prediction rows."""

from html import escape

import folium
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template
from folium.plugins import HeatMap, MarkerCluster


class _MapSizing(MacroElement):
    """Recalculate Leaflet dimensions after a hidden tab becomes visible."""

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            const map = {{ this._parent.get_name() }};
            const container = map.getContainer();
            let fitted = false;
            function resizeMap() {
                const size = container.getBoundingClientRect();
                if (size.width > 0 && size.height > 0) {
                    map.invalidateSize({pan: false});
                    if (!fitted && {{ this.bounds | tojson }} !== null) {
                        map.fitBounds({{ this.bounds | tojson }}, {
                            padding: [24, 24], maxZoom: 14
                        });
                        fitted = true;
                    }
                }
            }
            if (typeof ResizeObserver !== 'undefined') {
                new ResizeObserver(resizeMap).observe(container);
            }
            window.addEventListener('resize', resizeMap);
            document.addEventListener('visibilitychange', resizeMap);
            requestAnimationFrame(resizeMap);
        })();
        {% endmacro %}
        """)

    def __init__(self, bounds):
        super().__init__()
        self._name = "TerraLensMapSizing"
        self.bounds = bounds


def build_prediction_map(df: pd.DataFrame, value_col: str, mode: str, clip: bool = True):
    d = df.copy()
    for col in ["lat", "lon", value_col]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", value_col])
    if d.empty:
        raise ValueError("No valid locations are available for this map layer.")

    center = [float(d["lat"].median()), float(d["lon"].median())]
    map_view = folium.Map(
        location=center,
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True,
        scroll_wheel_zoom=False,
    )
    values = d[value_col]
    low, high = values.quantile([0.02, 0.98]) if clip else (values.min(), values.max())
    if value_col == "error":
        limit = max(abs(float(low)), abs(float(high)), 1e-9)
        low, high = -limit, limit
        colors = ["#2563eb", "#f8fafc", "#dc2626"]
    else:
        colors = ["#2563eb", "#8b5cf6", "#f97316"]
        if low == high:
            low, high = low - 1, high + 1
    color_scale = LinearColormap(colors, vmin=low, vmax=high)

    if mode == "Heatmap":
        # Error heatmaps show size of misses; signed weights would hide negatives.
        if value_col == "error":
            weights = values.abs()
            ceiling = float(weights.quantile(0.98) if clip else weights.max())
            weights = weights.clip(upper=ceiling)
        else:
            weights = values.clip(lower=low, upper=high) - low
        denominator = float(weights.max()) or 1.0
        HeatMap(
            [
                [float(row.lat), float(row.lon), float(weight / denominator)]
                for row, weight in zip(d.itertuples(), weights)
                if weight > 0
            ],
            radius=18,
            blur=16,
            max_zoom=14,
        ).add_to(map_view)
    else:
        markers = MarkerCluster().add_to(map_view) if mode == "Cluster" else map_view
        for _, row in d.iterrows():
            popup = (
                f"<b>Area:</b> {escape(str(row.get('tract_id', '—')))}<br>"
                f"<b>Data value:</b> {row.get('y_true', np.nan):,.2f}<br>"
                f"<b>Estimate:</b> {row.get('y_pred', np.nan):,.2f}<br>"
                f"<b>Error:</b> {row.get('error', np.nan):,.2f}"
            )
            folium.CircleMarker(
                [float(row["lat"]), float(row["lon"])],
                radius=5,
                weight=1,
                color=color_scale(float(np.clip(row[value_col], low, high))),
                fill=True,
                fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=250),
                tooltip=escape(str(row.get("tract_id", "View area"))),
            ).add_to(markers)
    bounds = None
    if len(d) > 1 and (d["lat"].nunique() > 1 or d["lon"].nunique() > 1):
        bounds = [
            [float(d.lat.min()), float(d.lon.min())],
            [float(d.lat.max()), float(d.lon.max())],
        ]
    _MapSizing(bounds).add_to(map_view)
    return map_view
