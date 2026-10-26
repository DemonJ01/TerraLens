"""Interactive maps for validated prediction rows."""

from html import escape

import folium
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template
from folium.plugins import HeatMap, MarkerCluster


def _popup_value(value, *, rate: bool = False) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if not np.isfinite(number):
        return "Not available"
    return f"{number:,.6f}" if rate else f"{number:,.2f}"


class _MapSizing(MacroElement):
    """Keep hidden-tab maps sized correctly and offer a way back to the data."""

    _template = Template("""
        {% macro script(this, kwargs) %}
        (function () {
            const map = {{ this._parent.get_name() }};
            const container = map.getContainer();
            const heatLayer = {{ this.heat_layer.get_name() if this.heat_layer else 'null' }};
            const bounds = {{ this.bounds | tojson }};
            const initialCenter = map.getCenter();
            const initialZoom = map.getZoom();
            let fitted = false;
            function resetView() {
                map.invalidateSize({pan: false});
                if (bounds !== null) {
                    map.fitBounds(bounds, {padding: [24, 24], maxZoom: 14});
                } else {
                    map.setView(initialCenter, initialZoom);
                }
            }
            function resizeMap() {
                const size = container.getBoundingClientRect();
                if (size.width > 0 && size.height > 0) {
                    // Preserve the geographic center as the available width changes.
                    map.invalidateSize({animate: false});
                    if (!fitted) {
                        resetView();
                        fitted = true;
                    }
                    if (heatLayer !== null && !map.hasLayer(heatLayer)) {
                        heatLayer.addTo(map);
                    }
                } else if (heatLayer !== null && map.hasLayer(heatLayer)) {
                    map.removeLayer(heatLayer);
                }
            }
            const ResetView = L.Control.extend({
                options: {position: 'topright'},
                onAdd: function () {
                    const control = L.DomUtil.create('div', 'leaflet-bar');
                    const button = L.DomUtil.create('button', '', control);
                    button.type = 'button';
                    button.textContent = 'Reset view';
                    button.title = 'Show all mapped areas';
                    button.style.cssText = 'background:white;color:#222;border:0;' +
                        'padding:6px 10px;cursor:pointer;font:14px/20px sans-serif';
                    L.DomEvent.disableClickPropagation(control);
                    L.DomEvent.disableScrollPropagation(control);
                    L.DomEvent.on(button, 'click', function (event) {
                        L.DomEvent.stop(event);
                        resetView();
                    });
                    return control;
                }
            });
            new ResetView().addTo(map);
            if (typeof ResizeObserver !== 'undefined') {
                new ResizeObserver(resizeMap).observe(container);
            }
            window.addEventListener('resize', resizeMap);
            window.addEventListener('pageshow', resizeMap);
            document.addEventListener('visibilitychange', resizeMap);
            requestAnimationFrame(resizeMap);
        })();
        {% endmacro %}
        """)

    def __init__(self, bounds, heat_layer=None):
        super().__init__()
        self._name = "TerraLensMapSizing"
        self.bounds = bounds
        self.heat_layer = heat_layer


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
        scroll_wheel_zoom=True,
        # The visibility-aware observer below handles resizing, including hidden iframes.
        track_resize=False,
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

    heat_layer = None
    if mode == "Heatmap":
        # Error heatmaps show size of misses; signed weights would hide negatives.
        if value_col == "error":
            weights = values.abs()
            ceiling = float(weights.quantile(0.98) if clip else weights.max())
            weights = weights.clip(upper=ceiling)
        else:
            weights = values.clip(lower=low, upper=high) - low
        denominator = float(weights.max()) or 1.0
        heat_layer = HeatMap(
            [
                [float(row.lat), float(row.lon), float(weight / denominator)]
                for row, weight in zip(d.itertuples(), weights)
                if weight > 0
            ],
            radius=18,
            blur=16,
            max_zoom=14,
            # Leaflet.heat cannot draw a zero-width canvas in an inactive tab.
            show=False,
        ).add_to(map_view)
    else:
        markers = MarkerCluster().add_to(map_view) if mode == "Cluster" else map_view
        for _, row in d.iterrows():
            rate = str(row.get("target", "")).endswith("_rate")
            units = "<b>Units:</b> rate (0–1 scale)<br>" if rate else ""
            popup = (
                f"<b>Area:</b> {escape(str(row.get('tract_id', '—')))}<br>"
                f"{units}"
                f"<b>Data value:</b> {_popup_value(row.get('y_true', np.nan), rate=rate)}<br>"
                f"<b>Estimate:</b> {_popup_value(row.get('y_pred', np.nan), rate=rate)}<br>"
                f"<b>Error:</b> {_popup_value(row.get('error', np.nan), rate=rate)}"
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
    _MapSizing(bounds, heat_layer).add_to(map_view)
    return map_view
