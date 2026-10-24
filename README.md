# TerraLens

TerraLens explores neighborhood estimates from open city data. The dashboard includes an interactive map, a quick accuracy check, model rankings, comparisons, and simple explanations.

## Run the dashboard

Open a terminal in this project folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make run
```

Then open [TerraLens](http://localhost:8501). If `.venv` already exists, skip the first command. On Windows, activate it with `.venv\Scripts\activate` and launch with `python -m streamlit run app.py`.

When real predictions are absent, the dashboard shows clearly labeled synthetic sample data. Sample values demonstrate the controls; their accuracy scores are not research results. Available cities, measures, feature sets, and models follow the prediction files that are present. The map needs an internet connection to load OpenStreetMap tiles.

## Use your own results

The training scripts support NYC, Chicago, and San Francisco, with these measures when source labels are available:

- Household income, rent, and home value
- Poverty and unemployment rates
- Share of adults with a bachelor's degree or higher

For a prepared dataset, train models and cache their predictions with:

```bash
.venv/bin/python scripts/cache_predictions.py --city nyc --models ridge,extra_trees,hist_gb --feature_sets osm_only --targets all
```

This reads `data/processed/nyc_dataset.parquet` and writes predictions plus `outputs/preds/manifest.csv`. Use `chicago` or `san_francisco` for another prepared city. XGBoost is also supported with `--models xgboost`; on macOS it may require the system OpenMP runtime.

Large source datasets and generated predictions are kept locally and excluded from Git. To prepare real NYC results on a fresh checkout, run:

```bash
.venv/bin/python scripts/prepare_city.py --city nyc --year 2021
```

This downloads official Census tract boundaries and ACS bulk data, plus a smaller NYC OpenStreetMap extract from BBBike. It calculates measured road lengths and POI counts, then trains three models for all six measures. Downloads are roughly 320 MB for NYC and are reused on subsequent runs. No Census API key is needed for the bulk-data route. To use the Census API instead, set `CENSUS_API_KEY` in your local environment; keys are never written to result metadata.

OSM processing uses `osmium-tool` if installed, otherwise Pyogrio's bundled GDAL OSM driver. The map displays a random 20% held-out sample, not training predictions. ACS 2021 covers 2017–2021; the OSM extract is current at download time, so the sources are not time-aligned and scores are retrospective, not forecasts. Some tracts lack rent or home-value observations and are omitted for those measures.

Run `make data-check` to inspect local availability. The optional full three-city workflow downloads larger state OSM extracts for the other cities:

```bash
bash scripts/run_everything.sh 2021
```

To reuse existing raw data and features, run `bash scripts/run_everything.sh 2021 --skip-downloads`. Use `bash scripts/run_everything.sh --help` to see the prerequisites. This workflow prepares three cities, trains models, caches predictions, and generates experiment summaries, figures, and maps. NYC 311 comparisons require separately prepared complaint features.

The dashboard reads saved predictions and does not train models. On a fresh checkout it creates its sample files locally. Choose **Reload results** after generating real predictions; real results replace sample choices automatically. NYC 311 data is optional and is not required for OSM-only results.

## Project layout

```text
app.py                 Streamlit dashboard
terralens/             Shared evaluation and utilities
scripts/               Data preparation, prediction caching, and visualizations
experiments/           Training and experiment runners
configs/               City and training settings
docs/                  Local map and figure exports (generated)
outputs/preds/         Local prediction cache (generated)
```

## Data and license

TerraLens is released under the [MIT License](LICENSE). Author: Saatvik Rai.

Real-data experiments use OpenStreetMap contributors (ODbL), US Census ACS (public domain), and NYC Open Data 311 records. Census tracts represent areas; estimates describe those areas rather than individual residents.
