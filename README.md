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

The dashboard works immediately with clearly labeled synthetic sample data when real predictions are absent. Sample values demonstrate the controls; their accuracy scores are not research results. Available cities, measures, feature sets, and models follow the prediction files that are present. The map needs an internet connection to load OpenStreetMap tiles.

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

Source data and model outputs are excluded from this project copy. Run `make data-check` to inspect local data availability. The full real-data workflow requires the external `osmium` command (`osmium-tool`) and downloads large state OSM extracts, Census tract geometries, and ACS labels:

```bash
bash scripts/run_everything.sh 2021
```

To reuse existing raw data and features, run `bash scripts/run_everything.sh 2021 --skip-downloads`. Use `bash scripts/run_everything.sh --help` to see the prerequisites. This workflow prepares three cities, trains models, caches predictions, and generates experiment summaries, figures, and maps. NYC 311 comparisons require separately prepared complaint features.

The dashboard reads saved predictions and does not train models. On a fresh checkout it creates its sample files locally. Reopen the dashboard after generating or changing real predictions.

## Check the project

```bash
make test        # Automated regression checks
make smoke       # Small synthetic dataset and Ridge training; no downloads
make data-check  # Availability of real NYC data; reports missing inputs
```

The synthetic training check writes to `outputs/smoke/`. Missing real data is reported separately from dashboard or training-code failures. `make` uses `.venv` automatically, with `venv` and `python3` as fallbacks.

## Project layout

```text
app.py                 Streamlit dashboard
terralens/             Shared evaluation and utilities
scripts/               Data preparation, prediction caching, and visualizations
experiments/           Training and experiment runners
configs/               City and training settings
tests/                 Automated checks
docs/                  Saved maps and figures
outputs/preds/         Local prediction cache (generated)
```

## Data and license

TerraLens is released under the [MIT License](LICENSE). Author: Saatvik Rai.

Real-data experiments use OpenStreetMap contributors (ODbL), US Census ACS (public domain), and NYC Open Data 311 records. Census tracts represent areas; estimates describe those areas rather than individual residents.
