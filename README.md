# TerraLens

TerraLens explores neighborhood estimates from open city data. The dashboard includes an interactive map, a quick accuracy check, model rankings, comparisons, and simple explanations.

**Compare** ranks all available models for the chosen city and target, with precise R², MAE, and RMSE values. Choose **Model A** and **Model B** to inspect their actual prediction gaps on matching tracts and download the paired values. Similar overall scores can come from different tract-level estimates. The sidebar model controls Map and Quick check; the comparison includes all models for that city and target. Models are compared only on shared observed tracts, and conflicting observed values or tract identities are rejected.

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

The dashboard lists cities with valid prediction files in `outputs/preds/`. Each city must be prepared separately. Preparing NYC makes NYC available; Chicago and San Francisco appear after their own preparation commands finish and you choose **Reload results**. Source datasets and generated predictions are excluded from Git, so a fresh checkout starts with labeled NYC sample data.

Data preparation is configured for these areas:

- `nyc`: New York City's five borough counties
- `chicago`: Cook County tracts, covering Chicago and other Cook County communities
- `san_francisco`: San Francisco County

Each prepared area can expose these measures when sufficient source labels are available:

- Household income, rent, and home value
- Poverty and unemployment rates
- Share of adults with a bachelor's degree or higher

For a prepared dataset, train models and cache their predictions with:

```bash
.venv/bin/python scripts/cache_predictions.py --city nyc --models ridge,extra_trees,hist_gb,xgboost --feature_sets osm_only --targets all
```

This reads `data/processed/nyc_dataset.parquet` and writes predictions plus `outputs/preds/manifest.csv`. Use `chicago` or `san_francisco` for another prepared area. These commands select Ridge Regression, Extra Trees, Histogram Gradient Boosting, and XGBoost. XGBoost on macOS may require the system OpenMP runtime.

Large source datasets and generated predictions are kept locally and excluded from Git. To prepare real NYC results on a fresh checkout, run:

```bash
.venv/bin/python scripts/prepare_city.py --city nyc --year 2021 --models ridge,extra_trees,hist_gb,xgboost
```

To add Chicago/Cook County and San Francisco, run their preparation commands too:

```bash
.venv/bin/python scripts/prepare_city.py --city chicago --year 2021 --models ridge,extra_trees,hist_gb,xgboost
.venv/bin/python scripts/prepare_city.py --city san_francisco --year 2021 --models ridge,extra_trees,hist_gb,xgboost
```

Each command downloads official Census tract boundaries, ACS labels, and an OpenStreetMap extract; builds measured geographic features; and trains the selected models for each measure with sufficient observations. Features include POI categories, road classes and densities, building footprints, land-cover fractions, and tract shape. Existing predictions for other cities are retained. After completion, choose **Reload results** to refresh the City menu.

To replace older cached geographic features while reusing downloaded inputs, add `--rebuild-features --skip-downloads` to a preparation command. For example:

```bash
.venv/bin/python scripts/prepare_city.py --city nyc --year 2021 --models ridge,extra_trees,hist_gb,xgboost --rebuild-features --skip-downloads
```

NYC uses a smaller BBBike extract, with roughly 320 MB of downloads. Chicago/Cook County uses the larger Illinois extract, and San Francisco uses the Northern California extract. Downloads are reused on subsequent runs. No Census API key is needed for the bulk-data route. To use the Census API instead, set `CENSUS_API_KEY` in your local environment; keys are never written to result metadata. Add `--dry-run` to a preparation command to inspect its steps without downloading or training.

OSM processing uses `osmium-tool` if installed, otherwise Pyogrio's bundled GDAL OSM driver. The map displays a random 20% held-out sample, not training predictions. ACS 2021 covers 2017–2021; the OSM extract is current at download time, so the sources are not time-aligned and scores are retrospective, not forecasts. Some tracts lack rent or home-value observations and are omitted for those measures.

Run `make data-check` to inspect local availability. The optional full three-city workflow downloads larger state OSM extracts for the other cities:

```bash
bash scripts/run_everything.sh 2021
```

To reuse existing raw data and features, run `bash scripts/run_everything.sh 2021 --skip-downloads`. Use `bash scripts/run_everything.sh --help` to see the prerequisites. This workflow prepares three cities, trains models, caches predictions, and generates experiment summaries, figures, and maps. NYC 311 comparisons require separately prepared complaint features.

The dashboard reads saved predictions and does not train models. On a fresh checkout it creates its sample files locally. Choose **Reload results** after generating real predictions; real results replace sample choices automatically. NYC 311 data is optional and is not required for OSM-only results.

## Model training and evaluation

The prediction-cache workflow reserves 20% of valid observations for testing. Model settings are selected with three-fold cross-validation inside the remaining training data. Imputation, scaling, Ridge feature selection and nonlinear basis fitting are repeated within each training fold. Monetary targets can be fitted directly or on a log scale; the choice is selected using cross-validation error in original dollar units. The test observations do not select model settings. Use `--cv-folds` and `--n-jobs` with `cache_predictions.py` to control validation folds and CPU threads.

Ridge remains a regularized linear regression over an explicitly transformed feature basis, with linear, smooth geographic, and spline options. Tree models retain their own model families. Census target columns and tract identifiers are excluded from predictors, and invalid or missing target labels are excluded for that target rather than replaced with zeros.

Each metrics sidecar records the input feature names, selected settings, training-only validation results, train/test tract identifiers, data and prediction hashes, and a baseline that predicts the training-data mean. **Quick check** compares the selected model with that baseline on the same test tracts. R² is not an accuracy percentage; positive R² alone does not establish that an estimate is useful for a particular decision. The selection procedure follows the separation of training, validation, and testing described in the [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

Random tract splits measure estimates within the prepared area. Run separate nested geographic block validation with:

```bash
.venv/bin/python scripts/validate_models.py --city nyc --models ridge,extra_trees,hist_gb,xgboost --targets all --n-jobs 2
```

This writes out-of-fold predictions and metrics under `outputs/spatial_validation/`. The four-by-four coordinate grid is defined without target values. Three outer folds hold whole grid cells out of training; three inner folds tune settings using only the remaining cells. Adjacent cells have no geographic buffer, and these scores do not establish transfer to another city or future year. **Quick check** shows matching geographic validation separately when its dataset and prediction hashes agree. The map and leaderboard continue to display the random holdout, rather than mixing these evaluation schemes.

OSM mapping completeness varies, so zero mapped features do not prove that a real-world feature is absent. ACS medians may be top-coded and are survey estimates; dollar errors are measured against the published values. Smaller areas and targets with fewer valid observations have less stable estimates. Generated predictions and validation reports remain local.

## Project layout

```text
app.py                 Streamlit dashboard
terralens/             Shared evaluation and utilities
scripts/               Data preparation, prediction caching, and visualizations
experiments/           Training and experiment runners
configs/               City and training settings
docs/                  Local map and figure exports (generated)
outputs/preds/         Local prediction cache (generated)
tests/                 Automated regression checks with synthetic fixtures
```

## Check changes

The development install above includes the test runner. Run:

```bash
make lint
make test
```

The tests cover dashboard loading and controls, matched model comparisons, training/test
separation, geographic validation, data preparation, and map exports. They use small synthetic
fixtures and mocked downloads, so no API key or city-data download is required. Two optional
checks of historical map exports skip when those local files are absent.

## Data and license

TerraLens is released under the [MIT License](LICENSE). Author: Saatvik Rai.

Real-data experiments use OpenStreetMap contributors (ODbL), US Census ACS (public domain), and NYC Open Data 311 records. Census tracts represent areas; estimates describe those areas rather than individual residents.
