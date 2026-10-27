# TerraLens

**Neighborhood estimates, made visible.**

TerraLens combines US Census indicators with measured OpenStreetMap features to estimate six socioeconomic measures across New York City, Chicago / Cook County, and San Francisco. Explore tract-level predictions, see where models miss, and compare four regression models in an interactive Streamlit dashboard.

[Get started](#get-started) · [Results](#results) · [Prepare real data](#prepare-real-data) · [Evaluation](#evaluation) · [Development](#development)

![XGBoost income prediction errors across census tracts in New York City, Chicago / Cook County, and San Francisco. Blue indicates underestimation and terracotta indicates overestimation.](docs/assets/income-residuals.png)

*Location makes a prediction error specific. Colored tracts show errors on held-out observations; pale tracts provide geographic context. Each map is scaled individually.*

## What you can explore

- **Maps:** switch between predicted values, published observations, and prediction errors.
- **Quick check:** compare the selected model with a training-mean baseline on the same test tracts.
- **Model comparisons:** inspect R², MAE, and RMSE, then compare two models' predictions on matching tracts and download the paired values.
- **Six measures:** median household income, median monthly gross rent, median home value, poverty rate, unemployment rate, and the share of adults with a bachelor's degree or higher.
- **Four models:** Ridge Regression, Extra Trees, Histogram Gradient Boosting, and XGBoost.

The sidebar model controls Map and Quick check. Compare includes all available models for the selected city and measure, using only shared observed tracts. Available choices follow the prediction files present on your machine.

## Results

These figures summarize saved OSM-only predictions using **2021 ACS labels and OSM extracts retrieved on 24–25 October 2026**. They are static snapshots; the dashboard reads your local results. The maps use XGBoost predictions from a random 20% holdout. Monetary errors use US dollars per year for income, dollars per month for rent, and dollars for home value. Rate errors are shown in percentage points. Each measure uses its own color scale.

### Compare models

![Income results for four models in New York City, Chicago / Cook County, and San Francisco: random holdout R², mean absolute error in US dollars per year, and geographic block validation R².](docs/assets/income-models.png)

*The income results tables, shown together in a single figure. Holdout and spatial validation use different test observations.*

![R² scores for Ridge Regression, Extra Trees, Histogram Gradient Boosting, and XGBoost across six measures and three study areas, evaluated on random held-out tracts.](docs/assets/model-comparison.png)

*Higher R² indicates better fit relative to the observed-value mean. R² is not an accuracy percentage and can be negative. Model scores are calculated on matching test observations.*

### Check geographic generalization

![XGBoost R² scores from random holdout and geographic block validation across six measures and three study areas.](docs/assets/validation-comparison.png)

*Geographic validation holds entire coordinate-grid cells out of training. It evaluates a different question from a random tract split; neither establishes transfer to another city or future year.*

### Understand the data coverage

![Number of valid observed tracts and random held-out test tracts for each measure across New York City, Chicago / Cook County, and San Francisco.](docs/assets/data-coverage.png)

*Coverage depends on the measure. Missing rent or home-value labels reduce the available observations; context polygons in the maps are not additional test predictions.*

<details>
<summary>Maps for every measure</summary>

Residual = prediction − published observation. Blue means the estimate is too low; terracotta means it is too high. Scales and units are printed on each figure.

### Median monthly gross rent

![Held-out XGBoost prediction errors for median monthly gross rent across New York City, Chicago / Cook County, and San Francisco, in USD per month.](docs/assets/rent-residuals.png)

### Median home value

![Held-out XGBoost prediction errors for median home value across the three study areas, in US dollars.](docs/assets/home-value-residuals.png)

### Poverty rate

![Held-out XGBoost prediction errors for the poverty rate across the three study areas, in percentage points.](docs/assets/poverty-residuals.png)

### Unemployment rate

![Held-out XGBoost prediction errors for the unemployment rate across the three study areas, in percentage points.](docs/assets/unemployment-residuals.png)

### Bachelor's degree or higher

![Held-out XGBoost prediction errors for the share of adults with a bachelor's degree or higher across the three study areas, in percentage points.](docs/assets/education-residuals.png)

</details>

The [model scores CSV](docs/assets/model-scores.csv), [geographic validation scores CSV](docs/assets/spatial-scores.csv), and [coverage CSV](docs/assets/coverage.csv) preserve the numerical results and tract counts. Score exports include MAE, RMSE, and their units. The [figure manifest](docs/assets/figure-manifest.json) records the source inputs. To regenerate all figures, first prepare predictions and run [geographic validation](#evaluation) for **all three cities**, then run:

```bash
python scripts/make_readme_figures.py
```

## Get started

Requires **Python 3.10 or newer**. From a terminal:

```bash
git clone https://github.com/DemonJ01/TerraLens.git
cd TerraLens
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open [localhost:8501](http://localhost:8501). For an existing checkout, open a terminal in its project folder and skip the clone and `cd` commands. If `.venv` already exists, skip its creation and start with activation. On Windows, activate with `.venv\Scripts\activate`. On macOS, XGBoost may also require the system OpenMP runtime. The map needs an internet connection to load OpenStreetMap tiles.

**A fresh checkout starts with clearly labeled synthetic NYC sample data.** Source datasets and prediction caches are excluded from Git. Sample values demonstrate the controls; their scores are not research results. Prepare a city below to replace the sample with real predictions. The dashboard reads saved predictions and does not train models.

## Prepare real data

With the virtual environment active, prepare each area you want to explore:

```bash
python scripts/prepare_city.py --city nyc --year 2021 --models ridge,extra_trees,hist_gb,xgboost
python scripts/prepare_city.py --city chicago --year 2021 --models ridge,extra_trees,hist_gb,xgboost
python scripts/prepare_city.py --city san_francisco --year 2021 --models ridge,extra_trees,hist_gb,xgboost
```

Run one command or all three. Each downloads Census tract boundaries, ACS labels, and an OSM extract; builds measured features; and trains models for every measure with sufficient observations. Existing results for other areas are retained. Choose **Reload results** in the dashboard when preparation finishes.

| City argument | Study area |
| --- | --- |
| `nyc` | New York City's five borough counties |
| `chicago` | Cook County, including Chicago and other communities |
| `san_francisco` | San Francisco County |

Features include POI categories, road classes and densities, building footprints, land-cover fractions, and tract shape. NYC uses a smaller BBBike OSM extract; the other areas use larger Illinois and Northern California extracts. Downloads are cached. The bulk Census route does not require an API key. OSM processing uses `osmium-tool` when available, otherwise Pyogrio's bundled GDAL OSM driver.

Add `--dry-run` to a preparation command to inspect its steps before downloading or training. Run `make data-check` to inspect local NYC data availability.

<details>
<summary>Reuse inputs, retrain models, or run the full workflow</summary>

To rebuild geographic features while reusing downloaded inputs:

```bash
python scripts/prepare_city.py --city nyc --year 2021 --models ridge,extra_trees,hist_gb,xgboost --rebuild-features --skip-downloads
```

To train from an existing processed dataset:

```bash
python scripts/cache_predictions.py --city nyc --models ridge,extra_trees,hist_gb,xgboost --feature_sets osm_only --targets all
```

This reads `data/processed/nyc_dataset.parquet` and writes predictions and `manifest.csv` under `outputs/preds/`. Substitute another city argument as needed. Use `--cv-folds` and `--n-jobs` to control validation folds and CPU threads.

The optional full workflow prepares all three areas, trains models, and generates experiment summaries, figures, and map exports:

```bash
bash scripts/run_everything.sh 2021
```

Use `bash scripts/run_everything.sh 2021 --skip-downloads` to reuse local inputs, or `bash scripts/run_everything.sh --help` for usage. NYC OSM + 311 comparisons require separately prepared complaint features; 311 data is optional for OSM-only results.

To use the Census API instead of bulk downloads, set `CENSUS_API_KEY` in your local environment. Keys are not written to result metadata.

</details>

## Evaluation

**Random holdout.** The prediction-cache workflow reserves 20% of valid observations for testing. Three-fold cross-validation within the remaining training data selects model settings. Imputation, scaling, Ridge feature selection, and nonlinear basis fitting occur within each training fold. Monetary targets can be fitted directly or on a log scale; selection uses validation errors in original dollar units. Test observations do not select settings.

Ridge uses a regularized linear regression over a selected linear, smooth geographic, or spline basis. The tree models retain their own model families. Census target columns and tract identifiers are excluded from predictors. Missing or invalid labels are excluded for that measure instead of replaced with zeros.

Metrics sidecars record feature names, selected settings, training-only validation results, train/test tract IDs, dataset and prediction hashes, and the training-mean baseline. The procedure follows the separation of training, validation, and testing described in the [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

**Geographic validation.** Run a separate nested evaluation that holds entire geographic blocks out of training:

```bash
python scripts/validate_models.py --city nyc --models ridge,extra_trees,hist_gb,xgboost --targets all --n-jobs 2
```

Repeat with `--city chicago` and `--city san_francisco` to generate the validation inputs for all README figures.

Results go to `outputs/spatial_validation/`. A four-by-four coordinate grid is defined without target values. Three outer folds evaluate held-out cells; three inner folds tune settings on the remaining cells. Adjacent cells have no geographic buffer. Quick check displays matching geographic validation separately when its dataset and prediction hashes agree; the dashboard map and leaderboard continue to use the random holdout.

### Reading the results responsibly

- **Source dates differ.** ACS 2021 covers 2017–2021; OSM reflects the extract's download date. These are retrospective estimates, not forecasts.
- **Published values have limits.** ACS figures are survey estimates and some medians are top-coded. Rent and home-value observations are unavailable for some tracts; those tracts are omitted for those measures.
- **Coverage varies.** Zero mapped OSM features do not prove a real-world feature is absent. Smaller areas and targets with fewer observations have less stable estimates.
- **Area estimates describe areas.** Census-tract estimates do not describe individual residents. Positive R² alone does not establish usefulness for a particular decision.

## Project layout

```text
app.py                       Streamlit dashboard
terralens/                   Shared data, evaluation, UI, and utilities
scripts/                     Data preparation, prediction caching, and visualizations
experiments/                 Training and experiment runners
configs/                     City and training settings
docs/assets/                 Versioned README figures
docs/maps/                   Local interactive map exports (generated)
data/                        Local source data and processed datasets (generated)
outputs/preds/               Local prediction cache (generated)
outputs/spatial_validation/  Local geographic validation reports (generated)
tests/                       Regression checks with synthetic fixtures
```

## Development

The install above includes development tools. Run:

```bash
make lint
make test
```

Tests cover dashboard loading and controls, matched model comparisons, training/test separation, geographic validation, data preparation, and map exports. They use small synthetic fixtures and mocked downloads, so no API key or city-data download is required. Optional historical map-export checks skip when their local files are absent.

## Data and license

Created by **Saatvik Rai**. Code is released under the [MIT License](LICENSE).

Data sources have their own terms: [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) (ODbL), [US Census American Community Survey](https://www.census.gov/programs-surveys/acs) (public domain), and optional [NYC Open Data 311 records](https://opendata.cityofnewyork.us/).
