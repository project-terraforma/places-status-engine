# places-status-engine

Alexander Lio

This project predicts whether a business or place is **open or closed** using Overture Maps data, URL liveness checks, and OpenStreetMap disused/abandoned tags.
The pipeline combines:

- Overture Places data via DuckDB S3 queries
- Foursquare-matched ground truth labels for training
- URL liveness checking (HEAD/GET with fallback)
- OpenStreetMap disused/abandoned tag matching via Overpass
- XGBoost classifier with target encoding and label cleaning

## Pipeline Structure

- src/
   - `pipeline.py`            -- End-to-end runner (train, query, check URLs, check OSM, predict)
   - `hybrid.py`              -- Feature engineering and labeled data loading
   - `train_xgb.py`           -- XGBoost model training, cross-validation, evaluation

   - utils/
      - `places_util.py`       -- Overture S3 client, bounding box helpers, DuckDB connection
      - `schema_live_places.py` -- Raw Overture data processing into clean features
      - `url_checker.py`       -- URL liveness checker with caching
      - `osm_checker.py`       -- OpenStreetMap disused/abandoned tag spatial join

- artifacts/
   - `fsq_api.py`             -- Foursquare API labeling script (generates training labels)

- assets/
   - `{city}_places_processed.parquet`  -- Processed Overture features per city
   - `{city}_places_labeled.parquet`    -- Foursquare-matched labels per city

- cache/
   - `url_status.json`                 -- Cached URL liveness results
   - `osm_features.parquet`            -- Cached OSM disused/abandoned spatial join
   - `brand_location_counts.json`      -- Brand location counts across Overture data

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/project-terraforma/places-status-engine
cd places-status-engine
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment variables
Create a `.env` file inside `src/`:
```bash
FSQ_API_KEY=YOUR_FOURSQUARE_KEY_HERE
```
The Foursquare key is only needed for generating new training labels via `artifacts/fsq_api.py`. The prediction pipeline does not require it.

## Running the pipeline

To run the pipeline, use `pipeline.py` inside `src/`.

Running `pipeline.py` performs the full workflow:

1. **Train model**
   Loads labeled data from all available cities, builds features, trains XGBoost with label cleaning.

2. **Query Overture Places**
   Fetches live Overture data for the target area via DuckDB S3 queries.

3. **Check URL liveness**
   Runs HEAD/GET requests against business websites (50 concurrent threads).

4. **Query OSM disused tags**
   Checks OpenStreetMap for disused/abandoned tags near each place via Overpass API.

5. **Generate predictions**
   Runs the trained model on the queried places and saves results.

Example usage:

```bash
cd src

# Predict for a point (lat/lon + radius in meters)
python pipeline.py --lat 37.780 --lon -122.409 --radius 500

# Predict for a bounding box
python pipeline.py --bbox "-122.42,37.77,-122.40,37.79"

# With custom output path
python pipeline.py --lat 37.780 --lon -122.409 --radius 500 --output outputs/my_predictions.csv

# With evaluation against manual labels
python pipeline.py --lat 37.780 --lon -122.409 --radius 500 --eval outputs/manual_labels.csv
```

## Notes on arguments

- `--lat`, `--lon` -- Center point coordinates
- `--radius` -- Search radius in meters (default: 500)
- `--bbox` -- Bounding box as `xmin,ymin,xmax,ymax` (overrides lat/lon)
- `--output` -- Output CSV path (default: `outputs/predictions.csv`)
- `--eval` -- Path to manual labels CSV for evaluation against predictions

## e.g.

```bash
python pipeline.py --bbox="-122.095,37.385,-122.06,37.4" --output=outputs/dtmv_pipeline_predictions.csv --eval=../docs/dtmv_labels.csv
```
## Output structure

Each run produces two files:

- `outputs/predictions.csv` -- Human-readable results sorted by closure probability
   - Columns: `name`, `address`, `category`, `P_closed`, `google_url`, `actually_closed`
   - `actually_closed` is blank for manual annotation

- `outputs/predictions_debug.csv` -- Full feature dump with model inputs and predictions
   - Includes raw fields, all engineered features, probability, and prediction

## Training and evaluation

To run cross-validation and threshold analysis on labeled data:

```bash
cd src
python train_xgb.py
```

This runs 5-fold stratified CV and prints precision, recall, F1, and PR-AUC at multiple thresholds, plus per-city breakdowns.

## Other utilities

```bash
# Build URL liveness cache for all labeled data
cd src/utils
python url_checker.py

# Build OSM disused/abandoned feature cache
python osm_checker.py

# Fetch and process raw Overture data for a city
python schema_live_places.py --city sf
```
