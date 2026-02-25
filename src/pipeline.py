import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to python path so we can import local modules
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from hybrid import get_data, build_features
from train_xgb import train_model, encode_sources
from utils.places_util import OverturePlaces, radius_to_bbox
from utils.url_checker import check_url
from utils.schema_live_places import process_places
from utils.osm_checker import query_osm_for_places


def main():
    parser = argparse.ArgumentParser(description="Places Status Engine - End-to-End Prediction Pipeline")
    parser.add_argument("--lat", type=float, help="Latitude of the center point")
    parser.add_argument("--lon", type=float, help="Longitude of the center point")
    parser.add_argument("--radius", type=float, default=500, help="Search radius in meters (default: 500)")
    parser.add_argument("--bbox", type=str, help="Bounding box as xmin,ymin,xmax,ymax. Overrides lat/lon.")
    parser.add_argument("--output", type=str, default="outputs/predictions.csv", help="Output CSV file path")
    parser.add_argument("--eval", type=str, default=None, help="Path to manual labels CSV to evaluate predictions against")
    args = parser.parse_args()

    if not args.bbox and (args.lat is None or args.lon is None):
        print("Error: Must provide either --bbox or both --lat and --lon")
        sys.exit(1)

    # 1. Train Model
    print("1. Training Model on Labeled Data")

    # Load labeled data
    X_train, y_train, weights_train, _ = get_data()
    # Train the model (bypassing the evaluation parts of train_xgb)
    clf = train_model(X_train, y_train, weights_train)

    # 2. Query Raw Data
    print("2. Querying Overture Places Data")
    client = OverturePlaces()
    
    if args.bbox:
        xmin, ymin, xmax, ymax = map(float, args.bbox.split(","))
        bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        print(f"Using bounding box: {bbox}")
    else:
        bbox = radius_to_bbox(args.lat, args.lon, args.radius)
        print(f"Using point ({args.lat}, {args.lon}) with radius {args.radius}m")
        print(f"Calculated bounding box: {bbox}")
        
    df_raw = client.query_bbox(bbox, include_all_fields=True)
    if df_raw.empty:
        print("No places found in the specified area. Exiting.")
        sys.exit(0)
        
    print("\nProcessing raw Overture data")
    df_clean = process_places(df_raw)

    # 3. Liveliness Check
    print("3. Running URL Liveliness Check on Queried Places")
    
    # Filter to places with a non-empty website
    has_url = df_clean[df_clean['website'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    print(f"Places with valid URLs to check: {len(has_url)}")
    
    url_results = {}
    import concurrent.futures
    from tqdm import tqdm
    
    if len(has_url) > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = {
                executor.submit(check_url, row['website']): row['id']
                for _, row in has_url.iterrows()
            }
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Checking URLs"):
                place_id = futures[future]
                url_results[place_id] = future.result()  # dict: {"alive": bool, "status_code": int}

    # Inject URL results directly so build_features picks them up via the cache-read path.
    # Store alive and status_code separately so derived features (url_is_4xx, url_is_error) work.
    def _r_alive(v) -> float:
        if v is None:
            return float('nan')
        return float(v.get('alive', False) if isinstance(v, dict) else bool(v))

    def _r_code(v) -> float:
        if v is None:
            return float('nan')
        if isinstance(v, dict):
            return float(v.get('status_code', -1))
        return 200.0 if v else -1.0

    df_clean['url_alive'] = df_clean['id'].map({k: _r_alive(v) for k, v in url_results.items()}).astype(float)
    df_clean['url_status_code'] = df_clean['id'].map({k: _r_code(v) for k, v in url_results.items()}).astype(float)
    
    # 4. Predict
    print("4. Generating Predictions")

    # Query OSM live for this area so osm_disused/osm_dist_m are real at inference time
    try:
        df_clean = query_osm_for_places(df_clean)
    except Exception as e:
        print(f"  [WARN] OSM query failed, defaulting to 0: {e}")
        df_clean['osm_disused'] = 0
        df_clean['osm_dist_m'] = float('nan')

    # Apply standard feature building
    X_inference = build_features(df_clean)
    X_inference = encode_sources(X_inference)
    
    # Ensure all required features are present, fill missing with defaults or 0
    # We must match the exact columns the model saw during training
    train_features = clf.named_steps['preprocess'].feature_names_in_
    for col in train_features:
        if col not in X_inference.columns:
            X_inference[col] = 0
            
    # Align column order
    X_inference = X_inference[train_features]

    output_path = Path(args.output)
    debug_path = output_path.with_name(f"{output_path.stem}_debug.csv")

    # Predict probabilities
    y_proba = clf.predict_proba(X_inference)[:, 1]
    y_pred = np.where(y_proba >= 0.5, 'Closed', 'Open')

    # Export raw fields + aligned model features + predictions for debugging.
    debug_base_cols = [
        c for c in [
            'id', 'name', 'category_primary', 'address_freeform', 'website', 'url_alive',
            'days_since_update', 'source_count', 'has_website', 'has_phone', 'has_email', 'has_address'
        ] if c in df_clean.columns
    ]
    debug_df = pd.concat(
        [
            df_clean[debug_base_cols].reset_index(drop=True),
            X_inference.reset_index(drop=True),
            pd.DataFrame({
                'probability_closed': y_proba,
                'prediction': y_pred,
            }),
        ],
        axis=1,
    )

    debug_df = debug_df.sort_values('probability_closed', ascending=False)
    debug_df.to_csv(debug_path, index=False)
    print(f"Saved debug features: {debug_path}")
    
    # Compile results — same columns as audit CSVs for easy review
    results = df_clean[['name', 'address_freeform', 'category_primary', 'lat', 'lon']].copy()
    results = results.rename(columns={'address_freeform': 'address', 'category_primary': 'category'})
    results['P_closed'] = y_proba.round(3)
    results['google_url'] = (
        'https://www.google.com/search?q='
        + results['name'].str.replace(' ', '%20', regex=False)
        + '%20'
        + results['address'].fillna('').str.replace(' ', '%20', regex=False)
    )
    results['actually_closed'] = ''

    results = results.sort_values('P_closed', ascending=False)
    results = results[['name', 'address', 'category', 'P_closed', 'google_url', 'actually_closed']]

    results.to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} predictions to {output_path}")
    print("\nTop 10 Most Likely Closed Places:")
    print(results.head(10)[['name', 'P_closed']])

    if args.eval:
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        labels = pd.read_csv(args.eval, skiprows=2)
        labels = labels[['name', 'actually_closed']].dropna(subset=['actually_closed'])
        labels['actually_closed'] = pd.to_numeric(labels['actually_closed'], errors='coerce').dropna()
        labels['_key'] = labels['name'].str.strip().str.lower()
        results['_key'] = results['name'].str.strip().str.lower()
        merged = results.merge(labels[['_key', 'actually_closed']], on='_key', how='inner')
        print(f"\n EVAL against {args.eval} ")
        print(f"Matched {len(merged)} / {len(labels)} labeled rows")
        y_true = merged['actually_closed'].astype(int).values
        y_prob = merged['P_closed'].values
        print(f"Base rate: {y_true.mean():.1%} actually closed\n")

        best_thresh, best_f1 = 0.5, 0.0
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            y_pred_t = (y_prob >= thresh).astype(int)
            flagged = y_pred_t.sum()
            if flagged == 0:
                continue
            p = precision_score(y_true, y_pred_t, zero_division=0)
            r = recall_score(y_true, y_pred_t, zero_division=0)
            f = f1_score(y_true, y_pred_t, zero_division=0)
            print(f"thresh={thresh}  flagged={flagged:3d}  precision={p:.2f}  recall={r:.2f}  f1={f:.2f}")
            if f > best_f1:
                best_f1, best_thresh = f, thresh

        # Confusion matrix at every threshold
        print(f"\n Confusion Matrix Detail ")
        print(f"{'thresh':>8}  {'TP':>5}  {'FP':>5}  {'TN':>5}  {'FN':>5}  {'FP rate':>8}  {'FN rate':>8}  {'of {n} closed caught':>10}")
        n_actual_closed = int(y_true.sum())
        n_actual_open   = int((1 - y_true).sum())
        print(f"          (actually closed={n_actual_closed}, actually open={n_actual_open})")
        print()
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            y_pred_t = (y_prob >= thresh).astype(int)
            if y_pred_t.sum() == 0:
                continue
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            print(f"  {thresh:>6}  {tp:>5}  {fp:>5}  {tn:>5}  {fn:>5}  {fpr:>8.2f}  {fnr:>8.2f}  {tp}/{n_actual_closed} closed caught")
        print()
        print(f"  TP = correctly flagged as closed")
        print(f"  FP = open place wrongly flagged (false alarm)")
        print(f"  TN = open place correctly left alone")
        print(f"  FN = closed place the model missed")

        # Top-10 feature importances from the trained model
    print(f"\n Top 10 Feature Importances ")
    xgb_step = clf.named_steps['xgb']
    pre_step = clf.named_steps['preprocess']
    feat_names = pre_step.get_feature_names_out()
    importances = xgb_step.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feat_names[i]:<40s} {importances[i]:.4f}")

if __name__ == "__main__":
    main()
