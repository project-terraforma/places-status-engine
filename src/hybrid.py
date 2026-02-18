import sklearn.preprocessing
import numpy as np
import pandas as pd
import json
from pathlib import Path


def load_merged():
    """Load places + labels from all available cities and merge."""
    assets = Path('../assets')
    
    dfs = []
    # Find all city datasets (sf_places_processed.parquet, nyc_places_processed.parquet, etc.)
    for processed_file in sorted(assets.glob('*_places_processed.parquet')):
        city = processed_file.name.split('_places_processed')[0]  # e.g. "sf", "nyc"
        label_file = assets / f'{city}_places_labeled_checkpoint.parquet'
        
        if not label_file.exists():
            print(f"  Skipping {city}: no labeled data yet")
            continue
        
        places = pd.read_parquet(processed_file)
        labels = pd.read_parquet(label_file)
        merged = places.merge(labels, left_on='id', right_on='overture_id', how='inner')
        merged = merged.drop(columns=['overture_id'])
        merged['city'] = city
        print(f"  {city}: {len(merged)} places")
        dfs.append(merged)
    
    if not dfs:
        raise FileNotFoundError("No labeled city data found in ../assets/")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(df)} places from {len(dfs)} cities")
    return df


def build_features(df):
#Shared feature processing, same for labeled and unlabeled data
    drop_cols = [
        'fsq_id', 'fsq_name', 'fsq_date_closed', 'fsq_label', 'fsq_suspected_closed',
        'match_status', 'match_score', 'search_distance',
        'operating_status', 'confidence',
        'id', 'name', 'lat', 'lon', 'city',
        'website', 'phone', 'email', 'social', 'brand_name',
        'last_update', 'address_freeform',
    ]

    # Load URL liveness cache (run url_checker.py first to populate)
    url_cache_path = Path('../cache/url_status.json')
    url_status = {}
    if url_cache_path.exists():
        url_status = json.loads(url_cache_path.read_text())

    # Map url_alive by place ID before dropping 'id'
    # True = alive, False = dead, NaN = no website or not checked
    X = df.copy()
    X['url_alive'] = X['id'].map(url_status).astype(float)  # NaN for missing

    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    # Score for how many contacts out of 4
    X['contact_richness'] = X['has_website'] + X['has_phone'] + X['has_email'] + X['has_address']

    # Convert list columns to counts (except source_datasets — keep for multi-hot)
    for col in X.columns:
        if col == 'source_datasets':
            continue
        if X[col].dtype == object:
            sample = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else None
            if isinstance(sample, (list, np.ndarray)):
                X[col] = X[col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)

    return X


def get_data():
    #Load labeled examples (open/closed) with features, labels, and weights
    df = load_merged()

    # Keep only labeled, high-quality matches
    df = df[df['fsq_label'].isin(['open', 'closed'])]
    df = df[df['match_status'].isin(['direct_match', 'search_match'])]
    df = df.drop_duplicates(subset=['id']).reset_index(drop=True)

    y = (df['fsq_label'] == 'closed').astype(int)
    cities = df['city'].values

    score = pd.to_numeric(df['match_score'], errors='coerce').fillna(0).clip(0, 1)
    weights = np.where(df['match_status'] == 'direct_match', 1.0, 0.5 + 0.5 * score)

    X = build_features(df)
    return X, y, weights, cities


def get_unlabeled():
    #Load unlabeled examples (fsq_label is None) with same features
    df = load_merged()

    # Keep only examples where FSQ couldn't determine open/closed
    df = df[~df['fsq_label'].isin(['open', 'closed'])]
    df = df.drop_duplicates(subset=['id']).reset_index(drop=True)

    X = build_features(df)
    return X


if __name__ == "__main__":
    X, y, weights = get_data()
    X_unl = get_unlabeled()

    print(f"Labeled   — X: {X.shape}, y: {y.value_counts().to_dict()}")
    print(f"Unlabeled — X: {X_unl.shape}")
