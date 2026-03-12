import numpy as np
import pandas as pd
import json
from pathlib import Path

# Categories where a dead/missing website is strongly diagnostic of closure.
# These are owner-operated, location-specific businesses — the website lives and dies
# with that single location. Derived from url_delta >= 0.25 across training data.
# Contrast: chains/tech (sushi_restaurant, cafe, it_service) have inverted or zero signal
# because a parent-company website outlives any individual closed location.
CONSUMER_CATEGORIES = {
    "middle_eastern_restaurant", "thai_restaurant", "breakfast_and_brunch_restaurant",
    "mexican_restaurant", "clothing_store", "gift_shop", "sandwich_shop",
    "smoothie_juice_bar", "american_restaurant", "pizza_restaurant", "burger_restaurant",
    "fast_food_restaurant", "discount_store", "legal_services", "tobacco_shop",
    "bar", "bakery", "retail", "pharmacy", "furniture_store", "restaurant",
    "nursery_and_gardening", "flowers_and_gifts_shop", "chicken_restaurant",
}


def load_merged():
    """Load places + labels from all available cities and merge."""
    current_dir = Path(__file__).resolve().parent
    assets = current_dir.parent / 'assets'
    
    dfs = []
    # Find all city datasets (sf_places_processed.parquet, nyc_places_processed.parquet, etc.)
    for processed_file in sorted(assets.glob('*_places_processed.parquet')):
        city = processed_file.name.split('_places_processed')[0]  # e.g. "sf", "nyc"
        label_file = assets / f'{city}_places_labeled.parquet'
        
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


def _alive(val) -> float:
    """Extract alive status from a URL cache entry. Returns 1.0, 0.0, or NaN."""
    if val is None:
        return float('nan')
    return float(val.get('alive', False) if isinstance(val, dict) else bool(val))


def _status_code(val) -> float:
    """Extract HTTP status code from a URL cache entry. Returns code or NaN."""
    if val is None:
        return float('nan')
    if isinstance(val, dict):
        return float(val.get('status_code', -1))
    return 200.0 if val else -1.0


def build_features(df):
    """Shared feature processing for both labeled training data and unlabeled inference data."""
    current_dir = Path(__file__).resolve().parent

    drop_cols = [
        'fsq_id', 'fsq_name', 'fsq_date_closed', 'fsq_label', 'fsq_suspected_closed',
        'match_status', 'match_score', 'search_distance',
        'operating_status',
        'name', 'lat', 'lon', 'city',
        'website', 'phone', 'email', 'social',
        'last_update', 'address_freeform', 'confidence'
    ]

    # OSM disused/abandoned: use live values if present, else fall back to training cache
    if 'osm_disused' not in df.columns:
        osm_cache_path = current_dir.parent / 'cache' / 'osm_features.parquet'
        if osm_cache_path.exists():
            osm = pd.read_parquet(osm_cache_path)[['id', 'osm_disused', 'osm_dist_m']]
            df = df.merge(osm, on='id', how='left')
    df['osm_disused'] = df.get('osm_disused', 0)
    df['osm_disused'] = pd.to_numeric(df['osm_disused'], errors='coerce').fillna(0).astype(int)

    # URL status: use live values if present, else fall back to cache
    url_cache_path = current_dir.parent / 'cache' / 'url_status.json'
    url_cache: dict = {}
    if url_cache_path.exists():
        url_cache = json.loads(url_cache_path.read_text())

    X = df.copy()

    cached_alive = X['id'].map({k: _alive(v) for k, v in url_cache.items()})
    if 'url_alive' in X.columns:
        X['url_alive'] = pd.to_numeric(X['url_alive'], errors='coerce')
        X['url_alive'] = X['url_alive'].where(X['url_alive'].notna(), cached_alive)
    else:
        X['url_alive'] = cached_alive

    cached_code = X['id'].map({k: _status_code(v) for k, v in url_cache.items()})
    if 'url_status_code' in X.columns:
        X['url_status_code'] = pd.to_numeric(X['url_status_code'], errors='coerce')
        X['url_status_code'] = X['url_status_code'].where(X['url_status_code'].notna(), cached_code)
    else:
        X['url_status_code'] = cached_code

    # url_is_4xx: page definitively gone (404, 410) but domain still responds
    X['url_is_4xx'] = np.where(
        (X['url_alive'] == 0) & X['url_status_code'].notna() & (X['url_status_code'] >= 400) & (X['url_status_code'] < 500),
        1.0, np.where(X['url_status_code'].isna(), float('nan'), 0.0)
    )
    # url_is_error: server unreachable (DNS fail, connection refused)
    X['url_is_error'] = np.where(
        (X['url_alive'] == 0) & X['url_status_code'].notna() & (X['url_status_code'] < 0),
        1.0, np.where(X['url_status_code'].isna(), float('nan'), 0.0)
    )

    # consumer_cat: 1 if the place is in a location-specific category where a dead
    # URL is strongly diagnostic; 0 for chains/tech/B2B where corporate sites survive
    X['consumer_cat'] = X['category_primary'].isin(CONSUMER_CATEGORIES).astype(float)

    # consumer_dead_url: consumer business with a dead or missing website.
    # Combines the two most useful signals for non-tech closures.
    # url_alive == 0 includes both "had a website, now dead" and (via has_website) "never had one"
    X['consumer_dead_url'] = (
        (X['consumer_cat'] == 1) & (X['url_alive'] == 0)
    ).astype(float)

    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    # id no longer needed after OSM join
    X = X.drop(columns=['id'], errors='ignore')

    # basic_category / taxonomy_top: simplified Overture Feb-2026+ taxonomy.
    # Fill missing (older processed rows) with empty string so the encoder sees them
    # as a single "unknown" bucket rather than NaN.
    for col in ['basic_category', 'taxonomy_top', 'taxonomy_primary']:
        if col in X.columns:
            X[col] = X[col].fillna('').astype(str)
        else:
            X[col] = ''

    # Score for how many contacts out of 4
    X['contact_richness'] = X['has_website'] + X['has_phone'] + X['has_email'] + X['has_address']

    # brand_loc_count: number of Overture locations sharing the same brand_name.
    # Computed from the full SF+NYC Overture data (non-leaky: from data source, not labels).
    # Branded chains have ~1-2% closure rate vs 15% for unbranded — this gives the model
    # a continuous signal about brand establishment level.
    brand_count_path = current_dir.parent / 'cache' / 'brand_location_counts.json'
    if brand_count_path.exists():
        _brand_counts = json.loads(brand_count_path.read_text())
        X['brand_loc_count'] = X['brand_name'].fillna('').astype(str).map(_brand_counts).fillna(0).astype(int)
    else:
        X['brand_loc_count'] = 0

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
    """Load labeled examples (open/closed) with features, labels, and weights."""
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


if __name__ == "__main__":
    X, y, weights, cities = get_data()
    print(f"Labeled — X: {X.shape}, y: {y.value_counts().to_dict()}")
