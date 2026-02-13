import numpy as np
import pandas as pd


def get_data():
    """Load and prepare X, y, weights for training."""
    
    # Load and join
    places = pd.read_parquet('../assets/sf_places_processed.parquet')
    labels = pd.read_parquet('../assets/sf_places_labeled_checkpoint.parquet')
    df = places.merge(labels, left_on='id', right_on='overture_id', how='inner')
    df = df.drop(columns=['overture_id'])
    
    #print(df['fsq_label'].value_counts(dropna=False))
    #print(df['fsq_suspected_closed'].value_counts(dropna=False))

    # Filter: open/closed only, good matches only
    df = df[df['fsq_label'].isin(['open', 'closed'])]
    df = df[df['match_status'].isin(['direct_match', 'search_match'])]
    df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
    
    # y: 1=closed, 0=open
    y = (df['fsq_label'] == 'closed').astype(int)
    
    # Sample weights: direct=1.0, search=0.5+0.5*score
    score = pd.to_numeric(df['match_score'], errors='coerce').fillna(0).clip(0, 1)
    weights = np.where(df['match_status'] == 'direct_match', 1.0, 0.5 + 0.5 * score)
    
    # X: drop leakage columns + identifiers + raw strings + lat/lon
    drop_cols = [
        'fsq_id', 'fsq_name', 'fsq_date_closed', 'fsq_label', 'fsq_suspected_closed',
        'match_status', 'match_score', 'search_distance',
        'operating_status', 'confidence',
        'id', 'name', 'lat', 'lon',
        'website', 'phone', 'email', 'social', 'brand_name',
        'last_update', 'address_freeform',
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    #score for how many contacts out of 4
    X['contact_richness'] = X['has_website'] + X['has_phone'] + X['has_email'] + X['has_address']

    #stale and sparse 
    X['stale_and_sparse'] = ((X['days_since_update'] > X['days_since_update'].median()) & 
                          (X['source_count'] <= 1)).astype(int)


    # Convert list columns to counts (except source_datasets - keep for multi-hot)
    for col in X.columns:
        if col == 'source_datasets':
            continue  # Keep as list for multi-hot encoding in train script
        if X[col].dtype == object:
            sample = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else None
            if isinstance(sample, (list, np.ndarray)):
                X[col] = X[col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
    
    return X, y, weights


if __name__ == "__main__":
    X, y, weights = get_data()
    
    print(f"X shape: {X.shape}")
    print(f"X columns: {list(X.columns)}")
    print(f"y distribution: {y.value_counts().to_dict()}")
    print(f"Weights range: [{weights.min():.2f}, {weights.max():.2f}]")
