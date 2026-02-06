import numpy as np
import pandas as pd

# Load and join
places = pd.read_parquet('../assets/sf_places_processed.parquet')
labels = pd.read_parquet('../assets/sf_places_labeled_checkpoint.parquet')
df = places.merge(labels, left_on='id', right_on='overture_id', how='inner')
df = df.drop(columns=['overture_id'])

print(f"Joined: {len(df)}")
print(f"Labels:\n{df['fsq_label'].value_counts(dropna=False)}")

# Filter: open/closed only, good matches only
df = df[df['fsq_label'].isin(['open', 'closed'])]
df = df[df['match_status'].isin(['direct_match', 'search_match'])]
df = df.drop_duplicates(subset=['id']).reset_index(drop=True)

print(f"\nAfter filtering: {len(df)}")
print(f"Labels:\n{df['fsq_label'].value_counts()}")

# y: 1=closed, 0=open
y = (df['fsq_label'] == 'closed').astype(int)

# Sample weights: direct=1.0, search=0.5+0.5*score
score = pd.to_numeric(df['match_score'], errors='coerce').fillna(0).clip(0, 1)
weights = np.where(df['match_status'] == 'direct_match', 1.0, 0.5 + 0.5 * score)

# X: drop leakage columns (fsq_*, match_*, operating_status, confidence)
drop_cols = [
    'fsq_id', 'fsq_name', 'fsq_date_closed', 'fsq_label', 'fsq_suspected_closed',
    'match_status', 'match_score', 'search_distance',
    'operating_status', 'confidence',
    'id', 'name',  # identifiers, not features
]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Convert list columns to counts
for col in X.columns:
    if X[col].dtype == object:
        sample = X[col].dropna().iloc[0] if len(X[col].dropna()) > 0 else None
        if isinstance(sample, (list, np.ndarray)):
            X[col] = X[col].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)

print(f"\nX shape: {X.shape}")
print(f"X columns: {list(X.columns)}")
print(f"y distribution: {y.value_counts().to_dict()}")
print(f"Weights range: [{weights.min():.2f}, {weights.max():.2f}]")

#dropping raw strings for now and also making sure not to feed lat lon because we want to generalize to different cities
#etc...
X = df.drop(columns=['name_len', 'lat', 'lon', 'website', 'phone', 'email', 'social', 'brand_name', 'last_update', 'address_freeform'])