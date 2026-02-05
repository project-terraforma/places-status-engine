import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from utils.places_util import OverturePlaces, SF_BBOX, extract_name, extract_category


# DATA FETCHING

def fetch_sf_places(limit: Optional[int] = None, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch all San Francisco places from Overture S3.
    
    Args:
        limit: Max places to fetch (None for all ~180k)
        save_path: If provided, save raw data to parquet
    
    Returns:
        Raw DataFrame with all Overture columns
    """
    print("=" * 60)
    print("FETCHING SAN FRANCISCO PLACES FROM OVERTURE")
    print("=" * 60)
    print(f"Bounding box: {SF_BBOX}")
    print(f"Limit: {limit or 'None (all places)'}")
    print()
    
    client = OverturePlaces()
    df = client.query_bbox(SF_BBOX, limit=limit, include_all_fields=True)
    
    if save_path and not df.empty:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        print(f"Saved raw data to: {save_path}")
    
    return df


def process_sf_places(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw Overture data into clean features.
    
    Extracts:
    - id, lat, lon
    - name (extracted from names dict)
    - category_primary, category_alternates
    - confidence
    - has_website, has_phone, has_social, has_email, has_brand
    - website_count, phone_count, social_count
    - operating_status
    - source_count
    - address fields
    """
    print("\nProcessing features...")
    
    out = pd.DataFrame()
    
    # ID and location
    out["id"] = df["id"]
    out["lat"] = df["lat"]
    out["lon"] = df["lon"]
    
    # Extract name
    out["name"] = df["names"].apply(extract_name)
    
    # Extract categories
    def get_primary_cat(x):
        if isinstance(x, dict):
            return x.get("primary", "")
        return ""
    
    def get_alt_count(x):
        if isinstance(x, dict):
            alt = x.get("alternate", [])
            return len(alt) if isinstance(alt, list) else 0
        return 0
    
    out["category_primary"] = df["categories"].apply(get_primary_cat)
    out["category_alt_count"] = df["categories"].apply(get_alt_count)
    
    # Confidence
    out["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
    
    # Operating status
    out["operating_status"] = df["operating_status"].fillna("unknown")
    
    # Count features (handle NA safely)
    def safe_len(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0
        if isinstance(x, (list, dict, str)):
            return len(x)
        return 0
    
    out["website_count"] = df["websites"].apply(safe_len)
    out["phone_count"] = df["phones"].apply(safe_len)
    out["social_count"] = df["socials"].apply(safe_len)
    out["email_count"] = df["emails"].apply(safe_len)
    
    # Binary presence features
    out["has_website"] = (out["website_count"] > 0).astype(int)
    out["has_phone"] = (out["phone_count"] > 0).astype(int)
    out["has_social"] = (out["social_count"] > 0).astype(int)
    out["has_email"] = (out["email_count"] > 0).astype(int)
    
    # Brand
    def has_brand(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0
        if isinstance(x, dict) and x:
            return 1
        return 0
    
    out["has_brand"] = df["brand"].apply(has_brand)
    
    # Source count
    out["source_count"] = df["sources"].apply(safe_len)
    
    # Address features
    def get_address_field(addresses, field):
        if addresses is None or (isinstance(addresses, float) and pd.isna(addresses)):
            return ""
        if isinstance(addresses, list) and len(addresses) > 0:
            addr = addresses[0]
            if isinstance(addr, dict):
                return addr.get(field, "") or ""
        return ""
    
    out["country"] = df["addresses"].apply(lambda x: get_address_field(x, "country"))
    out["region"] = df["addresses"].apply(lambda x: get_address_field(x, "region"))
    out["locality"] = df["addresses"].apply(lambda x: get_address_field(x, "locality"))
    out["postcode"] = df["addresses"].apply(lambda x: get_address_field(x, "postcode"))
    
    # Fill NaN
    out = out.fillna({"name": "", "category_primary": "", "country": "", "region": "", "locality": "", "postcode": ""})
    
    print(f"Processed {len(out)} places with {len(out.columns)} features")
    return out

# MAIN API

def get_sf_data(
    limit: Optional[int] = None,
    use_cache: bool = True,
    cache_path: str = "../assets/sf_places_raw.parquet"
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    cache_file = Path(cache_path)
    
    # Try to load from cache
    if use_cache and cache_file.exists():
        print(f"Loading from cache: {cache_path}")
        raw_df = pd.read_parquet(cache_path)
        if limit:
            raw_df = raw_df.head(limit)
    else:
        # Always save after fetching
        raw_df = fetch_sf_places(limit=limit, save_path=cache_path)
    
    processed_df = process_sf_places(raw_df)
    
    return raw_df, processed_df


def load_sf_data(path: str = "../assets/sf_places_raw.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch SF Overture Places data")
    parser.add_argument("--limit", type=int, default=None, help="Max places to fetch (None for all)")
    parser.add_argument("--no-cache", action="store_true", help="Don't use/save cache")
    args = parser.parse_args()
    
    print("SCHEMA SF - San Francisco Overture Places")
    
    raw_df, processed_df = get_sf_data(
        limit=args.limit,
        use_cache=not args.no_cache
    )
    
    print("RAW DATA SCHEMA")
    print(f"Shape: {raw_df.shape}")
    print(f"Columns: {list(raw_df.columns)}")
    
    print("PROCESSED DATA SCHEMA")
    print(f"Shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")
    
    print("SAMPLE DATA")
    print(processed_df[["name", "category_primary", "confidence", "operating_status", "has_website", "has_phone"]].head(10))
    
    print("OPERATING STATUS DISTRIBUTION")

    print(processed_df["operating_status"].value_counts())
