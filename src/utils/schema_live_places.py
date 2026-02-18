import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    from .places_util import OverturePlaces, SF_BBOX, CITY_BBOXES, extract_name, extract_category
except ImportError:
    from places_util import OverturePlaces, SF_BBOX, CITY_BBOXES, extract_name, extract_category


# DATA FETCHING

def fetch_places(bbox: dict, limit: Optional[int] = None, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch places from Overture S3 for a given bounding box.
    
    Args:
        bbox: Dict with xmin, xmax, ymin, ymax
        limit: Max places to fetch (None for all)
        save_path: If provided, save raw data to parquet
    
    Returns:
        Raw DataFrame with all Overture columns
    """
    print(f"FETCHING PLACES FROM OVERTURE")
    print(f"Bounding box: {bbox}")
    print(f"Limit: {limit or 'None (all places)'}")
    print()
    
    client = OverturePlaces()
    df = client.query_bbox(bbox, limit=limit, include_all_fields=True)
    
    if save_path and not df.empty:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path)
        print(f"Saved raw data to: {save_path}")
    
    return df


# Keep old name for backwards compatibility
def fetch_sf_places(limit=None, save_path=None):
    return fetch_places(SF_BBOX, limit=limit, save_path=save_path)


def process_sf_places(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw Overture data into clean features.
    
    Extracts actual values (not just counts):
    - id, lat, lon
    - name, name_len
    - category_primary, category_alternates (list), category_alt_count
    - confidence
    - website (first URL), website_count, has_website
    - phone (first number), phone_count, has_phone
    - email (first email), email_count, has_email
    - social (first URL), social_count, has_social
    - brand_name, has_brand
    - Full address: freeform, country, region, locality, postcode
    - sources: source_datasets (list), source_count
    - Temporal: last_update, days_since_update
    - operating_status
    """
    print("\nProcessing features...")
    
    out = pd.DataFrame()
    
    # Helper for safe iteration over lists/arrays
    def safe_iter(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        if hasattr(x, '__iter__') and not isinstance(x, (str, dict)):
            return list(x)
        return []
    
    def safe_len(x):
        return len(safe_iter(x))
    
  
    out["id"] = df["id"]
    out["lat"] = df["lat"]
    out["lon"] = df["lon"]
    

    out["name"] = df["names"].apply(extract_name)
    out["name_len"] = out["name"].apply(lambda x: len(x) if x else 0)
    
    def get_primary_cat(x):
        if isinstance(x, dict):
            return x.get("primary", "") or ""
        return ""
    
    def get_alt_cats(x):
        if isinstance(x, dict):
            alt = x.get("alternate", [])
            if isinstance(alt, (list, np.ndarray)):
                return list(alt)
        return []
    
    out["category_primary"] = df["categories"].apply(get_primary_cat)
    out["category_alternates"] = df["categories"].apply(get_alt_cats)
    out["category_alt_count"] = out["category_alternates"].apply(len)
    
   
    out["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
    

    def get_first_website(x):
        items = safe_iter(x)
        for item in items:
            if isinstance(item, dict):
                return item.get("value", "") or ""
            if isinstance(item, str):
                return item
        return ""
    
    out["website"] = df["websites"].apply(get_first_website)
    out["website_count"] = df["websites"].apply(safe_len)
    out["has_website"] = (out["website_count"] > 0).astype(int)

    def get_first_phone(x):
        items = safe_iter(x)
        for item in items:
            if isinstance(item, dict):
                return item.get("value", "") or ""
            if isinstance(item, str):
                return item
        return ""
    
    out["phone"] = df["phones"].apply(get_first_phone)
    out["phone_count"] = df["phones"].apply(safe_len)
    out["has_phone"] = (out["phone_count"] > 0).astype(int)
    

    def get_first_email(x):
        items = safe_iter(x)
        for item in items:
            if isinstance(item, dict):
                return item.get("value", "") or ""
            if isinstance(item, str):
                return item
        return ""
    
    out["email"] = df["emails"].apply(get_first_email)
    out["email_count"] = df["emails"].apply(safe_len)
    out["has_email"] = (out["email_count"] > 0).astype(int)
    
    def get_first_social(x):
        items = safe_iter(x)
        for item in items:
            if isinstance(item, dict):
                return item.get("value", "") or ""
            if isinstance(item, str):
                return item
        return ""
    
    out["social"] = df["socials"].apply(get_first_social)
    out["social_count"] = df["socials"].apply(safe_len)
    out["has_social"] = (out["social_count"] > 0).astype(int)
    
    def get_brand_name(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        if isinstance(x, dict):
            names = x.get("names", {})
            if isinstance(names, dict):
                primary = names.get("primary", "")
                if isinstance(primary, str):
                    return primary
                if isinstance(primary, dict):
                    return primary.get("en", "") or next(iter(primary.values()), "")
        return ""
    
    out["brand_name"] = df["brand"].apply(get_brand_name)
    out["has_brand"] = (out["brand_name"] != "").astype(int)

    def get_first_address(x):
        items = safe_iter(x)
        if items and isinstance(items[0], dict):
            return items[0]
        return {}
    
    addresses = df["addresses"].apply(get_first_address)
    
    out["address_freeform"] = addresses.apply(lambda a: a.get("freeform", "") or "")
    out["address_country"] = addresses.apply(lambda a: a.get("country", "") or "")
    out["address_region"] = addresses.apply(lambda a: a.get("region", "") or "")
    out["address_locality"] = addresses.apply(lambda a: a.get("locality", "") or "")
    out["address_postcode"] = addresses.apply(lambda a: a.get("postcode", "") or "")
    out["has_address"] = (out["address_freeform"] != "").astype(int)
    

    def get_source_datasets(x):
        """Get list of unique source datasets (meta, google, etc.)"""
        items = safe_iter(x)
        datasets = set()
        for item in items:
            if isinstance(item, dict):
                ds = item.get("dataset", "")
                if ds:
                    datasets.add(ds)
        return list(datasets)
    
    def get_latest_update(x):
        """Get most recent update timestamp from sources."""
        items = safe_iter(x)
        latest = None
        for item in items:
            if isinstance(item, dict):
                update_str = item.get("update_time")
                if update_str:
                    try:
                        dt = pd.to_datetime(update_str)
                        if latest is None or dt > latest:
                            latest = dt
                    except:
                        pass
        return latest
    
    out["source_datasets"] = df["sources"].apply(get_source_datasets)
    out["source_count"] = df["sources"].apply(safe_len)
    out["last_update"] = df["sources"].apply(get_latest_update)
    
    # Days since update (temporal feature)
    now = pd.Timestamp.now(tz='UTC')
    def calc_days_since(x):
        if pd.isna(x):
            return -1
        if x.tzinfo is None:
            x = x.tz_localize('UTC')
        return (now - x).days
    
    out["days_since_update"] = out["last_update"].apply(calc_days_since)
    

    out["operating_status"] = df["operating_status"].fillna("unknown")

    string_cols = ["name", "category_primary", "website", "phone", "email", "social", 
                   "brand_name", "address_freeform", "address_country", "address_region", 
                   "address_locality", "address_postcode"]
    for col in string_cols:
        out[col] = out[col].fillna("")
    
    print(f"Processed {len(out)} places with {len(out.columns)} features")
    return out

# MAIN API

def get_city_data(
    city: str = "sf",
    limit: Optional[int] = None,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and process Overture data for a city.
    
    Args:
        city: City key ('sf' or 'nyc')
        limit: Max places to fetch
        use_cache: Whether to use/save parquet caches
    """
    bbox = CITY_BBOXES[city]
    cache_path = f"../../assets/{city}_places_raw.parquet"
    processed_cache_path = f"../../assets/{city}_places_processed.parquet"

    cache_file = Path(cache_path)
    processed_cache_file = Path(processed_cache_path)
    
    # Try to load raw from cache
    if use_cache and cache_file.exists():
        print(f"Loading raw from cache: {cache_path}")
        raw_df = pd.read_parquet(cache_path)
        if limit:
            raw_df = raw_df.head(limit)
    else:
        raw_df = fetch_places(bbox, limit=limit, save_path=cache_path)
    
    # Try to load processed from cache (only if no limit and cache exists)
    if use_cache and limit is None and processed_cache_file.exists():
        print(f"Loading processed from cache: {processed_cache_path}")
        processed_df = pd.read_parquet(processed_cache_path)
    else:
        processed_df = process_sf_places(raw_df)
        if use_cache and limit is None:
            processed_df.to_parquet(processed_cache_path)
            print(f"Saved processed cache: {processed_cache_path}")
    
    return raw_df, processed_df


# Keep old name for backwards compatibility
def get_sf_data(limit=None, use_cache=True, **kwargs):
    return get_city_data("sf", limit=limit, use_cache=use_cache)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Overture Places data")
    parser.add_argument("--city", type=str, default="sf", choices=list(CITY_BBOXES.keys()), help="City to fetch (default: sf)")
    parser.add_argument("--limit", type=int, default=None, help="Max places to fetch (None for all)")
    parser.add_argument("--no-cache", action="store_true", help="Don't use/save cache")
    args = parser.parse_args()
    
    print(f"SCHEMA {args.city.upper()} - Overture Places")
    
    raw_df, processed_df = get_city_data(
        city=args.city,
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
    print(processed_df[["name", "category_primary", "confidence", "days_since_update", "has_website", "has_phone"]].head(10))
    
    print("OPERATING STATUS DISTRIBUTION")

    print(processed_df["operating_status"].value_counts())
