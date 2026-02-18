"""
Overture Places Data Utility

Query POIs from Overture Maps public S3 data.

Usage:
    from utils.places_util import get_places, OverturePlaces
    
    # Get places as list
    places = get_places(lat=37.78, lon=-122.41, radius_m=500)
    
    # Get places as DataFrame with all fields
    client = OverturePlaces()
    df = client.query_bbox(bbox, include_all_fields=True)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd


EARTH_RADIUS_M = 6378137
DEFAULT_RELEASE = "2026-1-21.0"

# San Francisco bounding box
SF_BBOX = {
    "xmin": -122.52,
    "xmax": -122.35,
    "ymin": 37.70,
    "ymax": 37.82
}

# New York City bounding box (all 5 boroughs)
NYC_BBOX = {
    "xmin": -74.26,
    "xmax": -73.70,
    "ymin": 40.49,
    "ymax": 40.92
}

# Lookup by city name
CITY_BBOXES = {
    "sf": SF_BBOX,
    "nyc": NYC_BBOX,
}


def get_latest_release() -> str:
    """Fetch latest Overture release version from S3."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket="overturemaps-us-west-2", 
            Prefix="release/", 
            Delimiter="/"
        )
        
        releases = [
            cp["Prefix"].replace("release/", "").strip("/")
            for page in pages
            for cp in page.get("CommonPrefixes", [])
        ]
        return sorted(releases)[-1] if releases else DEFAULT_RELEASE
    except Exception as e:
        print(f"[WARN] Could not fetch latest release: {e}")
        return DEFAULT_RELEASE


def get_s3_path(release: Optional[str] = None) -> str:
    """Get S3 path for places parquet files."""
    release = release or get_latest_release()
    return f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"


def create_connection() -> duckdb.DuckDBPyConnection:
    """Create DuckDB connection configured for S3."""
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET s3_url_style='path';")
    con.execute("SET s3_use_ssl=true;")
    con.execute("SET enable_object_cache=true;")
    con.execute("PRAGMA memory_limit='2GB';")
    con.execute("PRAGMA threads=4;")
    return con


def radius_to_bbox(lat: float, lon: float, radius_m: float) -> Dict[str, float]:
    """Convert center point + radius to bounding box."""
    delta_lat = (radius_m / EARTH_RADIUS_M) * (180 / math.pi)
    delta_lon = (radius_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))) * (180 / math.pi)
    return {
        "xmin": lon - delta_lon,
        "ymin": lat - delta_lat,
        "xmax": lon + delta_lon,
        "ymax": lat + delta_lat,
    }


def extract_name(names_obj: Any) -> Optional[str]:
    """Extract primary name from Overture names field."""
    if not isinstance(names_obj, dict):
        return None
    primary = names_obj.get("primary")
    if isinstance(primary, str):
        return primary
    if isinstance(primary, dict) and primary:
        return primary.get("en") or next(iter(primary.values()), None)
    return None


def extract_category(categories_obj: Any) -> Tuple[Optional[str], List[str]]:
    """Extract categories from Overture categories field."""
    if not isinstance(categories_obj, dict):
        return None, []
    primary = categories_obj.get("primary")
    alternates = categories_obj.get("alternate", [])
    return primary, alternates if isinstance(alternates, list) else []


class OverturePlaces:
    """Client for querying Overture Places data."""
    
    def __init__(self, release: Optional[str] = None):
        self.release = release or get_latest_release()
        self.s3_path = get_s3_path(self.release)
        print(f"[OverturePlaces] Release: {self.release}")
    
    def query_bbox(
        self,
        bbox: Dict[str, float],
        limit: Optional[int] = None,
        include_all_fields: bool = True
    ) -> pd.DataFrame:
        """
        Query places within a bounding box.
        
        Args:
            bbox: Dict with xmin, xmax, ymin, ymax
            limit: Max results (None for all)
            include_all_fields: Include all columns
        """
        bbox_filter = f"""
            bbox.xmax >= {bbox['xmin']} AND
            bbox.xmin <= {bbox['xmax']} AND
            bbox.ymax >= {bbox['ymin']} AND
            bbox.ymin <= {bbox['ymax']}
        """
        
        if include_all_fields:
            fields = """
                id, names, categories, confidence, 
                websites, socials, emails, phones, brand, addresses, sources,
                operating_status,
                ST_X(ST_Centroid(geometry)) AS lon,
                ST_Y(ST_Centroid(geometry)) AS lat
            """
        else:
            fields = """
                id, names, categories, confidence,
                ST_X(ST_Centroid(geometry)) AS lon,
                ST_Y(ST_Centroid(geometry)) AS lat
            """
        
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        try:
            con = create_connection()
            sql = f"""
                SELECT {fields}
                FROM read_parquet('{self.s3_path}', hive_partitioning=1)
                WHERE {bbox_filter}
                {limit_clause}
            """
            df = con.execute(sql).fetchdf()
            con.close()
            print(f"[OverturePlaces] Found {len(df)} places")
            return df
        except Exception as e:
            print(f"[OverturePlaces] Query failed: {e}")
            return pd.DataFrame()
    
    def query_radius(
        self,
        lat: float,
        lon: float,
        radius_m: float = 500,
        limit: int = 1000,
        include_all_fields: bool = True
    ) -> pd.DataFrame:
        """Query places within radius of a point."""
        bbox = radius_to_bbox(lat, lon, radius_m)
        return self.query_bbox(bbox, limit, include_all_fields)



def get_places(
    lat: float,
    lon: float,
    radius_m: float = 500,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """Get places around a point as list of dicts."""
    client = OverturePlaces()
    df = client.query_radius(lat, lon, radius_m, limit, include_all_fields=True)
    
    results = []
    for _, row in df.iterrows():
        primary, alternates = extract_category(row.get("categories"))
        results.append({
            "id": row["id"],
            "name": extract_name(row.get("names")),
            "category": primary,
            "confidence": row.get("confidence"),
            "lon": row["lon"],
            "lat": row["lat"],
        })
    return results


def get_sf_places(limit: Optional[int] = None) -> pd.DataFrame:
    """Get all places in San Francisco."""
    client = OverturePlaces()
    return client.query_bbox(SF_BBOX, limit=limit, include_all_fields=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Overture Places - San Francisco Sample")
    print("=" * 60)
    
    df = get_sf_places(limit=10)
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample:\n{df[['names', 'categories', 'confidence', 'operating_status']].head()}")
