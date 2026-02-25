"""
OSM closure feature builder.

One Overpass bbox query per city → get all disused/abandoned nodes → spatial join to Overture places.
Saves cache/osm_features.parquet with columns: id, osm_disused, osm_dist_m
"""
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import BallTree

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "osm_features.parquet"
MATCH_RADIUS_M = 20  # peak lift at 20m in validation; beyond this degrades to noise

DISUSED_QUERIES = [
    '"disused:amenity"',
    '"disused:shop"',
    '"disused:office"',
    '"abandoned:amenity"',
    '"abandoned:shop"',
]


def _overpass_bbox(xmin, ymin, xmax, ymax) -> str:
    """Fetch all disused/abandoned nodes+ways in a bbox. Returns list of {lat, lon, tags}."""
    tag_filters = "".join(
        f'  node[{t}]({ymin},{xmin},{ymax},{xmax});\n'
        f'  way[{t}]({ymin},{xmin},{ymax},{xmax});\n'
        for t in DISUSED_QUERIES
    )
    query = f"[out:json][timeout:90];\n(\n{tag_filters});\nout center;"

    for attempt in range(3):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            nodes = []
            for el in elements:
                lat = el.get("lat") or (el.get("center") or {}).get("lat")
                lon = el.get("lon") or (el.get("center") or {}).get("lon")
                if lat and lon:
                    nodes.append({"lat": float(lat), "lon": float(lon), "tags": el.get("tags", {})})
            return nodes
        except Exception as e:
            print(f"  Overpass attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return []


def _spatial_join(places_df: pd.DataFrame, osm_nodes: list, radius_m: float):
    """
    For each place, find nearest OSM disused node within radius_m.
    Returns array of distances (NaN if nothing within radius).
    """
    if not osm_nodes:
        return np.full(len(places_df), np.nan)

    osm_df = pd.DataFrame(osm_nodes)
    # BallTree uses radians
    osm_rad = np.radians(osm_df[["lat", "lon"]].values)
    places_rad = np.radians(places_df[["lat", "lon"]].values)

    tree = BallTree(osm_rad, metric="haversine")
    earth_r = 6371000  # metres
    radius_rad = radius_m / earth_r

    dists, _ = tree.query(places_rad, k=1)
    dists_m = dists[:, 0] * earth_r

    # NaN where no match within threshold
    dists_m[dists_m > radius_m] = np.nan
    return dists_m


def build_osm_features(force=False):
    """
    Build OSM features for all processed city datasets.
    Skips if cache already exists unless force=True.
    """
    if CACHE_PATH.exists() and not force:
        print(f"OSM cache already exists at {CACHE_PATH}. Pass force=True to rebuild.")
        return pd.read_parquet(CACHE_PATH)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).resolve().parents[2] / "assets"

    all_results = []

    for processed_file in sorted(assets.glob("*_places_processed.parquet")):
        city = processed_file.stem.split("_places_processed")[0]
        df = pd.read_parquet(processed_file)[["id", "lat", "lon"]]

        # projectc spans the whole US — skip, bbox would be too large
        if city == "projectc":
            print(f"  {city}: skipping (nationwide, bbox too large for Overpass)")
            df["osm_disused"] = 0
            df["osm_dist_m"] = np.nan
            all_results.append(df[["id", "osm_disused", "osm_dist_m"]])
            continue

        xmin, ymin = df["lon"].min(), df["lat"].min()
        xmax, ymax = df["lon"].max(), df["lat"].max()
        print(f"  {city}: {len(df)} places, querying Overpass bbox [{ymin:.3f},{xmin:.3f},{ymax:.3f},{xmax:.3f}]...")

        nodes = _overpass_bbox(xmin, ymin, xmax, ymax)
        print(f"    → {len(nodes)} disused/abandoned OSM nodes found")

        dists = _spatial_join(df, nodes, MATCH_RADIUS_M)
        df["osm_disused"] = (~np.isnan(dists)).astype(int)
        df["osm_dist_m"] = dists
        all_results.append(df[["id", "osm_disused", "osm_dist_m"]])

    result = pd.concat(all_results, ignore_index=True).drop_duplicates(subset=["id"])
    result.to_parquet(CACHE_PATH)
    print(f"\nSaved OSM features → {CACHE_PATH}")
    print(f"Total: {len(result)} places, {result['osm_disused'].sum()} matched to disused OSM node")
    return result


def query_osm_for_places(df: pd.DataFrame) -> pd.DataFrame:
    """
    Query OSM live for a given DataFrame of places (must have id, lat, lon).
    Returns the same df with osm_disused and osm_dist_m columns added.
    Useful for inference on areas not in the training cache.
    """
    xmin, ymin = df["lon"].min(), df["lat"].min()
    xmax, ymax = df["lon"].max(), df["lat"].max()
    print(f"  Querying Overpass for OSM disused nodes [{ymin:.3f},{xmin:.3f},{ymax:.3f},{xmax:.3f}]...")

    nodes = _overpass_bbox(xmin, ymin, xmax, ymax)
    print(f"  → {len(nodes)} disused/abandoned OSM nodes found")

    dists = _spatial_join(df, nodes, MATCH_RADIUS_M)
    df = df.copy()
    df["osm_disused"] = (~np.isnan(dists)).astype(int)
    df["osm_dist_m"] = dists
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild cache even if it exists")
    args = parser.parse_args()
    build_osm_features(force=args.force)
