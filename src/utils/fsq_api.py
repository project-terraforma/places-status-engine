"""
Hybrid Labeling - Use Foursquare API to label Overture places as open/closed.
With: match confidence, caching, retries, checkpointing.
"""

from pickle import NONE
import time
import os
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from .schema_live_places import get_sf_data, get_city_data
except ImportError:
    from schema_live_places import get_sf_data, get_city_data


FSQ_API_BASE = "https://places-api.foursquare.com"
CACHE_DIR = Path("../../cache/fsq")

# Match thresholds
MAX_DISTANCE_M = 100  # meters
MIN_NAME_SIMILARITY = 0.5  # 0-1 scale
SEARCH_LIMIT = 5  # candidates to consider

# Retry config
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # seconds

# Rate limit handling
RATE_LIMIT_PAUSE = 60  # seconds to pause after sustained 429s
CONSECUTIVE_429_THRESHOLD = 3  # how many 429s before long pause

# Global rate limit state
_consecutive_429s = 0


def get_cache_path(cache_type: str, key: str) -> Path:
    """Get cache file path."""
    safe_key = re.sub(r'[^\w\-]', '_', str(key))[:100]
    return CACHE_DIR / cache_type / f"{safe_key}.json"


def cache_get(cache_type: str, key: str):
    """Get from cache."""
    path = get_cache_path(cache_type, key)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def cache_set(cache_type: str, key: str, data):
    """Save to cache."""
    path = get_cache_path(cache_type, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def fsq_request(endpoint: str, api_key: str, params: dict = None):
    """Make request to FSQ API with retries and smart rate limit handling."""
    global _consecutive_429s
    
    url = f"{FSQ_API_BASE}{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "X-Places-Api-Version": "2025-06-17"
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            # Extract rate limit info from headers
            limit = response.headers.get("X-RateLimit-Limit")
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset_ts = response.headers.get("X-RateLimit-Reset")
            
            if response.status_code == 200:
                _consecutive_429s = 0  # Reset on success
                # Log remaining quota periodically (only if we have real quota info)
                if limit and int(limit) > 0:
                    rem = int(remaining) if remaining else 0
                    lim = int(limit)
                    # Log every 500 calls or when quota is low
                    if rem % 500 == 0 or rem < 100:
                        print(f"\n[FSQ] Quota: {rem}/{lim} remaining")
                return response.json()
            elif response.status_code == 404:
                _consecutive_429s = 0
                return None
            elif response.status_code == 429:
                _consecutive_429s += 1
                
                # Always show rate limit headers for debugging
                print(f"\n[FSQ] 429 Rate Limited - Headers: Limit={limit}, Remaining={remaining}, Reset={reset_ts}")
                
                # Calculate wait time from reset header if available
                wait_time = RETRY_BACKOFF[attempt]
                if reset_ts:
                    try:
                        reset_time = int(reset_ts)
                        wait_time = max(1, reset_time - int(time.time()))
                        print(f"[FSQ] Resets in {wait_time}s")
                    except:
                        pass
                
                # Check if sustained rate limiting
                if _consecutive_429s >= CONSECUTIVE_429_THRESHOLD:
                    # Use reset time if available, otherwise use default pause
                    pause_time = max(wait_time, RATE_LIMIT_PAUSE)
                    print(f"[FSQ] Sustained rate limiting ({_consecutive_429s} consecutive 429s)")
                    print(f"[FSQ] Pausing for {pause_time}s...")
                    time.sleep(pause_time)
                    _consecutive_429s = 0
                else:
                    print(f"[FSQ] Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                continue
            elif response.status_code >= 500:
                print(f"[FSQ] Server error {response.status_code}, retrying...")
                time.sleep(RETRY_BACKOFF[attempt])
                continue
            else:
                print(f"[FSQ] Error {response.status_code}: {response.text[:100]}")
                return None
        except requests.exceptions.Timeout:
            print(f"[FSQ] Timeout, retrying...")
            time.sleep(RETRY_BACKOFF[attempt])
        except Exception as e:
            print(f"[FSQ] Request failed: {e}")
            return None
    
    return None


def get_fsq_id(obj):
    """Extract FSQ ID from response (handles different field names)."""
    if obj is None:
        return None
    return obj.get("fsq_place_id") or obj.get("fsq_id") or obj.get("id")


def get_place_by_id(fsq_id: str, api_key: str):
    """Get place details by FSQ ID (with caching)."""
    # Check cache
    cached = cache_get("details", fsq_id)
    if cached is not None:
        return cached
    
    result = fsq_request(f"/places/{fsq_id}", api_key)
    
    # Cache result (even None to avoid re-fetching)
    if result is not None:
        cache_set("details", fsq_id, result)
    
    return result


def search_places(lat: float, lon: float, name: str, api_key: str):
    """Search for places (with caching). Returns list of candidates."""
    cache_key = f"{lat:.5f}_{lon:.5f}_{name}"
    cached = cache_get("search", cache_key)
    if cached is not None:
        return cached
    
    params = {
        "ll": f"{lat},{lon}",
        "query": name,
        "limit": SEARCH_LIMIT
    }
    result = fsq_request("/places/search", api_key, params)
    
    candidates = []
    if result and "results" in result:
        candidates = result["results"]
    
    cache_set("search", cache_key, candidates)
    return candidates


def name_similarity(name1: str, name2: str) -> float:
    """Calculate name similarity (0-1)."""
    if not name1 or not name2:
        return 0.0
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    return SequenceMatcher(None, n1, n2).ratio()


def normalize_phone(phone: str) -> str:
    """Normalize phone to digits only."""
    if not phone:
        return ""
    return re.sub(r'\D', '', phone)


def normalize_domain(url: str) -> str:
    """Extract domain from URL."""
    if not url:
        return ""
    url = url.lower().replace("https://", "").replace("http://", "").replace("www.", "")
    return url.split("/")[0].split("?")[0]


def score_candidate(candidate, query_name: str, query_phone: str = None, query_website: str = None) -> dict:
    """Score a search candidate for match quality."""
    distance = candidate.get("distance", 9999)
    fsq_name = candidate.get("name", "")
    fsq_phone = normalize_phone(candidate.get("tel", ""))
    fsq_website = normalize_domain(candidate.get("website", ""))
    
    name_sim = name_similarity(query_name, fsq_name)
    
    # Exact match bonuses
    phone_match = query_phone and fsq_phone and query_phone == fsq_phone
    domain_match = query_website and fsq_website and query_website == fsq_website
    
    # Compute overall score (higher = better)
    score = 0.0
    score += max(0, (MAX_DISTANCE_M - distance) / MAX_DISTANCE_M) * 0.4  # Distance (40%)
    score += name_sim * 0.4  # Name similarity (40%)
    score += 0.1 if phone_match else 0  # Phone bonus (10%)
    score += 0.1 if domain_match else 0  # Domain bonus (10%)
    
    return {
        "score": score,
        "distance": distance,
        "name_similarity": name_sim,
        "phone_match": phone_match,
        "domain_match": domain_match,
    }


def choose_best_candidate(candidates, query_name, query_phone=None, query_website=None):
    """Choose best candidate from search results."""
    if not candidates:
        return None, None
    
    best = None
    best_score_info = None
    
    for c in candidates:
        score_info = score_candidate(c, query_name, query_phone, query_website)
        if best is None or score_info["score"] > best_score_info["score"]:
            best = c
            best_score_info = score_info
    
    # Check if best meets minimum thresholds
    if best_score_info:
        if best_score_info["distance"] > MAX_DISTANCE_M:
            return None, {"rejected": "distance", **best_score_info}
        if best_score_info["name_similarity"] < MIN_NAME_SIMILARITY:
            return None, {"rejected": "name_similarity", **best_score_info}
    
    return best, best_score_info

def extract_labels(fsq_data, match_confidence: str = "high"):
    """Extract labels from FSQ response."""
    if fsq_data is None:
        return {
            "fsq_id": None,
            "fsq_name": None,
            "fsq_date_closed": None,
            "fsq_label": None,
            "fsq_suspected_closed": False,
        }
    
    date_closed = fsq_data.get("date_closed")
    flags = fsq_data.get("unresolved_flags", []) or []
    suspected_closed = "closed" in flags
    
    # Label logic with confidence gating
    if date_closed:
        label = "closed"
    elif suspected_closed:
        label = "suspected_closed"
    elif match_confidence == "high":
        label = "open"
    else:
        # Low confidence match - don't assume open
        label = None
    
    return {
        "fsq_id": get_fsq_id(fsq_data),
        "fsq_name": fsq_data.get("name"),
        "fsq_date_closed": date_closed,
        "fsq_label": label,
        "fsq_suspected_closed": suspected_closed,
    }



def get_fsq_id_from_sources(sources):
    """Extract Foursquare ID from Overture sources."""
    if sources is None:
        return None
    if hasattr(sources, '__iter__') and not isinstance(sources, (str, dict)):
        for src in sources:
            if isinstance(src, dict) and src.get("dataset") == "Foursquare":
                return src.get("record_id")
    return None


def get_cached_details(fsq_id: str):
    """Get cached details without API call. Returns None if not cached."""
    return cache_get("details", fsq_id)


def get_cached_search(lat: float, lon: float, name: str):
    """Get cached search without API call. Returns None if not cached."""
    cache_key = f"{lat:.5f}_{lon:.5f}_{name}"
    return cache_get("search", cache_key)


def _process_one_row(row, api_key, cache_only):
    """Process a single place row. Returns labels dict or None if skipped."""
    time.sleep(0.2)  # Rate limit: ~5 RPS per thread × 10 threads = ~50 RPS max
    fsq_id = row["_fsq_id"]
    query_phone = normalize_phone(row.get("phone", ""))
    query_website = normalize_domain(row.get("website", ""))
    
    if fsq_id:
        # Direct lookup (1 API call)
        if cache_only:
            fsq_data = get_cached_details(fsq_id)
            if fsq_data is None:
                return None
        else:
            fsq_data = get_place_by_id(fsq_id, api_key)
        match_status = "direct_match" if fsq_data else "id_not_found"
        match_confidence = "high" if fsq_data else None
        match_score = None
        search_distance = None
    else:
        # Search + details (2 API calls)
        if cache_only:
            candidates = get_cached_search(row["lat"], row["lon"], row["name"])
            if candidates is None:
                return None
        else:
            candidates = search_places(row["lat"], row["lon"], row["name"], api_key)
        
        best_candidate, score_info = choose_best_candidate(
            candidates, row["name"], query_phone, query_website
        )
        
        if best_candidate:
            found_id = get_fsq_id(best_candidate)
            if cache_only:
                fsq_data = get_cached_details(found_id) if found_id else None
                if fsq_data is None and found_id:
                    return None
            else:
                fsq_data = get_place_by_id(found_id, api_key) if found_id else None
            match_status = "search_match"
            match_confidence = "high"
            match_score = score_info["score"] if score_info else None
            search_distance = score_info["distance"] if score_info else None
        elif score_info and "rejected" in score_info:
            fsq_data = None
            match_status = f"search_rejected_{score_info['rejected']}"
            match_confidence = "low"
            match_score = score_info["score"]
            search_distance = score_info["distance"]
        else:
            fsq_data = None
            match_status = "search_not_found"
            match_confidence = None
            match_score = None
            search_distance = None
    
    labels = extract_labels(fsq_data, match_confidence)
    labels["match_status"] = match_status
    labels["match_score"] = match_score
    labels["search_distance"] = search_distance
    labels["overture_id"] = row["id"]
    return labels


def label_places(df, raw_df, api_key, limit=None, resume_from=0, cache_only=False, checkpoint_path=None, workers=10):
    """Label places with FSQ data using concurrent requests.
    
    Args:
        cache_only: If True, skip places that aren't cached (no API calls)
        workers: Number of parallel threads for API calls
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    df = df.copy()
    df["sources"] = raw_df["sources"].values
    df["_fsq_id"] = df["sources"].apply(get_fsq_id_from_sources)
    
    # Sort: FSQ ID places first (direct lookups are faster — 1 call vs 2)
    df = df.sort_values("_fsq_id", key=lambda x: x.isna(), kind="stable").reset_index(drop=True)
    
    total = limit or len(df)
    start_idx = resume_from
    
    print(f"Processing {total - start_idx} places (starting from {start_idx})...")
    print(f"  With FSQ ID: {df['_fsq_id'].notna().sum()} (direct lookup, fast)")
    print(f"  Need search: {df['_fsq_id'].isna().sum()} (search + details, slower)")
    print(f"  Workers: {workers}")
    if cache_only:
        print("  MODE: Cache-only (skipping uncached places)")
    
    results = []
    skipped = 0
    
    # Load checkpoint if exists
    if checkpoint_path and checkpoint_path.exists() and resume_from == 0 and not cache_only:
        checkpoint = pd.read_parquet(checkpoint_path)
        print(f"Resuming from checkpoint: {len(checkpoint)} rows")
        start_idx = len(checkpoint)
        results = checkpoint.to_dict('records')
    
    rows_to_process = df.iloc[start_idx:start_idx + (total - start_idx)]
    
    # Process in parallel batches
    batch_size = workers * 5  # 50 rows per batch with 10 workers
    rows_list = list(rows_to_process.iterrows())
    
    with tqdm(total=len(rows_list), desc="Labeling") as pbar:
        for batch_start in range(0, len(rows_list), batch_size):
            batch = rows_list[batch_start:batch_start + batch_size]
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_one_row, row, api_key, cache_only): idx
                    for idx, row in batch
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        skipped += 1
                    pbar.update(1)
            
            # Checkpoint every 500 new results
            if len(results) % 500 < batch_size and checkpoint_path:
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_parquet(checkpoint_path)
                pbar.set_postfix(saved=len(results))
    
    if cache_only:
        print(f"\nCache-only: processed {len(results)}, skipped {skipped} uncached")
    
    # Final checkpoint
    if results and checkpoint_path:
        checkpoint_df = pd.DataFrame(results)
        checkpoint_df.to_parquet(checkpoint_path)
        print(f"[Final checkpoint saved: {len(results)} rows]")
    
    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="sf", choices=["sf", "nyc"], help="City to label (default: sf)")
    parser.add_argument("--cache-only", action="store_true", help="Only process cached data, no API calls")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of places to process")
    args = parser.parse_args()
    
    load_dotenv()
    
    api_key = os.getenv("FSQ_API_KEY")
    if not api_key and not args.cache_only:
        print("ERROR: FSQ_API_KEY not found in .env")
        return
    
    # City-aware paths
    CHECKPOINT_PATH = Path(f"../../assets/{args.city}_places_labeled_checkpoint.parquet")
    OUTPUT_PATH = Path(f"../../assets/{args.city}_places_labeled.parquet")

    # Create cache dir
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"HYBRID LABELING ({args.city.upper()})" + (" (CACHE-ONLY MODE)" if args.cache_only else ""))

    
    raw_df, processed_df = get_city_data(city=args.city)
    print(f"Loaded {len(processed_df)} places\n")
    
    labeled = label_places(processed_df, raw_df, api_key, limit=args.limit, cache_only=args.cache_only, checkpoint_path=CHECKPOINT_PATH)
    

    print("RESULTS SUMMARY")

    
    print(f"\nTotal labeled: {len(labeled)}")
    print("\nLabel distribution:")
    print(labeled["fsq_label"].value_counts(dropna=False))
    
    print("\nMatch status distribution:")
    print(labeled["match_status"].value_counts())


if __name__ == "__main__":
    main()
