"""
Check URL liveness for all SF places that have a website.
Results are cached so re-runs only check new URLs.

Usage: python url_checker.py
"""
import requests
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CACHE_PATH = PROJECT_ROOT / 'cache' / 'url_status.json'
PARQUET_PATH = PROJECT_ROOT / 'assets' / 'sf_places_processed.parquet'


def check_url(url, timeout=5):
    """Send HEAD request to a URL. Returns True if site responds, False if dead."""
    # Some URLs don't have a scheme — add https://
    if not url.startswith('http'):
        url = 'https://' + url
    try:
        r = requests.head(
            url, timeout=timeout, allow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        return r.status_code < 400
    except Exception:
        return False


def main():
    # Load all places
    df = pd.read_parquet(PARQUET_PATH)

    # Load existing cache (skip already-checked URLs)
    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())

    # Filter to places with a non-empty website
    has_url = df[df['website'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    to_check = has_url[~has_url['id'].isin(cache)]

    print(f"Total with URL: {len(has_url)}")
    print(f"Already cached:  {len(cache)}")
    print(f"To check:        {len(to_check)}")

    if len(to_check) == 0:
        print("Nothing to check.")
        return

    # Check URLs in parallel (20 threads)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(check_url, row['website']): row['id']
            for _, row in to_check.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking URLs"):
            place_id = futures[future]
            cache[place_id] = future.result()

    # Save cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))

    # Print summary
    alive = sum(1 for v in cache.values() if v)
    dead = sum(1 for v in cache.values() if not v)
    total = alive + dead
    print(f"\nResults: {alive} alive, {dead} dead ({dead/total*100:.1f}% dead)")


if __name__ == '__main__':
    main()
