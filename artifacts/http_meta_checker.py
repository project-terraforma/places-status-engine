"""
Enhanced HTTP response metadata extraction.

Captures temporal and structural signals from HTTP responses WITHOUT body scraping:
- Response time (latency_ms)
- Content-Length header
- Last-Modified header → days_since_modified
- Server header
- Redirect chain: count, final domain, domain changed
- Content-Type (HTML vs redirect/parking page indicators)

Usage:
    python http_meta_checker.py --dtmv-only    # DTMV URLs only
    python http_meta_checker.py                 # All training + DTMV URLs
"""
import json
import sys
import time
import socket
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CACHE_PATH = PROJECT_ROOT / 'cache' / 'http_meta_cache.json'


def get_domain(url):
    if not isinstance(url, str) or not url.strip():
        return ''
    url = url.strip()
    if not url.startswith('http'):
        url = 'https://' + url
    try:
        return urlparse(url).netloc.lower().split(':')[0]
    except Exception:
        return ''


def get_url_path_depth(url):
    """Count path segments in URL. Root = 0, /about = 1, /locations/mv = 2."""
    if not isinstance(url, str) or not url.strip():
        return -1
    url = url.strip()
    if not url.startswith('http'):
        url = 'https://' + url
    try:
        path = urlparse(url).path.strip('/')
        if not path:
            return 0
        return len(path.split('/'))
    except Exception:
        return -1


# Known third-party listing domains (not the business's own website)
THIRD_PARTY_DOMAINS = {
    'facebook.com', 'instagram.com', 'twitter.com', 'x.com',
    'yelp.com', 'yelp.ca', 'tripadvisor.com',
    'linkedin.com', 'youtube.com', 'tiktok.com',
    'google.com', 'maps.google.com',
    'nextdoor.com', 'bbb.org',
}

# Known parking/registrar indicators in Server header
PARKING_SERVERS = {
    'parking', 'sedoparking', 'bodis', 'hugedomains',
}


def check_http_meta(url, timeout=8):
    """
    Get HTTP response metadata for a URL.

    Returns dict with temporal and structural signals.
    """
    result = {
        'latency_ms': None,
        'content_length': None,
        'last_modified': None,
        'days_since_modified': None,
        'server': '',
        'content_type': '',
        'redirect_count': 0,
        'final_domain': '',
        'domain_changed': False,
        'is_third_party': False,
        'url_path_depth': get_url_path_depth(url),
    }

    if not isinstance(url, str) or not url.strip():
        return result

    url = url.strip()
    if not url.startswith('http'):
        url = 'https://' + url

    orig_domain = get_domain(url).removeprefix('www.')
    result['is_third_party'] = any(tp in orig_domain for tp in THIRD_PARTY_DOMAINS)

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        start = time.monotonic()
        resp = requests.get(
            url, timeout=timeout, allow_redirects=True,
            headers=headers, stream=True,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        result['latency_ms'] = round(elapsed_ms)

        # Response headers
        result['content_length'] = resp.headers.get('Content-Length')
        if result['content_length']:
            try:
                result['content_length'] = int(result['content_length'])
            except (ValueError, TypeError):
                result['content_length'] = None

        result['server'] = (resp.headers.get('Server') or '')[:100]
        result['content_type'] = (resp.headers.get('Content-Type') or '')[:100]

        # Last-Modified → days since modified
        lm = resp.headers.get('Last-Modified')
        if lm:
            result['last_modified'] = lm
            try:
                from email.utils import parsedate_to_datetime
                lm_dt = parsedate_to_datetime(lm)
                now = datetime.now(timezone.utc)
                result['days_since_modified'] = (now - lm_dt).days
            except Exception:
                pass

        # Redirect chain
        result['redirect_count'] = len(resp.history)
        final_domain = get_domain(resp.url).removeprefix('www.')
        result['final_domain'] = final_domain
        result['domain_changed'] = (orig_domain != final_domain) if orig_domain and final_domain else False

    except requests.exceptions.Timeout:
        result['latency_ms'] = timeout * 1000
    except Exception:
        pass

    return result


def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--dtmv-only', action='store_true')
    parser.add_argument('--training-only', action='store_true')
    parser.add_argument('--workers', type=int, default=30)
    args = parser.parse_args()

    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())

    # Collect URLs: keyed by place ID
    urls_to_check = {}  # place_id -> url

    if not args.training_only:
        print("Loading DTMV snapshot...")
        dtmv = pd.read_parquet(PROJECT_ROOT / 'cache' / 'eval_snapshots' / 'dtmv_features.parquet')
        for _, row in dtmv.iterrows():
            pid = row.get('id', '')
            url = row.get('website', '')
            if isinstance(url, str) and url.strip() and pid and pid not in cache:
                urls_to_check[pid] = url

    if not args.dtmv_only:
        print("Loading training data...")
        for city_file in sorted((PROJECT_ROOT / 'assets').glob('*_places_processed.parquet')):
            city_df = pd.read_parquet(city_file)
            for _, row in city_df.iterrows():
                pid = row.get('id', '')
                url = row.get('website', '')
                if isinstance(url, str) and url.strip() and pid and pid not in cache:
                    urls_to_check[pid] = url

    print(f"URLs to check: {len(urls_to_check)}")
    print(f"Already cached: {len(cache)}")

    if not urls_to_check:
        print("Nothing to check.")
        return

    checked = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_http_meta, url): pid
            for pid, url in urls_to_check.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="HTTP meta"):
            pid = futures[future]
            try:
                cache[pid] = future.result()
            except Exception:
                cache[pid] = {
                    'latency_ms': None, 'content_length': None,
                    'last_modified': None, 'days_since_modified': None,
                    'server': '', 'content_type': '', 'redirect_count': 0,
                    'final_domain': '', 'domain_changed': False,
                    'is_third_party': False, 'url_path_depth': -1,
                }
            checked += 1
            if checked % 500 == 0:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                CACHE_PATH.write_text(json.dumps(cache))

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))

    # Summary
    has_lm = sum(1 for v in cache.values() if v.get('days_since_modified') is not None)
    has_cl = sum(1 for v in cache.values() if v.get('content_length') is not None)
    redirected = sum(1 for v in cache.values() if v.get('redirect_count', 0) > 0)
    domain_changed = sum(1 for v in cache.values() if v.get('domain_changed'))
    third_party = sum(1 for v in cache.values() if v.get('is_third_party'))

    print(f"\nResults ({len(cache)} total):")
    print(f"  Has Last-Modified: {has_lm}")
    print(f"  Has Content-Length: {has_cl}")
    print(f"  Redirected: {redirected}")
    print(f"  Domain changed: {domain_changed}")
    print(f"  Third-party URL: {third_party}")

    if has_lm > 0:
        days = [v['days_since_modified'] for v in cache.values() if v.get('days_since_modified') is not None]
        import statistics
        print(f"  Last-Modified: min={min(days)}d, max={max(days)}d, median={statistics.median(days):.0f}d")


if __name__ == '__main__':
    main()
