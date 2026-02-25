"""
Check URL liveness for all places across all cities.
Results are cached so re-runs only check new URLs.

Usage: python url_checker.py
"""
import requests
import pandas as pd
import json
import socket
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CACHE_PATH = PROJECT_ROOT / 'cache' / 'url_status.json'
ASSETS_DIR = PROJECT_ROOT / 'assets'


def _url_result_alive(val) -> bool:
    """Extract alive bool from either legacy bool or new dict cache value."""
    if isinstance(val, dict):
        return bool(val.get("alive", False))
    return bool(val)


def _url_result_status(val) -> int:
    """Extract status code from either legacy bool or new dict cache value.

    Legacy mapping: True → 200, False → -1 (unknown dead reason).
    New format stores the actual HTTP code, or:
      -1  connection error / exception
      -2  timeout
      -3  invalid / empty URL
    """
    if isinstance(val, dict):
        return int(val.get("status_code", -1))
    return 200 if val else -1


def check_url(url, timeout=5):
    """
    Check URL liveness.
    Uses HEAD first, then falls back to GET when HEAD is blocked or inconclusive.

    Returns dict: {"alive": bool, "status_code": int}
      status_code is the actual HTTP response code, or one of:
        -1  connection error / exception
        -2  request timeout
        -3  invalid / empty URL
    """
    if not isinstance(url, str) or not url.strip():
        return {"alive": False, "status_code": -3}

    url = url.strip()
    candidates = [url] if url.startswith(("http://", "https://")) else [f"https://{url}", f"http://{url}"]
    headers = {"User-Agent": "Mozilla/5.0"}
    alive_status = {401, 403, 405, 429}
    last_code = -1

    for candidate in candidates:
        try:
            head_resp = requests.head(
                candidate,
                timeout=timeout,
                allow_redirects=True,
                headers=headers,
            )
            code = head_resp.status_code
            if code < 400 or code in alive_status:
                return {"alive": True, "status_code": code}
            last_code = code
        except requests.exceptions.Timeout:
            last_code = -2
        except Exception:
            pass

        # Many sites reject HEAD but respond to GET.
        try:
            get_resp = requests.get(
                candidate,
                timeout=timeout,
                allow_redirects=True,
                headers=headers,
                stream=True,
            )
            code = get_resp.status_code
            if code < 400 or code in alive_status:
                return {"alive": True, "status_code": code}
            last_code = code
        except requests.exceptions.Timeout:
            if last_code == -1:
                last_code = -2
        except Exception:
            pass

    return {"alive": False, "status_code": last_code}


def network_available(timeout=3):
    """
    Quick connectivity preflight.
    Returns True when DNS + outbound HTTP appear available.
    """
    probe_hosts = ["example.com", "google.com", "cloudflare.com"]
    resolved = False
    for host in probe_hosts:
        try:
            socket.getaddrinfo(host, 80)
            resolved = True
            break
        except Exception:
            continue

    if not resolved:
        return False

    headers = {"User-Agent": "Mozilla/5.0"}
    for probe_url in ["https://example.com", "http://example.com"]:
        try:
            resp = requests.get(
                probe_url,
                timeout=timeout,
                allow_redirects=True,
                headers=headers,
                stream=True,
            )
            if resp.status_code < 500:
                return True
        except Exception:
            continue
    return False


def main():
    import sys
    sys.path.append(str(SCRIPT_DIR.parent))
    from hybrid import load_merged  # Import from local hybrid.py

    print("Loading labeled datasets...")
    # 1. Load merged datasets
    try:
        df = load_merged()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not network_available():
        print("Error: network preflight failed (DNS/HTTP unavailable). URL liveness checks were not run.")
        return

    # 2. Filter down to ONLY labeled data points
    df = df[df['fsq_label'].isin(['open', 'closed'])]
    print(f"\nTotal labeled places to potentially check: {len(df)}")

    # Load existing cache
    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())

    # Filter to places with a non-empty website
    has_url = df[df['website'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    to_check = has_url[~has_url['id'].isin(cache)]

    print(f"Total labeled with URL: {len(has_url)}")
    print(f"Already cached:         {len(cache)}")
    print(f"To check:               {len(to_check)}")

    if len(to_check) == 0:
        print("Nothing to check.")
        return

    # Migrate any legacy bool entries to new dict format before writing
    for pid, val in list(cache.items()):
        if not isinstance(val, dict):
            cache[pid] = {"alive": bool(val), "status_code": 200 if val else -1}

    # Check URLs in parallel (50 threads)
    checked = 0
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {
            executor.submit(check_url, row['website']): row['id']
            for _, row in to_check.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking URLs"):
            place_id = futures[future]
            cache[place_id] = future.result()  # already a dict from check_url
            checked += 1
            # Save checkpoint every 5000
            if checked % 5000 == 0:
                CACHE_PATH.write_text(json.dumps(cache))

    # Final save
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))

    # Summary
    alive = sum(1 for v in cache.values() if _url_result_alive(v))
    dead = sum(1 for v in cache.values() if not _url_result_alive(v))
    total = alive + dead
    print(f"\nResults: {alive} alive, {dead} dead ({dead/total*100:.1f}% dead)")
    # Status code breakdown for dead URLs
    dead_codes: dict[int, int] = {}
    for v in cache.values():
        if not _url_result_alive(v):
            code = _url_result_status(v)
            dead_codes[code] = dead_codes.get(code, 0) + 1
    print("Dead URL status codes:", dict(sorted(dead_codes.items())))


if __name__ == '__main__':
    main()
