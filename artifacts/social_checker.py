"""
Check social URL liveness for all labeled places.
Results cached to cache/social_status.json (keyed by place id).

Facebook page IDs (facebook.com/123456) return 404 when the page is deleted —
a reliable deleted-page signal for consumer businesses.
Twitter/X accounts return 404 when the account is deactivated or suspended.

Usage:
    python social_checker.py
"""
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CACHE_PATH = PROJECT_ROOT / "cache" / "social_status.json"
ASSETS_DIR = PROJECT_ROOT / "assets"

sys.path.insert(0, str(SCRIPT_DIR.parent))
from url_checker import check_url, network_available, _url_result_alive


def main():
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from hybrid import load_merged

    print("Loading labeled datasets...")
    try:
        df = load_merged()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not network_available():
        print("Error: network preflight failed. Social checks were not run.")
        return

    df = df[df["fsq_label"].isin(["open", "closed"])]
    print(f"Total labeled places: {len(df)}")

    has_social = df[
        df["social"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)
    ].copy()
    print(f"Places with social URL: {len(has_social)}")

    cache: dict = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())

    to_check = has_social[~has_social["id"].isin(cache)]
    print(f"Already cached:        {len(cache)}")
    print(f"To check:              {len(to_check)}")

    if len(to_check) == 0:
        print("Nothing to check.")
        _print_summary(cache)
        return

    checked = 0
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = {
            executor.submit(check_url, row["social"]): row["id"]
            for _, row in to_check.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking social URLs"):
            place_id = futures[future]
            cache[place_id] = future.result()
            checked += 1
            if checked % 2000 == 0:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                CACHE_PATH.write_text(json.dumps(cache))

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))
    _print_summary(cache)


def _print_summary(cache: dict):
    alive = sum(1 for v in cache.values() if _url_result_alive(v))
    dead = len(cache) - alive
    total = len(cache)
    if total == 0:
        print("\nNo results in cache.")
        return
    print(f"\nResults: {alive} alive, {dead} dead ({dead/total*100:.1f}% dead)")
    from url_checker import _url_result_status
    dead_codes: dict[int, int] = {}
    for v in cache.values():
        if not _url_result_alive(v):
            code = _url_result_status(v)
            dead_codes[code] = dead_codes.get(code, 0) + 1
    print("Dead social URL status codes:", dict(sorted(dead_codes.items())))


if __name__ == "__main__":
    main()
