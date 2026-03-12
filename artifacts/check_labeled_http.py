"""Check HTTP metadata for labeled training data only (not all 400K+ places)."""
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / 'src'))

from utils.http_meta_checker import check_http_meta, CACHE_PATH

def main():
    import pandas as pd
    from hybrid import load_merged

    print("Loading labeled data...")
    df = load_merged()
    df = df[df['fsq_label'].isin(['open', 'closed'])]
    df = df[df['match_status'].isin(['direct_match', 'search_match'])]
    df = df.drop_duplicates(subset=['id'])

    has_url = df[df['website'].fillna('').str.len() > 0]
    print(f"Labeled with URLs: {len(has_url)}")

    # Load existing cache
    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())
    print(f"Already cached: {len(cache)}")

    to_check = {row['id']: row['website'] for _, row in has_url.iterrows()
                if row['id'] not in cache}
    print(f"To check: {len(to_check)}")

    if not to_check:
        print("Nothing to check.")
        return

    checked = 0
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {
            executor.submit(check_http_meta, url): pid
            for pid, url in to_check.items()
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
            if checked % 2000 == 0:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                CACHE_PATH.write_text(json.dumps(cache))

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))
    print(f"\nDone. Cache now has {len(cache)} entries.")


if __name__ == '__main__':
    import os
    os.chdir(ROOT)
    main()
