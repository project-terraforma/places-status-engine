"""
SSL certificate and HTTP redirect analysis for business closure detection.

Two temporal causation signals:
1. SSL cert expiry: business closes → stops paying → cert expires
2. Redirect analysis: business closes → URL gets parked or removed → redirect changes

Usage:
    python ssl_checker.py                  # Check all training + DTMV URLs
    python ssl_checker.py --dtmv-only      # Check only DTMV URLs
"""
import ssl
import socket
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CACHE_PATH = PROJECT_ROOT / 'cache' / 'ssl_redirect_cache.json'

# Known domain parking / registrar domains
PARKING_DOMAINS = {
    'sedoparking.com', 'godaddy.com', 'parkingcrew.net', 'bodis.com',
    'hugedomains.com', 'afternic.com', 'dan.com', 'namecheap.com',
    'above.com', 'domainmarket.com', 'undeveloped.com', 'porkbun.com',
    'domainlore.co.uk', 'parked.com', 'sav.com',
    'web.archive.org',  # redirected to archive = domain may be dead
}


def get_domain(url):
    """Extract domain from URL string."""
    if not isinstance(url, str) or not url.strip():
        return ''
    url = url.strip()
    if not url.startswith('http'):
        url = 'https://' + url
    try:
        return urlparse(url).netloc.lower().split(':')[0]
    except Exception:
        return ''


def check_ssl_cert(domain, timeout=5):
    """
    Check SSL certificate for a domain.

    Returns dict with:
        cert_valid: bool - is the cert currently valid?
        cert_days_to_expiry: int - days until cert expires (negative = already expired)
        cert_issuer: str - certificate issuer (e.g. "Let's Encrypt", "DigiCert")
    """
    result = {
        'cert_valid': None,
        'cert_days_to_expiry': None,
        'cert_issuer': '',
    }

    if not domain:
        return result

    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(timeout)
            s.connect((domain, 443))
            cert = s.getpeercert()

        # Parse expiry
        not_after = cert.get('notAfter', '')
        if not_after:
            # Format: 'Sep 14 12:00:00 2025 GMT'
            expiry = datetime.strptime(not_after, '%b %d %H:%M:%S %Y %Z')
            expiry = expiry.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = (expiry - now).days
            result['cert_days_to_expiry'] = delta
            result['cert_valid'] = delta > 0

        # Parse issuer
        issuer = cert.get('issuer', ())
        for field in issuer:
            for key, value in field:
                if key == 'organizationName':
                    result['cert_issuer'] = value
                    break

    except ssl.SSLCertVerificationError:
        # Cert exists but is invalid (expired, wrong domain, etc.)
        result['cert_valid'] = False
        result['cert_days_to_expiry'] = -1
    except ssl.SSLError:
        result['cert_valid'] = False
        result['cert_days_to_expiry'] = -1
    except (socket.timeout, socket.gaierror, ConnectionRefusedError, OSError):
        # Can't reach the server at all
        pass
    except Exception:
        pass

    return result


def check_redirect(url, timeout=5):
    """
    Analyze HTTP redirect chain for a URL.

    Returns dict with:
        redirect_count: int - number of redirects
        domain_changed: bool - final domain differs from original
        final_domain: str - domain after all redirects
        parked: bool - final domain is a known parking service
    """
    result = {
        'redirect_count': 0,
        'domain_changed': False,
        'final_domain': '',
        'parked': False,
    }

    if not isinstance(url, str) or not url.strip():
        return result

    url = url.strip()
    if not url.startswith('http'):
        url = 'https://' + url

    orig_domain = get_domain(url)
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.head(
            url, timeout=timeout, allow_redirects=True, headers=headers
        )
        result['redirect_count'] = len(resp.history)
        result['final_domain'] = get_domain(resp.url)

    except Exception:
        try:
            resp = requests.get(
                url, timeout=timeout, allow_redirects=True,
                headers=headers, stream=True,
            )
            result['redirect_count'] = len(resp.history)
            result['final_domain'] = get_domain(resp.url)
        except Exception:
            return result

    if result['final_domain'] and orig_domain:
        # Check if domain changed (ignore www prefix)
        orig_base = orig_domain.removeprefix('www.')
        final_base = result['final_domain'].removeprefix('www.')
        result['domain_changed'] = orig_base != final_base

        # Check for parking
        for parking in PARKING_DOMAINS:
            if parking in final_base:
                result['parked'] = True
                break

    return result


def check_domain(domain, url=''):
    """Combined SSL + redirect check for a domain/URL."""
    ssl_result = check_ssl_cert(domain)
    redirect_result = check_redirect(url) if url else {
        'redirect_count': 0, 'domain_changed': False,
        'final_domain': '', 'parked': False,
    }
    return {**ssl_result, **redirect_result}


def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--dtmv-only', action='store_true',
                        help='Only check DTMV URLs')
    parser.add_argument('--workers', type=int, default=30,
                        help='Number of parallel workers')
    args = parser.parse_args()

    # Load cache
    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())

    # Collect URLs to check
    urls_to_check = {}  # domain -> url

    if args.dtmv_only:
        print("Loading DTMV snapshot...")
        df = pd.read_parquet(PROJECT_ROOT / 'cache' / 'eval_snapshots' / 'dtmv_features.parquet')
        for _, row in df.iterrows():
            url = row.get('website', '')
            if isinstance(url, str) and url.strip():
                domain = get_domain(url)
                if domain and domain not in cache:
                    urls_to_check[domain] = url
    else:
        # Training data
        print("Loading training data...")
        for city_file in sorted((PROJECT_ROOT / 'assets').glob('*_places_processed.parquet')):
            city_df = pd.read_parquet(city_file)
            if 'website' in city_df.columns:
                for url in city_df['website'].dropna():
                    if isinstance(url, str) and url.strip():
                        domain = get_domain(url)
                        if domain and domain not in cache:
                            urls_to_check[domain] = url

        # Also check DTMV
        dtmv_path = PROJECT_ROOT / 'cache' / 'eval_snapshots' / 'dtmv_features.parquet'
        if dtmv_path.exists():
            df = pd.read_parquet(dtmv_path)
            for _, row in df.iterrows():
                url = row.get('website', '')
                if isinstance(url, str) and url.strip():
                    domain = get_domain(url)
                    if domain and domain not in cache:
                        urls_to_check[domain] = url

    print(f"Domains to check: {len(urls_to_check)}")
    print(f"Already cached: {len(cache)}")

    if not urls_to_check:
        print("Nothing to check.")
        return

    # Check in parallel
    checked = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_domain, domain, url): domain
            for domain, url in urls_to_check.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking SSL/redirects"):
            domain = futures[future]
            try:
                cache[domain] = future.result()
            except Exception as e:
                cache[domain] = {
                    'cert_valid': None, 'cert_days_to_expiry': None,
                    'cert_issuer': '', 'redirect_count': 0,
                    'domain_changed': False, 'final_domain': '', 'parked': False,
                }
            checked += 1
            if checked % 500 == 0:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                CACHE_PATH.write_text(json.dumps(cache))

    # Final save
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))

    # Summary
    valid = sum(1 for v in cache.values() if v.get('cert_valid') is True)
    invalid = sum(1 for v in cache.values() if v.get('cert_valid') is False)
    unknown = sum(1 for v in cache.values() if v.get('cert_valid') is None)
    parked = sum(1 for v in cache.values() if v.get('parked'))
    domain_changed = sum(1 for v in cache.values() if v.get('domain_changed'))

    print(f"\nSSL Results: {valid} valid, {invalid} invalid/expired, {unknown} unreachable")
    print(f"Redirect Results: {domain_changed} domain changes, {parked} parked domains")

    # Cert expiry distribution
    expiries = [v['cert_days_to_expiry'] for v in cache.values()
                if v.get('cert_days_to_expiry') is not None]
    if expiries:
        import statistics
        expired = [e for e in expiries if e < 0]
        print(f"\nCert expiry: min={min(expiries)}d, max={max(expiries)}d, median={statistics.median(expiries):.0f}d")
        print(f"Expired certs: {len(expired)} ({len(expired)/len(expiries)*100:.1f}%)")


if __name__ == '__main__':
    main()
