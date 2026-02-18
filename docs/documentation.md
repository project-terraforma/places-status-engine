# Places Status Engine — Documentation

## Overview

Predict whether an Overture Maps place is **open or closed** using only its metadata (no external APIs at inference time). Labels come from Foursquare's closure data during training.

---

## Data Pipeline

### Sources
- **Overture Maps S3** — Raw place data (name, category, addresses, sources, contacts, coordinates)
- **Foursquare API** — Ground truth labels (`open`/`closed`/`None`)

### Labeling Flow
1. `schema_live_places.py` fetches + processes raw Overture data per city → `{city}_places_processed.parquet`
2. `fsq_api.py` matches each place to FSQ via ID lookup or name/lat-lon search → `{city}_places_labeled_checkpoint.parquet`
3. `hybrid.py` merges processed + labeled data, builds features, returns `X, y, weights`

### Match Types
- **Direct match** — Place has FSQ ID embedded in Overture sources. 1 API call. Weight = 1.0
- **Search match** — Search FSQ by name + coordinates, score candidates. 2 API calls. Weight = 0.5 + 0.5 × match_score
- **None** — FSQ couldn't determine status. Excluded from training

---

## Labeled Data Summary

| City | Total | Open | Closed | None | Closure Rate | Match Type |
|------|-------|------|--------|------|-------------|------------|
| SF   | 52,004 | 33,875 | 2,870 | 15,198 | 5.5% | 24% direct, 47% search |
| NYC  | 44,000 | 32,708 | 8,599 | 758 | 20.2% | 96% direct, 2% search |

**Key observation:** NYC has 4× higher closure rate and almost all direct matches (the FSQ-ID subset from Overture sources). NYC's labeled set is biased toward well-known/chain businesses that FSQ actively tracks.

---

## Features

### What the model uses
| Feature | Type | Importance | What it captures |
|---------|------|-----------|-----------------|
| `days_since_update` | Numeric | 19.4% | Data staleness — dead businesses stop getting updated |
| `email_count` | Numeric | 24.2% | Contact completeness — closed places lose contact info |
| `category_primary` | Target-encoded | 17.7% | Business type risk — restaurants close more than banks |
| `social_count` | Numeric | 10.3% | Social media presence |
| `src_meta` | Binary | 7.3% | Whether Meta/Facebook tracks the business |
| `url_alive` | Float | 4.4% | HTTP liveness check on the place's website |
| `contact_richness` | Numeric | 3.4% | Sum of has_website + has_phone + has_email + has_address |
| `src_Microsoft` | Binary | 3.3% | Whether Microsoft tracks the business |
| `has_address` | Binary | 2.6% | Address completeness |
| `source_count` | Numeric | — | Number of data providers tracking this place |
| `has_brand` | Binary | 0.8% | Chain/franchise indicator |

### What was dropped and why
| Feature | Reason |
|---------|--------|
| `lat`, `lon` | Raw coordinates have no signal — geographic patterns are captured by other features |
| `address_region`, `address_country` | Leaked city identity, caused model to learn "California = one pattern, NY = another" instead of generalizable features |
| `name`, `name_len` | Name text is too noisy as a raw feature |
| `address_postcode`, `address_locality` | Too many unique values, sparse signal |
| `category_alternates` | List column, alt count already captured |

---

## What We Tried and Results

### 1. Baseline SF-only model
- **What:** XGBoost on 52k SF places, all metadata features
- **Result:** PR-AUC 0.850, Precision 0.745, Recall 0.828
- **Why it works:** `days_since_update` alone carries ~36% of the signal. Stale data = likely closed.

### 2. URL liveness (`url_alive`)
- **What:** HTTP HEAD request to each place's website. Alive = 1, dead = 0, no website = NaN
- **Result:** Modest boost. Moved from ~0.84 to ~0.85 PR-AUC
- **Why:** Adds a real-time signal independent of metadata staleness. Dead URLs are strong closure indicators, but most places don't have URLs.

### 3. `category_primary` target encoding
- **What:** Replace category string with its historical closure rate (smoothed)
- **Result:** PR-AUC jumped ~+0.03 when first added
- **Why:** Restaurants close at ~12%, banks at ~2%. Category is a strong structural prior.

### 4. Geographic features (density + clustering) — NOT USED
- **What:** K-Means clustering (k=50) on lat/lon, target-encoding cluster IDs. Place density (count within 200m radius).
- **Result:** Closure rates varied from 1.5% to 12.1% across clusters — real signal exists. But adding to the model didn't improve PR-AUC. Actually slightly worse.
- **Why we skipped:** The geographic signal is already captured indirectly by existing features (`days_since_update`, `src_meta`). Stale data and missing Meta sources correlate with neighborhood quality. Redundant.

### 5. Overture release deltas — NOT USED
- **What:** Compared Jan 2025 vs Feb 2025 Overture releases. Computed field-level changes (source_count_delta, category_changed, etc.)
- **Result:** Almost zero signal. Only 1 month apart, nothing meaningful changed.
- **Why we skipped:** Overture releases are too close together. Would need 6-12 month gaps to see meaningful churn.

### 6. Adding NYC data (multi-city)
- **What:** Labeled 44k NYC places via FSQ (mostly direct match). Combined with 52k SF.
- **Result:**

| Metric | SF only | SF + NYC (combined) | Per-city SF | Per-city NYC |
|--------|---------|---------------------|-------------|-------------|
| PR-AUC | 0.852 | **0.871** | — | — |
| Precision | 0.745 | 0.740 | **0.771** | 0.731 |
| Recall | 0.828 | **0.862** | **0.847** | 0.867 |

- **Key finding:** Adding NYC improved the model overall (PR-AUC 0.852 → 0.871) and even improved SF-specific precision (0.745 → 0.771). The model learned generalizable patterns from diverse data.
- **NYC precision is lower (0.731)** because NYC has a 21% closure rate — harder classification problem with more positives.

### 7. Removing `address_region` / `address_country`
- **What:** Dropped geographic identity features
- **Result:** Slight precision drop in aggregate, but features were leaking city identity
- **Why we kept the change:** Want the model to learn universal patterns, not "California = X, New York = Y"

### 8. `scale_pos_weight` tuning
- **What:** Tried scale_pos_weight = 3 (original) vs 5
- **Result:** 5 tanked precision (0.645) while boosting recall (0.863). Reverted to 3.
- **Why:** Higher weight makes the model predict more things as closed. Good for recall, bad for precision. 3 is the sweet spot.

### 9. Pseudo-labeling — NOT USED
- **What:** Use model to score the ~16k unlabeled (None) examples. High-confidence predictions become pseudo-labels.
- **Result:** Of 16k unlabeled: 15.8k predicted as open (P < 0.10), only 14 predicted as closed (P > 0.80).
- **Why we didn't pursue:** Almost all unlabeled examples are "easy opens." Adding them wouldn't teach the model anything new — would just add more of what it already knows.

---

## Current Best Model

**Architecture:** XGBoost (2000 estimators, max_depth=4, lr=0.03, scale_pos_weight=3)  
**Training data:** 79,531 labeled examples (67,417 open, 12,114 closed) from SF + NYC  
**Pipeline:** TargetEncoder for `category_primary` → passthrough numerics → XGBoost

### Metrics (5-fold stratified CV)
| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.871 ± 0.002** |
| Precision | 0.735 ± 0.010 |
| Recall | 0.856 ± 0.002 |
| F1 | 0.791 ± 0.005 |

### Threshold Operating Points
| Threshold | Precision | Recall | F1 | Best for |
|-----------|-----------|--------|-----|----------|
| 0.5 | 0.740 | 0.862 | 0.796 | Balanced |
| 0.6 | 0.782 | 0.837 | 0.809 | Best F1 |
| 0.7 | 0.820 | 0.794 | 0.807 | High precision |
| 0.8 | 0.865 | 0.741 | 0.798 | Conservative |

---

## Why These Features Work

The model is fundamentally detecting **digital footprint decay**. When a business closes:
1. Data providers stop updating it → `days_since_update` increases
2. Contact info goes stale → `email_count`, `phone_count` drop
3. Large providers drop it → `src_meta`, `src_Microsoft` go to 0
4. Website goes down → `url_alive` = 0
5. The business type had high baseline risk → `category_primary` target encoding captures this

These signals are **universal** — they don't depend on geography. A stale record in Chicago means the same as in SF.

---

## Scalability Concerns & Future Directions

### The labeling problem
FSQ API has free tier limits. Labeling 427k NYC places takes days. Labeling every city in the US is impractical via FSQ alone.

### Scalable alternatives to FSQ labeling
1. **Overture changelogs** — Places `removed` between releases are free "closed" labels. Global scale, no API needed
2. **URL liveness at scale** — HTTP checks for every place. No labeling required, universal signal
3. **Anomaly detection** — Unsupervised approach: learn what "normal" (open) data looks like, flag deviations. Zero labels needed
4. **Active learning** — Only label the examples the model is most uncertain about (P ≈ 0.5). Get 80% of the benefit from 1% of the labels

### Ceiling estimate
With metadata-only features, **PR-AUC 0.87–0.90** is likely the practical ceiling. Beyond that requires fundamentally different signals (customer activity, review data, satellite imagery, public records).

### Can FSQ labels be trusted?
100 audits - 46/50 closed are actually closed. 30/50 open businesses are actually closed. 40% Huge number. 