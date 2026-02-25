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

## Why These Features Are Limiting

### The redundancy problem

Every feature we have measures the same underlying thing from a different angle: **how well-maintained is this place's digital presence?** They are all variations of data completeness and freshness:

- `days_since_update` → data freshness
- `email_count`, `phone_count`, `social_count` → contact completeness
- `src_meta`, `src_Microsoft` → data provider coverage
- `has_address`, `has_brand` → record completeness
- `url_alive` → website freshness
- `contact_richness` → aggregate completeness

These features are **correlated with each other**. A place that's been dropped by Meta probably also has a stale update date, missing emails, and a dead website. Adding more metadata fields (e.g., hours of operation, payment methods, parking info) would measure the same underlying signal — "is this record well-maintained?" — and provide diminishing returns. This is why geographic features (density, clusters), name length, and delta features all failed to improve the model: they were either redundant with existing signals or measured something unrelated to closure.

### The fundamental information gap

Metadata tells you about the **data record**, not the **business**. There are two failure modes this creates:

1. **Alive but poorly tracked** — A family-owned barbershop that's been open for 30 years but has no website, no email, and hasn't been updated in Overture since 2023. The metadata screams "closed" but the business is fine. It just has a small digital footprint. This is the model's main source of false positives.

2. **Dead but well-maintained** — A restaurant chain that went bankrupt last month but still has a live website, active social media (posting "we're closed" announcements), and was recently updated by data providers. The metadata says "healthy" but the business is gone. This causes false negatives, though it's rarer because data providers eventually catch up.

### What would actually break through the ceiling

To go beyond PR-AUC ~0.91, you'd need signals that measure **business activity** rather than **data quality**:

| Signal type | What it measures | Why it's different | Availability |
|-------------|-----------------|-------------------|-------------|
| **Temporal change** | Did this place's data change between releases? | Measures active maintenance vs abandonment, not just current state | Available via Overture changelogs |
| **Customer activity** | Recent reviews, check-ins, transactions | Direct evidence of people visiting | Proprietary (Google, Yelp, credit card data) |
| **Physical evidence** | Satellite/street imagery showing vacant storefront | Ground truth of physical presence | Google Street View (not open source) |
| **Public records** | Business license renewals, tax filings | Legal proof of operation | Government databases, varies by jurisdiction |
| **Social media** | Last Instagram post, last Facebook update | Business self-reporting | Requires scraping, noisy |

The key distinction: our current features are all **static snapshots** of metadata quality. The signals that would matter most are **temporal** (how things change over time) and **behavioral** (evidence of human activity). Overture changelogs are the most accessible of these — they're free, open, and provide temporal change signal across every release.

---

## Scalability Concerns & Future Directions

### The labeling problem
FSQ API has free tier limits. Labeling 427k NYC places takes days. Labeling every city in the US is impractical via FSQ alone.

### Scalable alternatives to FSQ labeling
1. **Overture changelogs** — Places `removed` between releases are free "closed" labels. Global scale, no API needed
2. **URL liveness at scale** — HTTP checks for every place. No labeling required, universal signal
3. **Anomaly detection** — Unsupervised approach: learn what "normal" (open) data looks like, flag deviations. Zero labels needed

---

## Label Quality Audit

### Manual Audit (100 places, direct_match only)

Randomly sampled 50 "closed" and 50 "open" FSQ labels. Manually verified each by Googling.

| FSQ says | Actually correct | Error rate |
|----------|-----------------|------------|
| **Closed** | 46/50 (92%) | 8% false closures |
| **Open** | 30/50 (60%) | **40% actually closed** |

**Key finding:** FSQ is reliable when it says a place is closed (92%), but 40% of its "open" labels are wrong — those businesses are actually closed, FSQ just hasn't caught up. This means the model's reported metrics understate its true performance, because it gets penalized for correctly predicting closures that FSQ missed.

### Per-city label accuracy
| City | Closed accuracy | Open accuracy |
|------|----------------|---------------|
| SF | 89% (8/9) | 58% (7/12) |
| NYC | 93% (38/41) | 61% (23/38) |

---

## Label Cleaning

### Technique
Since 40% of "open" labels are wrong, the model is being penalized for correct predictions during training. Label cleaning uses the model itself to identify and remove likely mislabeled examples:

1. Train model on all data (noisy labels)
2. Score every "open" example — get P(closed)
3. If model says P(closed) > threshold but FSQ says "open" → likely mislabeled
4. Remove those examples and retrain

### Why it works
The model learns the real pattern (stale data + missing contacts = closed) from the majority of correct labels. When it confidently disagrees with a label, the model is usually right and the label is usually wrong. Removing the contradictions lets the model trust its own signal.

### Audit of removed examples (30 samples at threshold 0.7)
- **70% (21/30) were actually closed** — model correctly identified mislabeled "open" examples
- At P ≥ 0.85 confidence: **80% (12/15) correct**
- At P 0.70–0.85: 60% (9/15) correct

### Results comparison

| Approach | Removed | PR-AUC | Precision | Recall | F1 |
|----------|---------|--------|-----------|--------|-----|
| No cleaning | 0 | 0.871 | 0.740 | 0.856 | 0.791 |
| Clean @ P>0.85 | 952 | **0.912** | 0.785 | 0.855 | 0.819 |
| Clean @ P>0.70 | 1,851 | 0.927 | 0.824 | 0.859 | 0.841 |

**Estimated true model performance: PR-AUC ~0.91** (threshold 0.85 is most honest since 80% of removed examples were verified as mislabeled).

---

## Predictions on Unlabeled Data

The model was used to predict on 16,473 places that FSQ could not label (no match found).

| Prediction | Count | % |
|-----------|-------|---|
| Likely open (P < 0.10) | 15,933 | 96.7% |
| Uncertain (0.30–0.70) | 97 | 0.6% |
| Likely closed (P > 0.70) | 73 | 0.4% |
| Likely closed (P > 0.90) | 20 | 0.1% |

### Manual verification of top 5 highest-confidence predictions

| Place | P(closed) | Verification |
|-------|-----------|-------------|
| La Boheme, 24 Minetta Ln, NYC | 0.990 |  Closed — replaced by "da Toscano" |
| Biriyani House, 4345 43rd St, NYC | 0.981 | Closed since 2018 — replaced by "Cardamom" |
| Madison Bistro, 238 Madison Ave, NYC | 0.974 | Closed — last menu from 2019 |
| Our Neighborhood Place, 2231 Chestnut St, SF | 0.962 | Closed May 2025 (was "The Tipsy Pig") |
| TSQ Brasserie, 723 7th Ave, NYC | 0.960 | Closed — replaced by "Lagos TSQ" |

**5/5 confirmed closed.** The model detects closures that FSQ's own labeling pipeline couldn't process — demonstrating it learned the underlying closure pattern, not just FSQ's answers.