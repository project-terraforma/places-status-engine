from hybrid import get_data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, TargetEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Columns dropped before model training — not predictive or geographically biased
DROP_COLS = [
    'address_postcode', 'name_len', 'address_locality',
    'category_alternates', 'address_region', 'address_country',
    'taxonomy_primary',  # redundant with taxonomy_top
    'days_since_update',  # Overture update_time is provider/dataset-level, not business-level
]


def create_pipeline(X, target_cols=None):
    if target_cols is None:
        # basic_category / taxonomy_top are high-cardinality categoricals — target encode same as category_primary
        target_cols = [c for c in ["category_primary", "brand_name", "basic_category", "taxonomy_top"] if c in X.columns]
    cat_cols = [c for c in ["address_region", "address_country"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols and c not in target_cols and c != 'source_datasets']

    preprocess = ColumnTransformer(
        transformers=[
            ("target", TargetEncoder(smooth="auto"), target_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop"
    )
    clf = Pipeline([
        ("preprocess", preprocess),
        ("xgb", XGBClassifier(
            n_estimators=2000,
            max_depth=4,
            min_child_weight=5,
            learning_rate=0.03,
            scale_pos_weight=3.0,
            reg_lambda=1.0,
            reg_alpha=0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        ))
    ])
    return clf


def encode_sources(X):
    if 'source_datasets' not in X.columns:
        return X
    mlb = MultiLabelBinarizer()
    sources = X['source_datasets'].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
    sources_encoded = mlb.fit_transform(sources)
    sources_df = pd.DataFrame(
        sources_encoded,
        columns=[f"src_{c}" for c in mlb.classes_],
        index=X.index
    )
    if 'src_Foursquare' in sources_df.columns:
        sources_df = sources_df.drop(columns=['src_Foursquare'])
    X = pd.concat([X.drop(columns=['source_datasets']), sources_df], axis=1)
    X['meta_dead_url'] = ((X['src_meta'] == 1) & (X['url_alive'] == 0)).astype(float)
    return X


def _prepare(X, y, weights):
    """Encode sources, drop non-predictive columns, convert y to numpy."""
    X = encode_sources(X)
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])
    y = y.to_numpy() if hasattr(y, 'to_numpy') else y
    return X, y, weights


def _run_cv(clf, X, y, weights, label=""):
    """5-fold stratified CV — prints precision, recall, F1, PR-AUC."""
    header = f" 5-FOLD CV{' — ' + label if label else ''} "
    print(f"\n{header}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    r = cross_validate(
        clf, X, y,
        cv=cv,
        scoring=['precision', 'recall', 'f1', 'average_precision'],
        params={'xgb__sample_weight': weights},
        return_train_score=False
    )
    print(f"Precision: {r['test_precision'].mean():.3f} ± {r['test_precision'].std():.3f}")
    print(f"Recall:    {r['test_recall'].mean():.3f} ± {r['test_recall'].std():.3f}")
    print(f"F1:        {r['test_f1'].mean():.3f} ± {r['test_f1'].std():.3f}")
    print(f"PR-AUC:    {r['test_average_precision'].mean():.3f} ± {r['test_average_precision'].std():.3f}")


def _print_features(clf, n=15):
    """Print top-n feature importances from a fitted pipeline."""
    feat_names = clf.named_steps['preprocess'].get_feature_names_out()
    feat_imp = clf.named_steps['xgb'].feature_importances_
    top_idx = np.argsort(feat_imp)[::-1][:n]
    print(f"\nTop {n} features:")
    for i in top_idx:
        print(f"  {feat_names[i]}: {feat_imp[i]:.4f}")


def _eval(clf, X_test, y_test, cities_test=None):
    """Threshold sweep + classification report + optional per-city breakdown."""
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    print("\nTHRESHOLD ANALYSIS")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred_t = (y_proba >= thresh).astype(int)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        f = f1_score(y_test, y_pred_t, zero_division=0)
        print(f"  thresh={thresh}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    print("\nDEFAULT (threshold=0.5)")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    if cities_test is not None:
        print("\n PER-CITY EVALUATION ")
        for city in sorted(set(cities_test)):
            mask = cities_test == city
            yc, pc = y_test[mask], (y_proba[mask] >= 0.5).astype(int)
            n, nc = mask.sum(), yc.sum()
            p = precision_score(yc, pc, zero_division=0)
            r = recall_score(yc, pc, zero_division=0)
            f = f1_score(yc, pc, zero_division=0)
            print(f"  {city}: n={n}, closed={nc} ({nc/n:.1%}), P={p:.3f}, R={r:.3f}, F1={f:.3f}")


def train_model(X_raw, y, weights):
    X, y, weights = _prepare(X_raw, y, weights)
    clf = create_pipeline(X)

    clf.fit(X, y, xgb__sample_weight=weights)

    # Identify and remove suspicious "open" labels the model disagrees with
    probs = clf.predict_proba(X)[:, 1]
    open_mask = y == 0
    suspicious = (probs[open_mask] > 0.85)
    open_indices = np.where(open_mask)[0]
    clean_mask = np.ones(len(y), dtype=bool)
    clean_mask[open_indices[suspicious]] = False

    X_clean = X.iloc[clean_mask].reset_index(drop=True)
    y_clean = y[clean_mask]
    w_clean = weights[clean_mask]

    print(f"\n LABEL CLEANING ")
    print(f"  'Open' examples: {open_mask.sum()}")
    print(f"  Model thinks are closed (P>0.85): {suspicious.sum()} ({suspicious.mean():.1%})")
    print(f"  Removed: {suspicious.sum()} likely mislabeled")
    print(f"  Clean dataset: {len(y_clean)} ({(y_clean == 1).sum()} closed, {(y_clean == 0).sum()} open)")

    clf.fit(X_clean, y_clean, xgb__sample_weight=w_clean)
    _run_cv(clf, X_clean, y_clean, w_clean, label="CLEANED DATA")
    return clf


def split_test():
    """
    Baseline evaluation: 80/20 stratified random split across all cities.
    CV on all data, then single-split for feature importance and threshold analysis.
    """
    X, y, weights, cities = get_data()
    X, y, weights = _prepare(X, y, weights)
    print(X['url_alive'].value_counts())

    clf = create_pipeline(X)
    _run_cv(clf, X, y, weights, label="ALL DATA")

    X_train, X_test, y_train, y_test, w_train, _, cities_train, cities_test = train_test_split(
        X, y, weights, cities, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train, xgb__sample_weight=w_train)
    _print_features(clf)
    _eval(clf, X_test, y_test, cities_test)


if __name__ == "__main__":
    split_test()
