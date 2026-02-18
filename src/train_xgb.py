from hybrid import get_data, get_unlabeled
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, TargetEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

def main():
    X, y, weights, cities = get_data()
    y = y.to_numpy()
    
    # Multi-hot encode source_datasets before pipeline
    if 'source_datasets' in X.columns:
        mlb = MultiLabelBinarizer()
        # Ensure lists (handle None/NaN)
        sources = X['source_datasets'].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else [])
        sources_encoded = mlb.fit_transform(sources)
        sources_df = pd.DataFrame(
            sources_encoded, 
            columns=[f"src_{c}" for c in mlb.classes_],
            index=X.index
        )
        X = pd.concat([X.drop(columns=['source_datasets']), sources_df], axis=1)
        print(f"Multi-hot encoded sources: {list(mlb.classes_)}")
    X = X.drop(columns=[
    'address_postcode','name_len', 'address_locality', 
    'category_alternates', 'address_region', 'address_country'
    ])
    
    target_cols = ["category_primary"]

    # Categorical columns
    cat_cols = ["address_region", "address_country"]
    cat_cols = [c for c in cat_cols if c in X.columns]
    
    # Numeric columns (everything else)
    num_cols = [c for c in X.columns if c not in cat_cols and c not in target_cols]
    
    print(f"Categorical: {cat_cols}")
    print(f"Numeric: {num_cols}")
    
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
            scale_pos_weight=3,
            reg_lambda=1.0,
            reg_alpha=0.0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        ))
    ])
    
    # 5-FOLD CROSS VALIDATION
    print("\n 5-FOLD STRATIFIED CV ")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        clf, X, y,
        cv=cv,
        scoring=['precision', 'recall', 'f1', 'average_precision'],
        params={'xgb__sample_weight': weights},
        return_train_score=False
    )
    
    print(f"Precision: {cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}  {cv_results['test_precision']}")
    print(f"Recall:    {cv_results['test_recall'].mean():.3f} ± {cv_results['test_recall'].std():.3f}  {cv_results['test_recall']}")
    print(f"F1:        {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}  {cv_results['test_f1']}")
    print(f"PR-AUC:    {cv_results['test_average_precision'].mean():.3f} ± {cv_results['test_average_precision'].std():.3f}")
    
    #SINGLE SPLIT (for feature importance + threshold)
    X_train, X_test, y_train, y_test, w_train, _, cities_train, cities_test = train_test_split(
        X, y, weights, cities, test_size=0.2, random_state=42, stratify=y
    )
    
    clf.fit(X_train, y_train, xgb__sample_weight=w_train)
    
    # Feature importance
    feat_names = clf.named_steps['preprocess'].get_feature_names_out()
    feat_imp = clf.named_steps['xgb'].feature_importances_
    top_idx = np.argsort(feat_imp)[::-1][:15]
    print("\nTop 15 features:")
    for i in top_idx:
        print(f"  {feat_names[i]}: {feat_imp[i]:.4f}")
    
    # Threshold analysis
    y_proba = clf.predict_proba(X_test)[:, 1]
    print("\nTHRESHOLD ANALYSIS")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred_t = (y_proba >= thresh).astype(int)
        p = precision_score(y_test, y_pred_t)
        r = recall_score(y_test, y_pred_t)
        f1 = f1_score(y_test, y_pred_t)
        print(f"Threshold {thresh}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
    
    y_pred = clf.predict(X_test)
    print("\nDEFAULT (threshold=0.5)")
    print(classification_report(y_test, y_pred, digits=3))

    # Per-city evaluation
    print("\n PER-CITY EVALUATION ")
    y_proba_all = clf.predict_proba(X_test)[:, 1]
    for city in sorted(set(cities_test)):
        mask = cities_test == city
        y_city = y_test[mask]
        proba_city = y_proba_all[mask]
        pred_city = (proba_city >= 0.5).astype(int)
        n_total = mask.sum()
        n_closed = y_city.sum()
        p = precision_score(y_city, pred_city, zero_division=0)
        r = recall_score(y_city, pred_city, zero_division=0)
        f1 = f1_score(y_city, pred_city, zero_division=0)
        print(f"  {city}: n={n_total}, closed={n_closed} ({n_closed/n_total:.1%}), P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    # =========================================================================
    # LABEL CLEANING
    # =========================================================================
    # WHY THIS WORKS:
    # Our audit showed FSQ "open" labels are wrong 40% of the time — those places
    # are actually closed. The model learns the REAL pattern from the majority of
    # correct labels. When it confidently says "this is closed" but the label says
    # "open", the model is probably right and the label is probably wrong.
    #
    # By removing these likely-mislabeled examples, we stop the model from being
    # penalized for correct predictions, and it sharpens its decision boundary.
    # =========================================================================
    print("\n LABEL CLEANING ")

    # Step 1: Train on ALL data (not just train split) to get best predictions
    clf.fit(X, y, xgb__sample_weight=weights)

    # Step 2: Score every "open" example — what does the model think?
    probs_all = clf.predict_proba(X)[:, 1]
    open_mask = (y == 0)  # FSQ says "open"
    open_probs = probs_all[open_mask]

    # Step 3: Find suspicious "open" labels — model says P(closed) > 0.7
    threshold = 0.85
    suspicious = open_probs > threshold
    n_suspicious = suspicious.sum()
    n_open = open_mask.sum()
    print(f"  'Open' examples: {n_open}")
    print(f"  Model thinks are closed (P>{threshold}): {n_suspicious} ({n_suspicious/n_open:.1%})")

    # Step 4: Remove suspicious examples from training data
    # Build a mask: keep all "closed" labels + keep "open" labels the model agrees with
    open_indices = np.where(open_mask)[0]
    suspicious_indices = open_indices[suspicious]
    clean_mask = np.ones(len(y), dtype=bool)
    clean_mask[suspicious_indices] = False

    X_clean = X.iloc[clean_mask].reset_index(drop=True)
    y_clean = y[clean_mask]
    w_clean = weights[clean_mask]
    cities_clean = cities[clean_mask]

    print(f"  Removed: {n_suspicious} likely mislabeled")
    print(f"  Clean dataset: {len(y_clean)} ({(y_clean==1).sum()} closed, {(y_clean==0).sum()} open)")

    # Step 5: Re-evaluate with cross-validation on clean data
    print("\n 5-FOLD CV ON CLEANED DATA ")
    cv_clean = cross_validate(
        clf, X_clean, y_clean,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=['precision', 'recall', 'f1', 'average_precision'],
        params={'xgb__sample_weight': w_clean},
        return_train_score=False
    )
    print(f"Precision: {cv_clean['test_precision'].mean():.3f} ± {cv_clean['test_precision'].std():.3f}")
    print(f"Recall:    {cv_clean['test_recall'].mean():.3f} ± {cv_clean['test_recall'].std():.3f}")
    print(f"F1:        {cv_clean['test_f1'].mean():.3f} ± {cv_clean['test_f1'].std():.3f}")
    print(f"PR-AUC:    {cv_clean['test_average_precision'].mean():.3f} ± {cv_clean['test_average_precision'].std():.3f}")


if __name__ == "__main__":
    main()
