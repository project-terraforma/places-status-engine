from hybrid import get_data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def main():
    X, y, weights = get_data()
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
    X = X.drop(columns=['address_postcode', 'name_len', 'category_primary', 'address_locality', 'category_alternates'])
    # Categorical columns
    cat_cols = ["address_region", "address_country"]
    cat_cols = [c for c in cat_cols if c in X.columns]
    
    # Numeric columns (everything else)
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    print(f"Categorical: {cat_cols}")
    print(f"Numeric: {num_cols}")
    
    preprocess = ColumnTransformer(
        transformers=[
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
            scale_pos_weight=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        ))
    ])
    
    # === 5-FOLD CROSS VALIDATION ===
    print("\n=== 5-FOLD STRATIFIED CV ===")
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
    
    # === SINGLE SPLIT (for feature importance + threshold) ===
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
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
    print("\n=== THRESHOLD ANALYSIS ===")
    from sklearn.metrics import precision_score, recall_score, f1_score
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred_t = (y_proba >= thresh).astype(int)
        p = precision_score(y_test, y_pred_t)
        r = recall_score(y_test, y_pred_t)
        f1 = f1_score(y_test, y_pred_t)
        print(f"Threshold {thresh}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
    
    y_pred = clf.predict(X_test)
    print("\n=== DEFAULT (threshold=0.5) ===")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
