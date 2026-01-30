from schema_a import get_schema
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    X, y = get_schema(include_base=True, include_deltas=True)
    y = y.astype(int).to_numpy()

    # One Hot Encode categorical columns
    categorical_cols = ["country", "region", "category_primary"]

    # Current numeric features
    current_numeric = [
        "confidence", "source_count", "name_len", "alternate_category_count",
        "website_count", "social_count", "email_count", "phone_count",
        "has_website", "has_social", "has_email", "has_phone", "has_brand",
        "address_freeform_len", "has_street"
    ]
    
    # Base numeric features
    base_numeric = [
        "confidence_base", "source_count_base", "name_len_base", "alternate_category_count_base",
        "website_count_base", "social_count_base", "email_count_base", "phone_count_base",
        "has_website_base", "has_social_base", "has_email_base", "has_phone_base", "has_brand_base",
        "address_freeform_len_base", "has_street_base"
    ]
    
    # Delta features (these are likely the most predictive!)
    delta_numeric = [
        "confidence_delta", "source_count_delta", "name_len_delta", "alternate_category_count_delta",
        "website_count_delta", "social_count_delta", "email_count_delta", "phone_count_delta",
        "address_freeform_len_delta"
    ]
    
    # Binary change features
    change_binary = [
        "lost_website", "lost_social", "lost_email", "lost_phone", "lost_brand", "lost_street",
        "gained_website", "gained_social", "gained_email", "gained_phone", "gained_brand",
        "name_changed", "category_changed", "country_changed", "locality_changed"
    ]
    
    # Combine all numeric columns
    numeric_cols = delta_numeric + change_binary

    preprocess = ColumnTransformer(
        transformers=[

            ("cat-ohe", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop"
    )

    rfc = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        n_jobs=1,
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("rfc", rfc)
    ])

    X_train, X_test, Y_train, y_test = train_test_split(X,y, test_size=.2, random_state=None, stratify=y)

    print(X_test.shape)
    print(X_train.shape)
    print(y_test.shape)
    print(Y_train.shape)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    main()

