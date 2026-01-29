
from tkinter import ON
from schema_a import get_schema
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    X,y = get_schema()
    y = y.astype(int).to_numpy()

    #One Hot Encode
    categorical_cols = ["country", "region", "category_primary"]

    numeric_cols = [
        "confidence", "source_count", "name_len", "alternate_category_count",
        "website_count", "social_count", "email_count", "phone_count",
        "has_website", "has_social", "has_email", "has_phone", "has_brand",
        "address_freeform_len", "has_street"
    ]

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
        max_depth=10,
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

