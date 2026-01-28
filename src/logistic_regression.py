from schema_a import get_schema

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support

def main():
    X,y = get_schema()
    y = y.astype(int).to_numpy()

    #tf-idf
    name_col = "name_primary"
    category_col = "category_primary"
    #One Hot Encode
    categorical_cols = ["country", "region"]  # add "locality" later

    numeric_cols = [
        "confidence", "source_count", "name_len", "alternate_category_count",
        "website_count", "social_count", "email_count", "phone_count",
        "has_website", "has_social", "has_email", "has_phone", "has_brand",
        "address_freeform_len", "has_street"
    ]

    preprocess = ColumnTransformer(
        transformers=[
            # TF-IDF must be applied to each text column separately
            ("tfidf_name", TfidfVectorizer(ngram_range=(1,2)), name_col),
            ("tfidf_category", TfidfVectorizer(ngram_range=(1,2)), category_col),
            ("cat-ohe", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", Pipeline(steps=[("scale", StandardScaler(with_mean=False))]), numeric_cols),
        ],
        remainder="drop"
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("log-reg", LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=42
            )),

        ]
    )

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