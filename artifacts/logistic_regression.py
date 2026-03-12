from utils.schema_a import get_schema

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
    X, y = get_schema()
    y = y.astype(int).to_numpy()

    # TF-IDF on text columns
    name_col = "name_primary"
    category_col = "category_primary"
    
    # One Hot Encode categorical columns
    categorical_cols = ["country", "region"]

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
            ("tfidf_name", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2), name_col),
            ("tfidf_category", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2), category_col),
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