import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _is_nan(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, (list, dict, str)):
        return False
    if isinstance(x, np.ndarray):
        return False  # Arrays are not considered NaN
    try:
        result = pd.isna(x)
        # pd.isna can return a scalar bool or an array
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return False
    except Exception:
        return False


def parse_json_maybe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()  # Convert numpy array to list
    if _is_nan(x):
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def safe_len(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, (list, dict, str)):
        return len(x)
    if isinstance(x, np.ndarray):
        return len(x)
    if _is_nan(x):
        return 0
    return 0


def get_primary_from_obj(obj: Any, key: str = "primary") -> str:

    if not isinstance(obj, dict):
        return ""
    v = obj.get(key)
    if v is None:
        return ""
    return str(v)


def get_alternate_count(categories_obj: Any) -> int:
    if not isinstance(categories_obj, dict):
        return 0
    alt = categories_obj.get("alternate")
    if isinstance(alt, list):
        return len(alt)
    return 0


def get_first_address(addresses_obj: Any) -> Dict[str, Any]:
    if isinstance(addresses_obj, list) and len(addresses_obj) > 0 and isinstance(addresses_obj[0], dict):
        return addresses_obj[0]
    return {}


def postcode_prefix(postcode: str, n: int = 3) -> str:
    pc = normalize_text(postcode)
    pc = re.sub(r"[^0-9a-z]", "", pc)
    if not pc:
        return ""
    return pc[:n]


# Schema A extraction (with delta features)

def _extract_features_for_prefix(df: pd.DataFrame, parsed: dict, prefix: str, postcode_prefix_len: int) -> pd.DataFrame:
    
    def col(name: str) -> str:
        return f"{prefix}{name}"
    
    suffix = "_base" if prefix == "base_" else ""
    out = pd.DataFrame(index=df.index)
    
    # confidence
    conf_col = col("confidence")
    if conf_col in df.columns:
        out[f"confidence{suffix}"] = pd.to_numeric(df[conf_col], errors="coerce")
    else:
        out[f"confidence{suffix}"] = np.nan

    # source_count
    out[f"source_count{suffix}"] = parsed[col("sources")].map(safe_len)

    # names
    names_obj = parsed[col("names")]
    out[f"name_primary{suffix}"] = names_obj.map(lambda o: normalize_text(get_primary_from_obj(o, "primary")))
    out[f"name_len{suffix}"] = out[f"name_primary{suffix}"].map(len)

    # categories
    cats_obj = parsed[col("categories")]
    out[f"category_primary{suffix}"] = cats_obj.map(lambda o: normalize_text(get_primary_from_obj(o, "primary")))
    out[f"alternate_category_count{suffix}"] = cats_obj.map(get_alternate_count)

    # websites/socials/emails/phones presence + counts
    websites = parsed[col("websites")]
    socials = parsed[col("socials")]
    emails = parsed[col("emails")]
    phones = parsed[col("phones")]

    out[f"website_count{suffix}"] = websites.map(safe_len)
    out[f"social_count{suffix}"] = socials.map(safe_len)
    out[f"email_count{suffix}"] = emails.map(safe_len)
    out[f"phone_count{suffix}"] = phones.map(safe_len)

    out[f"has_website{suffix}"] = (out[f"website_count{suffix}"] > 0).astype(int)
    out[f"has_social{suffix}"] = (out[f"social_count{suffix}"] > 0).astype(int)
    out[f"has_email{suffix}"] = (out[f"email_count{suffix}"] > 0).astype(int)
    out[f"has_phone{suffix}"] = (out[f"phone_count{suffix}"] > 0).astype(int)

    # brand
    brand_obj = parsed[col("brand")]
    out[f"has_brand{suffix}"] = brand_obj.map(lambda o: int(isinstance(o, dict) and len(o) > 0))

    # address features
    addresses_obj = parsed[col("addresses")]
    addr0 = addresses_obj.map(get_first_address)

    out[f"country{suffix}"] = addr0.map(lambda a: normalize_text(a.get("country")) if isinstance(a, dict) else "")
    out[f"region{suffix}"] = addr0.map(lambda a: normalize_text(a.get("region")) if isinstance(a, dict) else "")
    out[f"locality{suffix}"] = addr0.map(lambda a: normalize_text(a.get("locality")) if isinstance(a, dict) else "")
    out[f"postcode_prefix{suffix}"] = addr0.map(
        lambda a: postcode_prefix(str(a.get("postcode", "")), n=postcode_prefix_len) if isinstance(a, dict) else ""
    )

    freeform = addr0.map(lambda a: a.get("freeform") if isinstance(a, dict) else "")
    out[f"address_freeform_len{suffix}"] = freeform.map(lambda s: len(normalize_text(s)))
    out[f"has_street{suffix}"] = (out[f"address_freeform_len{suffix}"] > 0).astype(int)

    return out


def extract_schema_a(
    df: pd.DataFrame,
    *,
    include_base: bool = True,
    include_deltas: bool = True,
    postcode_prefix_len: int = 3,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    
    # Parse JSON columns for both current and base
    parsed = {}
    for prefix in ["", "base_"]:
        for name in ["sources", "names", "categories", "websites", "socials", "emails", "phones", "brand", "addresses"]:
            c = f"{prefix}{name}"
            if c in df.columns:
                parsed[c] = df[c].map(parse_json_maybe)
            else:
                parsed[c] = pd.Series([None] * len(df), index=df.index)

    # Start building features
    out = pd.DataFrame(index=df.index)

    # IDs
    if "id" in df.columns:
        out["overture_id"] = df["id"].astype(str)
    if "base_id" in df.columns:
        out["base_id"] = df["base_id"].astype(str)

    # Extract current features
    current_features = _extract_features_for_prefix(df, parsed, "", postcode_prefix_len)
    out = pd.concat([out, current_features], axis=1)

    # Extract base features if requested
    if include_base:
        base_features = _extract_features_for_prefix(df, parsed, "base_", postcode_prefix_len)
        out = pd.concat([out, base_features], axis=1)

    # Compute delta features if requested
    if include_deltas and include_base:
        # Numeric deltas (current - base)
        out["confidence_delta"] = out["confidence"] - out["confidence_base"]
        out["source_count_delta"] = out["source_count"] - out["source_count_base"]
        out["name_len_delta"] = out["name_len"] - out["name_len_base"]
        out["alternate_category_count_delta"] = out["alternate_category_count"] - out["alternate_category_count_base"]
        out["website_count_delta"] = out["website_count"] - out["website_count_base"]
        out["social_count_delta"] = out["social_count"] - out["social_count_base"]
        out["email_count_delta"] = out["email_count"] - out["email_count_base"]
        out["phone_count_delta"] = out["phone_count"] - out["phone_count_base"]
        out["address_freeform_len_delta"] = out["address_freeform_len"] - out["address_freeform_len_base"]
        
        # Binary "lost" features (had it before, don't have it now)
        out["lost_website"] = ((out["has_website_base"] == 1) & (out["has_website"] == 0)).astype(int)
        out["lost_social"] = ((out["has_social_base"] == 1) & (out["has_social"] == 0)).astype(int)
        out["lost_email"] = ((out["has_email_base"] == 1) & (out["has_email"] == 0)).astype(int)
        out["lost_phone"] = ((out["has_phone_base"] == 1) & (out["has_phone"] == 0)).astype(int)
        out["lost_brand"] = ((out["has_brand_base"] == 1) & (out["has_brand"] == 0)).astype(int)
        out["lost_street"] = ((out["has_street_base"] == 1) & (out["has_street"] == 0)).astype(int)
        
        # Binary "gained" features (didn't have it before, have it now)
        out["gained_website"] = ((out["has_website_base"] == 0) & (out["has_website"] == 1)).astype(int)
        out["gained_social"] = ((out["has_social_base"] == 0) & (out["has_social"] == 1)).astype(int)
        out["gained_email"] = ((out["has_email_base"] == 0) & (out["has_email"] == 1)).astype(int)
        out["gained_phone"] = ((out["has_phone_base"] == 0) & (out["has_phone"] == 1)).astype(int)
        out["gained_brand"] = ((out["has_brand_base"] == 0) & (out["has_brand"] == 1)).astype(int)
        
        # Name/category changed
        out["name_changed"] = (out["name_primary"] != out["name_primary_base"]).astype(int)
        out["category_changed"] = (out["category_primary"] != out["category_primary_base"]).astype(int)
        out["country_changed"] = (out["country"] != out["country_base"]).astype(int)
        out["locality_changed"] = (out["locality"] != out["locality_base"]).astype(int)

    # Labels if present
    y = None
    if "label" in df.columns:
        y = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

    # Clean up missing strings -> empty strings
    text_cols = [c for c in out.columns if any(x in c for x in ["name_primary", "category_primary", "country", "region", "locality", "postcode_prefix"])]
    for c in text_cols:
        out[c] = out[c].fillna("")

    # Fill NaN values in numeric columns with 0
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].fillna(0)
    
    # Replace any inf values with 0
    out = out.replace([np.inf, -np.inf], 0)

    return out, y


def get_schema():
    #Load parquet
    df = pd.read_parquet("../assets/sample_3k_overture_places.parquet")

    #Extract Schema A features with base and delta features
    schema_a_df, y = extract_schema_a(
        df, 
        include_base=True, 
        include_deltas=True,
        postcode_prefix_len=3
    )

    print("Schema A columns:", list(schema_a_df.columns))
    print(f"Total features: {len(schema_a_df.columns)}")
    if y is not None:
        print("Label distribution:\n", y.value_counts(dropna=False))

    # 3) Clean X for sklearn (drop ID columns)
    X = schema_a_df.drop(columns=[c for c in ["overture_id", "base_id"] if c in schema_a_df.columns])

    return X, y
