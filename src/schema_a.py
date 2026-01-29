import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def _is_nan(x: Any) -> bool:
    """Check if x is NaN/None, handling numpy arrays safely."""
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
    """
    Overture columns stores JSON as strings. This parses them safely.
    - If x is already a dict/list -> return as-is
    - If x is None/NaN -> return None
    - If x is a string -> try json.loads, otherwise None
    """
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
    """Get length of x safely, returning 0 for None/NaN/invalid types."""
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
    """
    For objects like {"primary": "...", ...}
    """
    if not isinstance(obj, dict):
        return ""
    v = obj.get(key)
    if v is None:
        return ""
    return str(v)


def get_alternate_count(categories_obj: Any) -> int:
    """
    For categories like {"primary": "...", "alternate": ["..", ".."]}
    """
    if not isinstance(categories_obj, dict):
        return 0
    alt = categories_obj.get("alternate")
    if isinstance(alt, list):
        return len(alt)
    return 0


def get_first_address(addresses_obj: Any) -> Dict[str, Any]:
    """
    For addresses like [{"freeform":"...", "locality":"...", ...}, ...]
    """
    if isinstance(addresses_obj, list) and len(addresses_obj) > 0 and isinstance(addresses_obj[0], dict):
        return addresses_obj[0]
    return {}


def postcode_prefix(postcode: str, n: int = 3) -> str:
    pc = normalize_text(postcode)
    pc = re.sub(r"[^0-9a-z]", "", pc)
    if not pc:
        return ""
    return pc[:n]


# ----------------------------
# Schema A extraction
# ----------------------------
def extract_schema_a(
    df: pd.DataFrame,
    *,
    use_base_fields: bool = False,
    postcode_prefix_len: int = 3,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    prefix = "base_" if use_base_fields else ""

    def col(name: str) -> str:
        return f"{prefix}{name}"

    # Parse JSON columns once
    parsed = {}
    for c in [
        col("sources"), col("names"), col("categories"), col("websites"), col("socials"),
        col("emails"), col("phones"), col("brand"), col("addresses")
    ]:
        if c in df.columns:
            parsed[c] = df[c].map(parse_json_maybe)
        else:
            parsed[c] = pd.Series([None] * len(df), index=df.index)

    # Start building features
    out = pd.DataFrame(index=df.index)

    if "id" in df.columns:
        out["overture_id"] = df["id"].astype(str)
    if "base_id" in df.columns:
        out["base_id"] = df["base_id"].astype(str)

    # confidence
    conf_col = col("confidence")
    if conf_col in df.columns:
        out["confidence"] = pd.to_numeric(df[conf_col], errors="coerce")
    else:
        out["confidence"] = np.nan

    # source_count
    out["source_count"] = parsed[col("sources")].map(safe_len)

    # names
    names_obj = parsed[col("names")]
    out["name_primary"] = names_obj.map(lambda o: normalize_text(get_primary_from_obj(o, "primary")))
    out["name_len"] = out["name_primary"].map(len)

    # categories
    cats_obj = parsed[col("categories")]
    out["category_primary"] = cats_obj.map(lambda o: normalize_text(get_primary_from_obj(o, "primary")))
    out["alternate_category_count"] = cats_obj.map(get_alternate_count)

    # websites/socials/emails/phones presence + counts
    websites = parsed[col("websites")]
    socials = parsed[col("socials")]
    emails = parsed[col("emails")]
    phones = parsed[col("phones")]

    out["website_count"] = websites.map(safe_len)
    out["social_count"] = socials.map(safe_len)
    out["email_count"] = emails.map(safe_len)
    out["phone_count"] = phones.map(safe_len)

    out["has_website"] = (out["website_count"] > 0).astype(int)
    out["has_social"] = (out["social_count"] > 0).astype(int)
    out["has_email"] = (out["email_count"] > 0).astype(int)
    out["has_phone"] = (out["phone_count"] > 0).astype(int)

    # brand
    brand_obj = parsed[col("brand")]
    out["has_brand"] = brand_obj.map(lambda o: int(isinstance(o, dict) and len(o) > 0))

    # address features (use first address record)
    addresses_obj = parsed[col("addresses")]
    addr0 = addresses_obj.map(get_first_address)

    out["country"] = addr0.map(lambda a: normalize_text(a.get("country")) if isinstance(a, dict) else "")
    out["region"] = addr0.map(lambda a: normalize_text(a.get("region")) if isinstance(a, dict) else "")
    out["locality"] = addr0.map(lambda a: normalize_text(a.get("locality")) if isinstance(a, dict) else "")
    out["postcode_prefix"] = addr0.map(
        lambda a: postcode_prefix(str(a.get("postcode", "")), n=postcode_prefix_len) if isinstance(a, dict) else ""
    )

    freeform = addr0.map(lambda a: a.get("freeform") if isinstance(a, dict) else "")
    out["address_freeform_len"] = freeform.map(lambda s: len(normalize_text(s)))

    # has street hueristic
    out["has_street"] = (out["address_freeform_len"] > 0).astype(int)

    # Labels if present
    y = None
    if "label" in df.columns:
        # Your parquet has label values like 1.0/0.0
        y = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

    # Clean up missing strings -> empty strings
    for c in ["name_primary", "category_primary", "country", "region", "locality", "postcode_prefix"]:
        out[c] = out[c].fillna("")


    return out, y


def get_schema():
    # 1) Load  parquet
    df = pd.read_parquet("../assets/sample_3k_overture_places.parquet")

    # 2) Extract Schema A features 
    schema_a_df, y = extract_schema_a(df, use_base_fields=False, postcode_prefix_len=3)

    print("Schema A columns:", list(schema_a_df.columns))
    if y is not None:
        print("Label distribution:\n", y.value_counts(dropna=False))

    # 3)clean X for sklearn
    X = schema_a_df.drop(columns=[c for c in ["overture_id", "base_id"] if c in schema_a_df.columns])

    return X,y
