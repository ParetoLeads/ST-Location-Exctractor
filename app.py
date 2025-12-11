import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import shapely.geometry as geom
import streamlit as st


# You can override the geocoder endpoint if the default is blocked on your host.
GEOCODER_URL = os.getenv("GEOCODER_URL", "https://nominatim.openstreetmap.org/search")
_raw_key = os.getenv("GEOCODER_API_KEY", "")
GEOCODER_API_KEY = _raw_key.strip() if _raw_key else None  # e.g., for https://geocode.maps.co
USER_AGENT = "location-filter-app/1.0"
APP_VERSION = "v1.04"


st.set_page_config(page_title="Location Search Term Filter", layout="wide")
st.title(f"Location Search Term Filter ({APP_VERSION})")
st.write(
    "Upload a CSV of search terms with impressions, enter a target area (e.g., "
    "`Melbourne, Australia`), and get keep vs exclude location suggestions."
)


# -------- Helpers -------- #
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lowered = {c: c.lower() for c in df.columns}
    col_map: Dict[str, str] = {}

    for candidate in ["search_term", "search term", "term", "query"]:
        match = next((c for c in df.columns if lowered[c] == candidate), None)
        if match:
            col_map[match] = "search_term"
            break
    for candidate in ["impressions", "impr", "impression", "impr."]:
        match = next((c for c in df.columns if lowered[c] == candidate), None)
        if match:
            col_map[match] = "impressions"
            break

    missing = {"search_term", "impressions"} - set(col_map.values())
    if missing:
        raise ValueError(
            "CSV must include columns for search term and impressions. "
            "Add headers like 'search_term' and 'impressions'."
        )

    df = df.rename(columns=col_map)
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0)
    df["search_term"] = df["search_term"].astype(str)
    return df[["search_term", "impressions"]]


@lru_cache(maxsize=512)
def geocode(query: str, polygon: bool = False) -> Tuple[Optional[Dict], Optional[str]]:
    # Detect which geocoder API we're using
    is_maps_co = "geocode.maps.co" in GEOCODER_URL
    
    params = {"q": query}
    
    if is_maps_co:
        # geocode.maps.co format - simpler, just needs q and api_key
        # Note: maps.co doesn't support polygon requests, so ignore polygon param
        if GEOCODER_API_KEY:
            # Strip any whitespace that might have been copied
            clean_key = GEOCODER_API_KEY.strip()
            params["api_key"] = clean_key
        else:
            return None, "API key is required for geocode.maps.co"
    else:
        # Nominatim format
        params.update({
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
        })
        if polygon:
            params["polygon_geojson"] = 1
    
    headers = {"User-Agent": USER_AGENT}
    last_error = None
    
    for attempt in range(3):
        try:
            resp = requests.get(
                GEOCODER_URL,
                params=params,
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 200:
                error_text = resp.text[:300]
                last_error = f"HTTP {resp.status_code}: {error_text}"
                # Don't retry on 401 (invalid key) - it won't work
                if resp.status_code == 401:
                    break
                time.sleep(0.5)
                continue
            data = resp.json()
            if not data:
                last_error = "Empty response"
                return None, last_error
            
            result = data[0]
            
            # Normalize response format - maps.co returns lat/lon as strings
            if is_maps_co:
                if "lat" in result and isinstance(result["lat"], str):
                    result["lat"] = float(result["lat"])
                if "lon" in result and isinstance(result["lon"], str):
                    result["lon"] = float(result["lon"])
                # maps.co uses "display_name" or "formatted" for the name
                if "display_name" not in result and "formatted" in result:
                    result["display_name"] = result["formatted"]
            
            return result, None
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(0.5)
            continue
        except (ValueError, KeyError, IndexError) as exc:
            last_error = f"Parse error: {exc}"
            time.sleep(0.5)
            continue
    
    return None, last_error


def bbox_to_polygon(bbox: List[str]) -> Optional[geom.Polygon]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        south, north, west, east = map(float, bbox)
        return geom.Polygon(
            [
                (west, south),
                (east, south),
                (east, north),
                (west, north),
                (west, south),
            ]
        )
    except Exception:
        return None


def location_shape(feature: Dict) -> Optional[geom.base.BaseGeometry]:
    if feature.get("geojson"):
        try:
            return geom.shape(feature["geojson"])
        except Exception:
            pass
    if feature.get("boundingbox"):
        return bbox_to_polygon(feature["boundingbox"])
    if feature.get("lat") and feature.get("lon"):
        return geom.Point(float(feature["lon"]), float(feature["lat"]))
    return None


def is_inside(candidate: Dict, target_geom: geom.base.BaseGeometry) -> Optional[bool]:
    cand_geom = location_shape(candidate)
    if cand_geom is None:
        return None
    if isinstance(cand_geom, geom.Point):
        return target_geom.contains(cand_geom)
    return target_geom.intersects(cand_geom)


def extract_locations(df: pd.DataFrame, target_geom: geom.base.BaseGeometry) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        term = row["search_term"]
        impressions = row["impressions"]

        result, _ = geocode(term, polygon=False)
        if result is None:
            records.append(
                {
                    "search_term": term,
                    "impressions": impressions,
                    "location_name": None,
                    "lat": None,
                    "lon": None,
                    "status": "unmatched",
                }
            )
            continue

        inside = is_inside(result, target_geom)
        status = "keep" if inside else "exclude" if inside is False else "unmatched"

        records.append(
            {
                "search_term": term,
                "impressions": impressions,
                "location_name": result.get("display_name", ""),
                "lat": result.get("lat"),
                "lon": result.get("lon"),
                "status": status,
            }
        )
    return pd.DataFrame(records)


def geocode_target_area(area_text: str) -> Tuple[Optional[Dict], Optional[geom.base.BaseGeometry], List[str]]:
    result, err = geocode(area_text, polygon=True)
    if not result:
        error_msg = f"No match found for '{area_text}'."
        if err:
            error_msg += f" {err}"
            # Add helpful hint for 401 errors
            if "401" in err or "Invalid API Key" in err:
                error_msg += " Please check your API key in Streamlit Secrets (Settings → Secrets). Make sure there are no extra spaces."
        return None, None, [error_msg]

    shape = location_shape(result)
    warnings = []
    if shape is None:
        warnings.append("Could not derive geometry for the target area.")
    return result, shape, warnings


def download_link(label: str, df: pd.DataFrame, filename: str, key: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv", key=key)


# -------- UI -------- #
st.sidebar.header("How to use")
st.sidebar.markdown(
    "- Upload CSV with columns `search_term` and `impressions`.\n"
    "- Enter target area (e.g., `Melbourne, Australia`).\n"
    "- We use OpenStreetMap Nominatim (free; rate-limited). Cached per session.\n"
    "- Output shows keep vs exclude by location."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
target_area = st.text_input("Target area (city/region/country)", value="Melbourne, Australia")
throttle = st.slider("Geocode pause (seconds) to ease rate limits", 0.0, 2.0, 0.2, 0.1)

st.caption(f"Geocoder URL: {GEOCODER_URL}")
st.caption(f"API key present: {'yes' if GEOCODER_API_KEY else 'no'}")
if GEOCODER_API_KEY:
    # Show key length for debugging (without exposing the actual key)
    st.caption(f"API key length: {len(GEOCODER_API_KEY)} characters")
    # Show first/last 4 chars for verification (without exposing full key)
    if len(GEOCODER_API_KEY) >= 8:
        st.caption(f"API key preview: {GEOCODER_API_KEY[:4]}...{GEOCODER_API_KEY[-4:]}")
st.caption(f"App version: {APP_VERSION}")

# Debug section - test API key
if GEOCODER_API_KEY and "geocode.maps.co" in GEOCODER_URL:
    with st.expander("🔧 Debug: Test API Key"):
        if st.button("Test API Key with 'Miami, FL'"):
            test_url = f"{GEOCODER_URL}?q=Miami, FL&api_key={GEOCODER_API_KEY}"
            st.code(test_url, language=None)
            try:
                resp = requests.get(GEOCODER_URL, params={"q": "Miami, FL", "api_key": GEOCODER_API_KEY}, timeout=10)
                st.write(f"**Status Code:** {resp.status_code}")
                st.write(f"**Response:** {resp.text[:500]}")
                if resp.status_code == 200:
                    st.success("✅ API key works!")
                else:
                    st.error(f"❌ API key failed: {resp.text[:200]}")
            except Exception as e:
                st.error(f"Error: {e}")


if uploaded and target_area.strip():
    with st.spinner("Reading CSV..."):
        try:
            df_input = pd.read_csv(uploaded)
            df_input = _standardize_columns(df_input)
        except Exception as exc:
            st.error(f"Problem reading CSV: {exc}")
            st.stop()

    target_result, target_geom, target_warnings = geocode_target_area(target_area)
    if target_warnings:
        for w in target_warnings:
            st.warning(w)
    if target_geom is None:
        st.error("Could not geocode the target area. Try a more specific name.")
        st.stop()

    st.success(f"Target area matched: {target_result.get('display_name')}")

    results: List[pd.DataFrame] = []
    with st.spinner("Geocoding search terms (cached)..."):
        # Reset cache between runs if needed
        geocode.cache_clear()
        # Iterate with throttling
        all_records = []
        for idx, row in df_input.iterrows():
            term = row["search_term"]
            impressions = row["impressions"]
            result, _ = geocode(term, polygon=False)
            if throttle:
                time.sleep(throttle)
            if result is None:
                all_records.append(
                    {
                        "search_term": term,
                        "impressions": impressions,
                        "location_name": None,
                        "lat": None,
                        "lon": None,
                        "status": "unmatched",
                    }
                )
                continue

            inside = is_inside(result, target_geom)
            status = "keep" if inside else "exclude" if inside is False else "unmatched"
            all_records.append(
                {
                    "search_term": term,
                    "impressions": impressions,
                    "location_name": result.get("display_name", ""),
                    "lat": result.get("lat"),
                    "lon": result.get("lon"),
                    "status": status,
                }
            )

        results_df = pd.DataFrame(all_records)

    keep_df = results_df[results_df["status"] == "keep"]
    exclude_df = results_df[results_df["status"] == "exclude"]
    unmatched_df = results_df[results_df["status"] == "unmatched"]

    agg_keep = (
        keep_df.groupby("location_name", dropna=True)["impressions"].sum().reset_index()
        if not keep_df.empty
        else pd.DataFrame(columns=["location_name", "impressions"])
    )
    agg_exclude = (
        exclude_df.groupby("location_name", dropna=True)["impressions"].sum().reset_index()
        if not exclude_df.empty
        else pd.DataFrame(columns=["location_name", "impressions"])
    )

    st.subheader("Suggested keeps (inside target area)")
    st.dataframe(agg_keep.sort_values("impressions", ascending=False))
    download_link("Download keeps (agg)", agg_keep, "keeps.csv", "keeps-agg")

    st.subheader("Suggested excludes (outside target area)")
    st.dataframe(agg_exclude.sort_values("impressions", ascending=False))
    download_link("Download excludes (agg)", agg_exclude, "excludes.csv", "excludes-agg")

    st.subheader("Row-level results")
    st.dataframe(results_df)
    download_link("Download full results", results_df, "results.csv", "full-results")

    if not unmatched_df.empty:
        st.info(
            f"{len(unmatched_df)} terms could not be matched. "
            "Try using more explicit place names or lowering throttle if rate-limited."
        )
else:
    st.info("Upload a CSV and set a target area to begin.")

