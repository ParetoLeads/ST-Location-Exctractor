import os
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import shapely.geometry as geom
import streamlit as st

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


# You can override the geocoder endpoint if the default is blocked on your host.
GEOCODER_URL = os.getenv("GEOCODER_URL", "https://nominatim.openstreetmap.org/search")
_raw_key = os.getenv("GEOCODER_API_KEY", "")
GEOCODER_API_KEY = _raw_key.strip() if _raw_key else None  # e.g., for https://geocode.maps.co
_openai_key = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY = _openai_key.strip() if _openai_key else None
USER_AGENT = "location-filter-app/1.0"
APP_VERSION = "v1.08"

# Initialize OpenAI client if available and API key is set
openai_client = None
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None


st.set_page_config(page_title="Location Search Term Filter", layout="wide")
st.title(f"Location Search Term Filter ({APP_VERSION})")
st.write(
    "Upload a CSV of search terms with impressions, enter a target area (e.g., "
    "`Melbourne, Australia`), and get keep vs exclude location suggestions."
)


# -------- Helpers -------- #
@lru_cache(maxsize=1000)
def _extract_location_with_llm(search_term: str) -> Optional[str]:
    """
    Use OpenAI API to extract location name from search term.
    No hardcoding - LLM understands context and extracts locations intelligently.
    """
    if not search_term or not isinstance(search_term, str) or not search_term.strip():
        return None
    
    if not openai_client:
        return None
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap, good enough for this task
            messages=[
                {
                    "role": "system",
                    "content": "You are a location extraction expert. Extract ONLY the location name from search terms. Return just the location name, nothing else. If no location is found, return exactly 'NONE'. Do not include service terms, business names, or qualifiers. Examples: 'locksmith adelaide' -> 'adelaide', 'mawson lakes locksmith' -> 'mawson lakes', 'locksmith near me' -> 'NONE'."
                },
                {
                    "role": "user",
                    "content": f"Extract the location name from this search term: '{search_term}'\n\nReturn ONLY the location name (e.g., 'adelaide', 'mawson lakes', 'st agnes'). If no location exists, return 'NONE'."
                }
            ],
            temperature=0.1,  # Low temperature for consistent results
            max_tokens=50,
        )
        
        extracted = response.choices[0].message.content.strip()
        
        # Clean up the response
        extracted = extracted.strip('"').strip("'").strip()
        
        # Handle "NONE" or empty responses
        if not extracted or extracted.upper() == "NONE" or len(extracted) < 2:
            return None
        
        # Capitalize properly (first letter of each word)
        location = ' '.join(word.capitalize() for word in extracted.split())
        
        return location
        
    except Exception as e:
        # If OpenAI fails, return None (will be handled as unmatched)
        return None


def _extract_location_from_search_term(search_term: str) -> Optional[str]:
    """
    Extract location name from a search term using LLM.
    Falls back to simple extraction if OpenAI is not available.
    """
    # Try LLM first if available
    if openai_client:
        location = _extract_location_with_llm(search_term)
        if location:
            return location
    
    # Fallback: simple extraction (remove common words, find capitalized words)
    if not search_term or not isinstance(search_term, str):
        return None
    
    # Simple fallback - try to find capitalized words (likely locations)
    words = search_term.split()
    location_words = [w for w in words if w and w[0].isupper() and len(w) > 2 and w.lower() not in ["near", "me", "local", "now", "today"]]
    
    if location_words:
        return ' '.join(location_words)
    
    return None


def _read_and_clean_csv(uploaded_file) -> pd.DataFrame:
    """Read CSV file, handling Google Ads format with metadata rows."""
    # Try reading with skiprows to handle Google Ads format
    # Google Ads CSVs often have 2 metadata rows before headers
    try:
        # First, try reading normally
        df = pd.read_csv(uploaded_file)
        
        # Check if first row looks like metadata (not column headers)
        first_col = df.columns[0] if len(df.columns) > 0 else ""
        if isinstance(first_col, str) and ("report" in first_col.lower() or df.iloc[0, 0] == "Search terms report"):
            # Skip first 2 rows and read again
            uploaded_file.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded_file, skiprows=2)
    except Exception:
        # If that fails, try reading normally
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    
    # Remove rows that are totals or empty
    df = df[df.iloc[:, 0].astype(str).str.lower() != "total"]
    df = df.dropna(subset=[df.columns[0]])  # Drop rows where first column is empty
    
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and extract search_term and impressions."""
    lowered = {c: c.lower() for c in df.columns}
    col_map: Dict[str, str] = {}

    # Find search term column
    for candidate in ["search_term", "search term", "term", "query"]:
        match = next((c for c in df.columns if lowered[c] == candidate), None)
        if match:
            col_map[match] = "search_term"
            break
    
    # Find impressions column
    for candidate in ["impressions", "impr", "impression", "impr."]:
        match = next((c for c in df.columns if lowered[c] == candidate), None)
        if match:
            col_map[match] = "impressions"
            break

    missing = {"search_term", "impressions"} - set(col_map.values())
    if missing:
        available_cols = ", ".join(df.columns[:5])
        raise ValueError(
            f"CSV must include columns for search term and impressions.\n"
            f"Found columns: {available_cols}...\n"
            f"Expected: 'Search term' (or 'search_term') and 'Impr.' (or 'impressions')"
        )

    df = df.rename(columns=col_map)
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0)
    df["search_term"] = df["search_term"].astype(str).str.strip()
    
    # Remove empty search terms
    df = df[df["search_term"].str.len() > 0]
    
    return df[["search_term", "impressions"]]


def _aggregate_search_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate search terms by summing their impressions."""
    aggregated = df.groupby("search_term", as_index=False)["impressions"].sum()
    aggregated = aggregated.sort_values("impressions", ascending=False)
    return aggregated.reset_index(drop=True)


def _extract_locations_from_search_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract location names from search terms and create a new dataframe.
    Only includes rows where a location was successfully extracted.
    """
    results = []
    
    for _, row in df.iterrows():
        search_term = row["search_term"]
        impressions = row["impressions"]
        
        extracted_location = _extract_location_from_search_term(search_term)
        
        if extracted_location:
            results.append({
                "original_search_term": search_term,
                "extracted_location": extracted_location,
                "impressions": impressions,
            })
    
    if not results:
        return pd.DataFrame(columns=["original_search_term", "extracted_location", "impressions"])
    
    result_df = pd.DataFrame(results)
    
    # Aggregate by extracted location (same location might come from different search terms)
    aggregated = result_df.groupby("extracted_location", as_index=False).agg({
        "impressions": "sum",
        "original_search_term": lambda x: ", ".join(x.unique()[:3]) + ("..." if len(x.unique()) > 3 else "")
    })
    
    aggregated = aggregated.sort_values("impressions", ascending=False)
    return aggregated.reset_index(drop=True)


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
    "- Upload CSV (Google Ads format supported - auto-detects 'Search term' and 'Impr.' columns).\n"
    "- Duplicate search terms are automatically aggregated (impressions summed).\n"
    "- **🤖 Location extraction**: Uses AI (OpenAI) to intelligently extract location names from any search term - no hardcoding needed!\n"
    "- Enter target area (e.g., `Melbourne, Australia`).\n"
    "- Progress indicators show what's happening at each step.\n"
    "- Output shows keep vs exclude by location with aggregated impressions."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
target_area = st.text_input("Target area (city/region/country)", value="Melbourne, Australia")
throttle = st.slider("Geocode pause (seconds) to ease rate limits", 0.0, 2.0, 0.2, 0.1)

st.caption(f"Geocoder URL: {GEOCODER_URL}")
st.caption(f"Geocoder API key: {'✅ Present' if GEOCODER_API_KEY else '❌ Not set'}")
if openai_client:
    st.caption(f"🤖 OpenAI API: ✅ Enabled (using LLM for intelligent location extraction)")
else:
    st.caption(f"🤖 OpenAI API: ❌ Not configured (using fallback extraction method)")
if GEOCODER_API_KEY:
    # Show key length for debugging (without exposing the actual key)
    st.caption(f"Geocoder key length: {len(GEOCODER_API_KEY)} characters")
    # Show first/last 4 chars for verification (without exposing full key)
    if len(GEOCODER_API_KEY) >= 8:
        st.caption(f"Geocoder key preview: {GEOCODER_API_KEY[:4]}...{GEOCODER_API_KEY[-4:]}")
st.caption(f"App version: {APP_VERSION}")

# Debug section - test API key
if GEOCODER_API_KEY and "geocode.maps.co" in GEOCODER_URL:
    with st.expander("🔧 Debug: Test API Key"):
        st.info("**If you're getting 401 errors:**\n1. Go to https://geocode.maps.co and log into your account\n2. Check that your API key matches exactly what's shown in your account\n3. Try regenerating/copying the key again\n4. Make sure your account is active (check usage limits)")
        if st.button("Test API Key with 'Miami, FL'"):
            test_url = f"{GEOCODER_URL}?q=Miami, FL&api_key={GEOCODER_API_KEY}"
            st.code(test_url, language=None)
            try:
                resp = requests.get(GEOCODER_URL, params={"q": "Miami, FL", "api_key": GEOCODER_API_KEY}, timeout=10)
                st.write(f"**Status Code:** {resp.status_code}")
                st.write(f"**Response:** {resp.text[:500]}")
                if resp.status_code == 200:
                    st.success("✅ API key works!")
                elif resp.status_code == 401:
                    st.error(f"❌ API key is INVALID. Please:\n1. Check https://geocode.maps.co - log in and verify your key\n2. Copy the key directly from the website (don't type it)\n3. Make sure there are no spaces before/after the key in Streamlit Secrets\n4. Try regenerating your API key if it still doesn't work")
                else:
                    st.error(f"❌ API key failed: {resp.text[:200]}")
            except Exception as e:
                st.error(f"Error: {e}")


if uploaded and target_area.strip():
    # Step 1: Read and clean CSV
    status_container = st.container()
    with status_container:
        st.info("📄 **Step 1/4:** Reading and cleaning CSV file...")
    
    try:
        df_raw = _read_and_clean_csv(uploaded)
        with status_container:
            st.success(f"✅ CSV loaded: {len(df_raw)} rows found")
        
        df_cleaned = _standardize_columns(df_raw)
        with status_container:
            st.success(f"✅ Columns standardized: found 'search_term' and 'impressions' columns")
    except Exception as exc:
        st.error(f"❌ Problem reading CSV: {exc}")
        st.stop()

    # Step 2: Aggregate search terms
    with status_container:
        st.info(f"📊 **Step 2/5:** Aggregating duplicate search terms...")
        st.caption(f"Processing {len(df_cleaned)} rows...")
    
    df_aggregated = _aggregate_search_terms(df_cleaned)
    duplicates_found = len(df_cleaned) - len(df_aggregated)
    
    with status_container:
        if duplicates_found > 0:
            st.success(f"✅ Aggregated {len(df_cleaned)} rows into {len(df_aggregated)} unique search terms ({duplicates_found} duplicates merged)")
        else:
            st.success(f"✅ No duplicates found: {len(df_aggregated)} unique search terms")
        st.caption(f"Total impressions: {df_aggregated['impressions'].sum():,.0f}")

    # Step 3: Extract locations from search terms
    with status_container:
        if openai_client:
            st.info(f"🤖 **Step 3/5:** Using AI to extract location names from search terms...")
            st.caption(f"AI is analyzing each search term to intelligently identify location names (no hardcoding!)...")
        else:
            st.info(f"🔍 **Step 3/5:** Extracting location names from search terms...")
            st.caption(f"Using fallback method (OpenAI not configured)...")
    
    df_locations = _extract_locations_from_search_terms(df_aggregated)
    locations_found = len(df_locations)
    terms_without_locations = len(df_aggregated) - len(df_locations)
    
    with status_container:
        if locations_found > 0:
            st.success(f"✅ Extracted {locations_found} unique locations from {len(df_aggregated)} search terms")
            if terms_without_locations > 0:
                st.caption(f"⚠️ {terms_without_locations} search terms had no extractable location (e.g., 'locksmith near me', 'key replacement')")
            # Show some examples
            example_locations = df_locations.head(5)["extracted_location"].tolist()
            st.caption(f"Sample extracted locations: {', '.join(example_locations)}")
        else:
            st.warning(f"⚠️ No locations could be extracted from search terms. Check if search terms contain location names.")
            st.stop()

    # Step 4: Geocode target area
    with status_container:
        st.info(f"🌍 **Step 4/5:** Geocoding target area '{target_area}'...")
    
    target_result, target_geom, target_warnings = geocode_target_area(target_area)
    if target_warnings:
        for w in target_warnings:
            st.warning(w)
    if target_geom is None:
        st.error("Could not geocode the target area. Try a more specific name.")
        st.stop()

    with status_container:
        st.success(f"✅ Target area matched: {target_result.get('display_name')}")

    # Step 5: Geocode extracted locations with progress
    with status_container:
        st.info(f"🗺️ **Step 5/5:** Geocoding {locations_found} extracted locations...")
    
    # Reset cache between runs if needed
    geocode.cache_clear()
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_records = []
    total_locations = len(df_locations)
    geocoded_count = 0
    unmatched_count = 0
    
    for idx, row in df_locations.iterrows():
        extracted_location = row["extracted_location"]
        original_terms = row["original_search_term"]
        impressions = row["impressions"]
        
        # Update progress
        progress = (idx + 1) / total_locations
        progress_bar.progress(progress)
        status_text.text(f"Geocoding {idx + 1}/{total_locations}: '{extracted_location}' | Matched: {geocoded_count} | Unmatched: {unmatched_count}")
        
        result, _ = geocode(extracted_location, polygon=False)
        
        if throttle:
            time.sleep(throttle)
        
        if result is None:
            unmatched_count += 1
            all_records.append(
                {
                    "extracted_location": extracted_location,
                    "original_search_terms": original_terms,
                    "impressions": impressions,
                    "geocoded_name": None,
                    "lat": None,
                    "lon": None,
                    "status": "unmatched",
                }
            )
            continue

        geocoded_count += 1
        inside = is_inside(result, target_geom)
        status = "keep" if inside else "exclude" if inside is False else "unmatched"
        all_records.append(
            {
                "extracted_location": extracted_location,
                "original_search_terms": original_terms,
                "impressions": impressions,
                "geocoded_name": result.get("display_name", ""),
                "lat": result.get("lat"),
                "lon": result.get("lon"),
                "status": status,
            }
        )

    results_df = pd.DataFrame(all_records)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    with status_container:
        st.success(f"✅ Geocoding complete! Processed {total_locations} locations ({geocoded_count} matched, {unmatched_count} unmatched)")

    keep_df = results_df[results_df["status"] == "keep"]
    exclude_df = results_df[results_df["status"] == "exclude"]
    unmatched_df = results_df[results_df["status"] == "unmatched"]

    agg_keep = (
        keep_df.groupby("geocoded_name", dropna=True)["impressions"].sum().reset_index()
        if not keep_df.empty
        else pd.DataFrame(columns=["geocoded_name", "impressions"])
    )
    agg_exclude = (
        exclude_df.groupby("geocoded_name", dropna=True)["impressions"].sum().reset_index()
        if not exclude_df.empty
        else pd.DataFrame(columns=["geocoded_name", "impressions"])
    )

    st.subheader("✅ Suggested keeps (locations inside target area)")
    if not agg_keep.empty:
        st.dataframe(agg_keep.sort_values("impressions", ascending=False))
        download_link("Download keeps (aggregated)", agg_keep, "keeps.csv", "keeps-agg")
        st.caption(f"Total impressions to keep: {agg_keep['impressions'].sum():,.0f}")
    else:
        st.info("No locations found inside the target area.")

    st.subheader("❌ Suggested excludes (locations outside target area)")
    if not agg_exclude.empty:
        st.dataframe(agg_exclude.sort_values("impressions", ascending=False))
        download_link("Download excludes (aggregated)", agg_exclude, "excludes.csv", "excludes-agg")
        st.caption(f"Total impressions to exclude: {agg_exclude['impressions'].sum():,.0f}")
    else:
        st.info("No locations found outside the target area.")

    st.subheader("📋 Detailed results (all locations)")
    st.dataframe(results_df.sort_values("impressions", ascending=False))
    download_link("Download full results", results_df, "results.csv", "full-results")

    if not unmatched_df.empty:
        st.warning(
            f"⚠️ {len(unmatched_df)} extracted locations could not be geocoded. "
            "These may be invalid location names or need manual review. "
            "Try adjusting the throttle if rate-limited."
        )
        with st.expander("View unmatched locations"):
            st.dataframe(unmatched_df[["extracted_location", "original_search_terms", "impressions"]])
else:
    st.info("Upload a CSV and set a target area to begin.")

