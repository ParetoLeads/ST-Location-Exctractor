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
    OPENAI_IMPORT_ERROR = None
except ImportError as e:
    OPENAI_AVAILABLE = False
    OpenAI = None
    OPENAI_IMPORT_ERROR = str(e)
except Exception as e:
    OPENAI_AVAILABLE = False
    OpenAI = None
    OPENAI_IMPORT_ERROR = f"{type(e).__name__}: {str(e)}"


# You can override the geocoder endpoint if the default is blocked on your host.
GEOCODER_URL = os.getenv("GEOCODER_URL", "https://nominatim.openstreetmap.org/search")
_raw_key = os.getenv("GEOCODER_API_KEY", "")
GEOCODER_API_KEY = _raw_key.strip() if _raw_key else None  # e.g., for https://geocode.maps.co

USER_AGENT = "location-filter-app/1.0"
APP_VERSION = "v1.16"

# Function to get OpenAI API key from Streamlit secrets or environment
def _get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets or environment variable."""
    _openai_key = None
    
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets'):
            # Try direct access
            try:
                _openai_key = st.secrets["OPENAI_API_KEY"]
            except (KeyError, AttributeError, TypeError):
                # Try get method
                try:
                    _openai_key = st.secrets.get("OPENAI_API_KEY", None)
                except (AttributeError, TypeError):
                    pass
    except Exception:
        pass
    
    # Fallback to environment variable (for local development)
    if not _openai_key:
        _openai_key = os.getenv("OPENAI_API_KEY", None)
    
    # Convert to string and strip
    if _openai_key:
        _openai_key = str(_openai_key).strip()
        return _openai_key if _openai_key else None
    
    return None

# Initialize OpenAI client (will be set after Streamlit initializes)
openai_client = None


st.set_page_config(page_title="Location Search Term Filter", layout="wide")

# Get OpenAI API key now that Streamlit is initialized
OPENAI_API_KEY = _get_openai_api_key()

# Initialize OpenAI client if available and API key is set
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        # Log error but don't crash - will use fallback
        openai_client = None
else:
    openai_client = None

# Display logo and developer info
# Main headline (no version) and styling
st.markdown("""
<style>
    .main-headline {
        font-size: 4rem;
        font-weight: 800;
        color: #000;
        margin-bottom: 1rem;
        text-align: center;
        margin-top: 1rem;
        line-height: 1.2;
    }
    .subheadline {
        font-size: 1.75rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 500;
        line-height: 1.6;
    }
    .dev-info {
        font-size: 0.875rem;
        color: #888;
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #e5e5e5;
    }
    .dev-info a {
        color: #ff6603;
        text-decoration: none;
        font-weight: 500;
    }
    .dev-info a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-headline">Location Search Term Filter</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheadline">Filter and analyze location-based search terms from Google Ads</p>', unsafe_allow_html=True)

# Developer information below headline
st.markdown("""
<div class="dev-info">
    <p>Developed by Nathan Shapiro • App version: """ + APP_VERSION + """ • <a href="https://paretoleads.com" target="_blank">Visit us: Paretoleads.com</a></p>
</div>
""", unsafe_allow_html=True)


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


def _extract_location_components(display_name: str) -> Dict[str, str]:
    """
    Extract key location components (city, state, country) from a geocoded display name.
    Returns a dict with 'city', 'state', 'country' keys.
    """
    if not display_name:
        return {}
    
    # Split by comma and clean up
    parts = [p.strip() for p in display_name.split(',')]
    
    components = {}
    
    # Typically: "City, State/Province, Country" or "City, Admin Level, State, Country"
    # Try to identify components intelligently
    if len(parts) >= 2:
        # Last part is usually country
        components['country'] = parts[-1]
        
        # First part is usually city
        components['city'] = parts[0]
        
        # Middle parts might be state/province - look for common patterns
        # Skip admin levels (like "City Council", "County", etc.)
        admin_keywords = ['council', 'county', 'municipality', 'region', 'district', 'borough', 'city council', 
                         'local government', 'lga', 'shire', 'town', 'village']
        state_candidates = []
        
        for part in parts[1:-1]:  # Skip first (city) and last (country)
            part_lower = part.lower()
            # Skip if it looks like an admin level
            is_admin = any(keyword in part_lower for keyword in admin_keywords)
            # Also skip if it's the same as the city name (e.g., "Adelaide, Adelaide City Council")
            is_duplicate_city = part_lower == parts[0].lower()
            
            if not is_admin and not is_duplicate_city:
                state_candidates.append(part)
        
        # Use the first non-admin part as state, or join them (limit to 2 to avoid too verbose)
        if state_candidates:
            # Prefer shorter state names (usually states/provinces are 1-3 words)
            # Sort by length, preferring shorter ones
            state_candidates.sort(key=lambda x: len(x.split()))
            components['state'] = state_candidates[0]
        elif len(parts) >= 3:
            # Fallback: use second-to-last part as state (might be admin level, but better than nothing)
            potential_state = parts[-2]
            # Only use if it doesn't look like an admin level
            if not any(keyword in potential_state.lower() for keyword in admin_keywords):
                components['state'] = potential_state
    
    return components


def _build_contextualized_query(location: str, target_components: Dict[str, str], target_display_name: str) -> List[str]:
    """
    Build contextualized geocoding queries with fallback options.
    Returns a list of query strings to try, ordered from most specific to least specific.
    """
    if not location:
        return []
    
    location_lower = location.lower()
    queries = []
    
    # Check if location already contains sufficient context (avoid duplication)
    # We need to be smart: just containing the city name isn't enough - we need city + state/country
    city = target_components.get('city', '').lower()
    state = target_components.get('state', '').lower()
    country = target_components.get('country', '').lower()
    
    # Check if location already contains multiple target components (city + state, or city + country)
    # This means it's likely already contextualized
    has_city = city and city in location_lower
    has_state = state and state in location_lower
    has_country = country and country in location_lower
    
    # Only skip adding context if location has city AND (state OR country)
    # This prevents cases like "Adelaide Cbd" from skipping context when it needs it
    if has_city and (has_state or has_country):
        # Location seems already contextualized, but still add a fallback with just the location
        queries.append(location)
        return queries
    
    # Build queries from most specific to least specific
    city_val = target_components.get('city', '')
    state_val = target_components.get('state', '')
    country_val = target_components.get('country', '')
    
    # Query 1: Full context (most specific)
    if target_display_name:
        queries.append(f"{location}, {target_display_name}")
    
    # Query 2: City + State + Country (if we have all components)
    if city_val and state_val and country_val:
        queries.append(f"{location}, {city_val}, {state_val}, {country_val}")
    
    # Query 3: City + Country (simpler)
    if city_val and country_val:
        queries.append(f"{location}, {city_val}, {country_val}")
    
    # Query 4: Just location (fallback)
    queries.append(location)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)
    
    return unique_queries


def _add_target_context(location: str, target_area: str) -> str:
    """
    Add target area context to location name for better geocoding accuracy.
    This is a simple version that returns the most specific query.
    For fallback logic, use _build_contextualized_query instead.
    
    Examples:
    - "Newton" + "Melbourne, Victoria, Australia" -> "Newton, Melbourne, Victoria, Australia"
    - "Newton, Australia" + "Melbourne, Victoria, Australia" -> "Newton, Melbourne, Victoria, Australia" (avoid duplicate)
    """
    queries = _build_contextualized_query(location, {}, target_area)
    return queries[0] if queries else location


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
st.sidebar.markdown("## How to use")
st.sidebar.markdown("""
**1. Upload CSV from Google Ads ST**

**2. Input the target area**

**3. Hit Run**

That's it!
""")

# Comprehensive styling inspired by file transfer sites
st.markdown("""
<style>
    /* Upload area styling - large rectangular area */
    [data-testid="stFileUploader"] {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        background-color: #ffffff;
        min-height: 280px !important;
        padding: 60px 40px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #ff6603;
        background-color: #fff5f0;
    }
    [data-testid="stFileUploader"] > div {
        min-height: 280px !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    [data-testid="stFileUploader"] .uploadedFile {
        min-height: 280px !important;
        padding: 60px 40px !important;
    }
    
    /* Target area input styling */
    .stTextInput > div > div > input {
        font-size: 18px !important;
        padding: 16px 20px !important;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
        background-color: #ffffff;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ff6603;
        outline: none;
        box-shadow: 0 0 0 3px rgba(255, 102, 3, 0.1);
    }
    
    /* CTA Button styling - ParetoLeads orange */
    .stButton > button {
        background-color: #ff6603 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 18px 40px !important;
        border-radius: 8px;
        border: none;
        min-height: 60px !important;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255, 102, 3, 0.3);
    }
    .stButton > button:hover {
        background-color: #e55a00 !important;
        box-shadow: 0 6px 16px rgba(255, 102, 3, 0.4);
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Section spacing */
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 0 40px;
    }
    
    /* Input labels - bigger and brighter */
    .input-label {
        font-size: 22px;
        font-weight: 700;
        color: #000;
        margin-bottom: 6px;
        display: block;
    }
    
    /* Reduce gap between label and widget */
    .stFileUploader {
        margin-top: 6px !important;
    }
    .stTextInput {
        margin-top: 6px !important;
    }
    
    /* Remove default Streamlit spacing */
    .element-container {
        margin-bottom: 2.5rem;
    }
    
    /* Add bigger gap before CTA button */
    .cta-spacing {
        margin-top: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main container with centered content
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Upload CSV section
st.markdown('<label class="input-label">Upload CSV from Google Ads</label>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag and drop your CSV file here or click to browse",
    type=["csv"],
    help="Upload your Google Ads search terms CSV file",
    label_visibility="collapsed"
)

# Target Area section  
st.markdown('<label class="input-label">Target Area</label>', unsafe_allow_html=True)
target_area = st.text_input(
    "Enter target area (e.g., Adelaide, Australia)",
    value="Melbourne, Australia",
    help="Enter the target area you want to filter locations for",
    label_visibility="collapsed"
)

# Advanced settings in sidebar
with st.sidebar.expander("⚙️ Advanced Settings"):
    throttle = st.slider("Geocode pause (seconds) to ease rate limits", 0.0, 2.0, 0.2, 0.1)

# Run button with rocket emoji and bigger gap above
st.markdown('<div class="cta-spacing"></div>', unsafe_allow_html=True)
run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Technical details in collapsible section
with st.expander("🔧 Technical Details & Status"):
    st.caption(f"Geocoder URL: {GEOCODER_URL}")
    st.caption(f"Geocoder API key: {'✅ Present' if GEOCODER_API_KEY else '❌ Not set'}")
    
    # Show OpenAI status with warning if package not available
    if not OPENAI_AVAILABLE:
        st.error("⚠️ **OpenAI package not installed!** The `openai` package is missing. Please check the debug section below for fix instructions.")
    elif openai_client:
        st.caption(f"🤖 OpenAI API: ✅ Enabled (using LLM for intelligent location extraction)")
    else:
        st.warning(f"🤖 OpenAI API: ⚠️ API key found but client not initialized. Check debug section.")
    if GEOCODER_API_KEY:
        # Show key length for debugging (without exposing the actual key)
        st.caption(f"Geocoder key length: {len(GEOCODER_API_KEY)} characters")
        # Show first/last 4 chars for verification (without exposing full key)
        if len(GEOCODER_API_KEY) >= 8:
            st.caption(f"Geocoder key preview: {GEOCODER_API_KEY[:4]}...{GEOCODER_API_KEY[-4:]}")
    st.caption(f"App version: {APP_VERSION}")

# Comprehensive Debug Section (moved inside Technical Details)
with st.expander("🔍 **DEBUG INFO - Copy this if you need help**"):
    debug_info = []
    debug_info.append("## OpenAI Configuration Debug")
    debug_info.append("")
    debug_info.append(f"- **OpenAI Package Available**: {OPENAI_AVAILABLE}")
    if not OPENAI_AVAILABLE:
        debug_info.append(f"- **Import Error**: {OPENAI_IMPORT_ERROR if 'OPENAI_IMPORT_ERROR' in globals() and OPENAI_IMPORT_ERROR else 'Package not installed'}")
        debug_info.append("")
        debug_info.append("### ⚠️ FIX REQUIRED:")
        debug_info.append("The `openai` package is not installed. To fix:")
        debug_info.append("1. Make sure `requirements.txt` includes `openai`")
        debug_info.append("2. Push changes to GitHub")
        debug_info.append("3. In Streamlit Cloud, go to Settings → Dependencies")
        debug_info.append("4. Click 'Reboot app' to reinstall dependencies")
        debug_info.append("")
    debug_info.append(f"- **OpenAI API Key Found**: {'Yes' if OPENAI_API_KEY else 'No'}")
    
    if OPENAI_API_KEY:
        debug_info.append(f"- **API Key Length**: {len(OPENAI_API_KEY)} characters")
        debug_info.append(f"- **API Key Preview**: {OPENAI_API_KEY[:7]}...{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 11 else '***'}")
        debug_info.append(f"- **API Key Starts With**: {OPENAI_API_KEY[:3] if len(OPENAI_API_KEY) >= 3 else 'N/A'}")
    else:
        debug_info.append("- **API Key**: Not found")
    
    debug_info.append(f"- **OpenAI Client Initialized**: {'Yes' if openai_client else 'No'}")
    debug_info.append("")
    
    # Check Streamlit secrets
    debug_info.append("## Streamlit Secrets Check")
    try:
        if hasattr(st, 'secrets'):
            debug_info.append("- **st.secrets available**: Yes")
            try:
                # Try to get all keys
                try:
                    secrets_keys = list(st.secrets.keys())
                    debug_info.append(f"- **Available secret keys**: {', '.join(secrets_keys) if secrets_keys else 'None found'}")
                except Exception as e1:
                    debug_info.append(f"- **Error listing keys**: {str(e1)}")
                    secrets_keys = []
                
                # Try direct access
                try:
                    direct_key = st.secrets["OPENAI_API_KEY"]
                    debug_info.append(f"- **Direct access (st.secrets['OPENAI_API_KEY'])**: ✅ Success (length: {len(str(direct_key))})")
                except KeyError:
                    debug_info.append("- **Direct access (st.secrets['OPENAI_API_KEY'])**: ❌ KeyError - key not found")
                except Exception as e2:
                    debug_info.append(f"- **Direct access error**: {str(e2)} ({type(e2).__name__})")
                
                # Try get method
                try:
                    get_key = st.secrets.get("OPENAI_API_KEY", None)
                    if get_key:
                        debug_info.append(f"- **Get method (st.secrets.get('OPENAI_API_KEY'))**: ✅ Success (length: {len(str(get_key))})")
                    else:
                        debug_info.append("- **Get method (st.secrets.get('OPENAI_API_KEY'))**: ❌ Returned None")
                except Exception as e3:
                    debug_info.append(f"- **Get method error**: {str(e3)} ({type(e3).__name__})")
                
            except Exception as e:
                debug_info.append(f"- **Error reading secrets**: {str(e)} ({type(e).__name__})")
        else:
            debug_info.append("- **st.secrets available**: No")
    except Exception as e:
        debug_info.append(f"- **Error checking secrets**: {str(e)} ({type(e).__name__})")
    debug_info.append("")
    
    # Environment variables check
    debug_info.append("## Environment Variables Check")
    env_openai = os.getenv("OPENAI_API_KEY", None)
    debug_info.append(f"- **OPENAI_API_KEY in env**: {'Yes' if env_openai else 'No'}")
    if env_openai:
        debug_info.append(f"- **Env key length**: {len(env_openai)} characters")
    debug_info.append("")
    
    # Test OpenAI connection
    debug_info.append("## OpenAI Connection Test")
    if openai_client:
        try:
            # Try a simple test call
            test_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            debug_info.append("- **Test API Call**: ✅ Success")
            debug_info.append(f"- **Model Response**: {test_response.choices[0].message.content[:50]}")
        except Exception as e:
            debug_info.append(f"- **Test API Call**: ❌ Failed")
            debug_info.append(f"- **Error**: {str(e)}")
            debug_info.append(f"- **Error Type**: {type(e).__name__}")
    else:
        debug_info.append("- **Test API Call**: Skipped (client not initialized)")
    debug_info.append("")
    
    # System info
    debug_info.append("## System Information")
    debug_info.append(f"- **Python Version**: {os.sys.version.split()[0]}")
    debug_info.append(f"- **Streamlit Version**: {st.__version__}")
    debug_info.append("")
    
    # Check requirements.txt
    debug_info.append("## Requirements Check")
    try:
        requirements_path = "requirements.txt"
        if os.path.exists(requirements_path):
            with open(requirements_path, 'r') as f:
                requirements = f.read()
            debug_info.append(f"- **requirements.txt exists**: Yes")
            debug_info.append(f"- **Contains 'openai'**: {'Yes' if 'openai' in requirements.lower() else 'No'}")
            debug_info.append(f"- **Requirements file contents**:")
            for line in requirements.strip().split('\n'):
                if line.strip():
                    debug_info.append(f"  - {line.strip()}")
        else:
            debug_info.append(f"- **requirements.txt exists**: No")
    except Exception as e:
        debug_info.append(f"- **Error reading requirements.txt**: {str(e)}")
    debug_info.append("")
    
    # Output as markdown
    st.markdown("\n".join(debug_info))
    
    # Also provide as copyable text
    st.text_area(
        "📋 Copy this debug info:",
        value="\n".join(debug_info),
        height=400,
        key="debug_output"
    )

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


# Only process if Run button is clicked and inputs are provided
if run_button:
    if not uploaded:
        st.error("❌ Please upload a CSV file first.")
        st.stop()
    if not target_area.strip():
        st.error("❌ Please enter a target area.")
        st.stop()

if run_button and uploaded and target_area.strip():
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
    
    # Extract target area display name for context
    target_display_name = target_result.get('display_name', target_area)
    
    # Extract location components for building better queries
    # Try to get components from address details first (more reliable)
    target_components = {}
    if target_result.get('address'):
        address = target_result['address']
        # Different geocoders use different address field names
        target_components['city'] = (
            address.get('city') or 
            address.get('town') or 
            address.get('municipality') or
            address.get('city_district') or
            None
        )
        target_components['state'] = (
            address.get('state') or 
            address.get('region') or 
            address.get('province') or
            address.get('state_district') or
            None
        )
        target_components['country'] = (
            address.get('country') or
            None
        )
    
    # Fallback to parsing display_name if address details aren't available
    if not target_components.get('city') or not target_components.get('state'):
        parsed_components = _extract_location_components(target_display_name)
        # Merge, preferring address details but filling gaps with parsed components
        for key in ['city', 'state', 'country']:
            if not target_components.get(key) and parsed_components.get(key):
                target_components[key] = parsed_components[key]
    
    # Step 5: Geocode extracted locations with progress
    with status_container:
        st.info(f"🗺️ **Step 5/5:** Geocoding {locations_found} extracted locations...")
        # Show debug info about target area
        if target_components:
            debug_parts = []
            if target_components.get('city'):
                debug_parts.append(f"City: {target_components['city']}")
            if target_components.get('state'):
                debug_parts.append(f"State: {target_components['state']}")
            if target_components.get('country'):
                debug_parts.append(f"Country: {target_components['country']}")
            if debug_parts:
                st.caption(f"Target area components: {', '.join(debug_parts)}")
    
    # Reset cache between runs if needed
    geocode.cache_clear()
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_records = []
    total_locations = len(df_locations)
    geocoded_count = 0
    unmatched_count = 0
    audit_log = []  # Comprehensive audit log for all locations
    
    for idx, row in df_locations.iterrows():
        extracted_location = row["extracted_location"]
        original_terms = row["original_search_term"]
        impressions = row["impressions"]
        
        # Initialize audit log entry for this location
        audit_entry = {
            "original_search_terms": original_terms,
            "extracted_location": extracted_location,
            "impressions": impressions,
            "queries_tried": [],
            "successful_query": None,
            "geocoded_result": None,
            "inside_target": None,
            "status": None,
            "reasoning": []
        }
        
        # Build contextualized queries with fallback options
        queries = _build_contextualized_query(extracted_location, target_components, target_display_name)
        primary_query = queries[0] if queries else extracted_location
        audit_entry["queries_tried"] = queries.copy()
        
        # Update progress
        progress = (idx + 1) / total_locations
        progress_bar.progress(progress)
        if len(queries) > 1:
            status_text.text(f"Geocoding {idx + 1}/{total_locations}: '{extracted_location}' → '{primary_query}' (will try {len(queries)} variants) | Matched: {geocoded_count} | Unmatched: {unmatched_count}")
        else:
            status_text.text(f"Geocoding {idx + 1}/{total_locations}: '{extracted_location}' → '{primary_query}' | Matched: {geocoded_count} | Unmatched: {unmatched_count}")
        
        # Try queries in order until one succeeds
        result = None
        error_messages = []
        successful_query = None
        
        for query_idx, query in enumerate(queries):
            result, error = geocode(query, polygon=False)
            
            if throttle:
                time.sleep(throttle)
            
            if result is not None:
                successful_query = query
                audit_entry["successful_query"] = query
                audit_entry["reasoning"].append(f"Query {query_idx + 1} succeeded: '{query}'")
                break
            else:
                error_msg = error or "No results returned"
                error_messages.append(f"Query {query_idx + 1} ('{query}'): {error_msg}")
                audit_entry["reasoning"].append(f"Query {query_idx + 1} failed: '{query}' → {error_msg}")
        
        if result is None:
            unmatched_count += 1
            audit_entry["status"] = "unmatched"
            audit_entry["reasoning"].append("All queries failed - location could not be geocoded")
            audit_log.append(audit_entry)
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
        geocoded_name = result.get("display_name", "")
        geocoded_lat = result.get("lat")
        geocoded_lon = result.get("lon")
        
        # Check if inside target area
        inside = is_inside(result, target_geom)
        audit_entry["geocoded_result"] = {
            "name": geocoded_name,
            "lat": geocoded_lat,
            "lon": geocoded_lon
        }
        audit_entry["inside_target"] = inside
        
        # Determine status
        if inside is True:
            status = "keep"
            audit_entry["status"] = "keep"
            audit_entry["reasoning"].append(f"Geocoded location '{geocoded_name}' is INSIDE target area '{target_display_name}'")
        elif inside is False:
            status = "exclude"
            audit_entry["status"] = "exclude"
            audit_entry["reasoning"].append(f"Geocoded location '{geocoded_name}' is OUTSIDE target area '{target_display_name}'")
        else:
            status = "unmatched"
            audit_entry["status"] = "unmatched"
            audit_entry["reasoning"].append(f"Could not determine if '{geocoded_name}' is inside/outside target area (geometry check returned None)")
        
        audit_log.append(audit_entry)
        all_records.append(
            {
                "extracted_location": extracted_location,
                "original_search_terms": original_terms,
                "impressions": impressions,
                "geocoded_name": geocoded_name,
                "lat": geocoded_lat,
                "lon": geocoded_lon,
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
    
    # Comprehensive Audit Log
    st.subheader("📋 Audit Log - Complete Workflow")
    st.info(
        "This log shows the complete workflow for each location: original search terms → extracted location → "
        "geocoding queries tried → results → final status. Use this to audit and debug the process."
    )
    
    # Create formatted audit log text
    audit_log_text = []
    audit_log_text.append("=" * 80)
    audit_log_text.append("AUDIT LOG - Location Geocoding Workflow")
    audit_log_text.append("=" * 80)
    audit_log_text.append(f"Target Area (Input): {target_area}")
    audit_log_text.append(f"Target Area (Geocoded): {target_display_name}")
    if target_components:
        audit_log_text.append("Target Components Extracted:")
        if target_components.get('city'):
            audit_log_text.append(f"  City: {target_components['city']}")
        if target_components.get('state'):
            audit_log_text.append(f"  State/Province: {target_components['state']}")
        if target_components.get('country'):
            audit_log_text.append(f"  Country: {target_components['country']}")
    audit_log_text.append("")
    audit_log_text.append(f"Total Locations Processed: {total_locations}")
    audit_log_text.append(f"Successfully Geocoded: {geocoded_count}")
    audit_log_text.append(f"Unmatched: {unmatched_count}")
    audit_log_text.append("")
    audit_log_text.append("=" * 80)
    audit_log_text.append("")
    
    for idx, entry in enumerate(audit_log, 1):
        audit_log_text.append(f"LOCATION #{idx}")
        audit_log_text.append("-" * 80)
        audit_log_text.append(f"Original Search Terms: {entry['original_search_terms']}")
        audit_log_text.append(f"Extracted Location: {entry['extracted_location']}")
        audit_log_text.append(f"Impressions: {entry['impressions']}")
        audit_log_text.append("")
        
        audit_log_text.append("Geocoding Queries Tried:")
        for i, query in enumerate(entry['queries_tried'], 1):
            marker = "✓" if query == entry['successful_query'] else "✗"
            audit_log_text.append(f"  {marker} Query {i}: {query}")
        audit_log_text.append("")
        
        if entry['successful_query']:
            audit_log_text.append(f"Successful Query: {entry['successful_query']}")
            if entry['geocoded_result']:
                audit_log_text.append(f"Geocoded Result:")
                audit_log_text.append(f"  Name: {entry['geocoded_result']['name']}")
                audit_log_text.append(f"  Coordinates: ({entry['geocoded_result']['lat']}, {entry['geocoded_result']['lon']})")
                audit_log_text.append(f"  Inside Target Area: {entry['inside_target']}")
        else:
            audit_log_text.append("Successful Query: NONE (all queries failed)")
        audit_log_text.append("")
        
        audit_log_text.append("Reasoning:")
        for reason in entry['reasoning']:
            audit_log_text.append(f"  • {reason}")
        audit_log_text.append("")
        
        status_emoji = {"keep": "✅", "exclude": "❌", "unmatched": "⚠️"}.get(entry['status'], "❓")
        audit_log_text.append(f"Final Status: {status_emoji} {entry['status'].upper()}")
        audit_log_text.append("")
        audit_log_text.append("=" * 80)
        audit_log_text.append("")
    
    audit_log_string = "\n".join(audit_log_text)
    
    # Display audit log in expandable section
    with st.expander("📋 View Complete Audit Log", expanded=False):
        st.text_area(
            "Copy this audit log for debugging:",
            value=audit_log_string,
            height=600,
            key="audit_log_display"
        )
    
    # Also show a summary table
    audit_summary = []
    for entry in audit_log:
        audit_summary.append({
            "Original Search Terms": entry['original_search_terms'],
            "Extracted Location": entry['extracted_location'],
            "Queries Tried": len(entry['queries_tried']),
            "Successful Query": entry['successful_query'] or "None",
            "Geocoded Name": entry['geocoded_result']['name'] if entry['geocoded_result'] else "N/A",
            "Inside Target": str(entry['inside_target']) if entry['inside_target'] is not None else "N/A",
            "Status": entry['status'].upper()
        })
    
    audit_summary_df = pd.DataFrame(audit_summary)
    with st.expander("📊 Audit Summary Table"):
        st.dataframe(audit_summary_df)
elif not run_button:
    # Show initial state message only if Run hasn't been clicked
    st.info("👆 Upload a CSV file, enter a target area, and click 'Run' to begin.")

