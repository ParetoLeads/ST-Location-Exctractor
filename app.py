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
APP_VERSION = "v1.13"

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

# Display logo
logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=300)

st.title("Location Search Term Filter")

# Developer information (under headline)
st.markdown(f"**App version:** {APP_VERSION}")
st.markdown("**Developer:** Nathan Shapiro")
st.markdown("**Visit us:** [Paretoleads.com](https://paretoleads.com)")

# Add separator line
st.markdown("---")


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


def _get_result_administrative_level(result: Dict) -> str:
    """
    Determine the administrative level of a geocoded result.
    Returns: 'country', 'state', 'city', 'suburb', 'street', or 'unknown'
    """
    if not result:
        return 'unknown'
    
    # Check type/class from geocoder
    result_type = result.get('type', '').lower()
    result_class = result.get('class', '').lower()
    
    # Map geocoder types to administrative levels
    if result_type in ['country'] or result_class in ['country']:
        return 'country'
    if result_type in ['state', 'province', 'region'] or result_class in ['state', 'province', 'region', 'administrative']:
        return 'state'
    if result_type in ['city', 'town'] or result_class in ['city', 'town']:
        return 'city'
    
    # Check address components to infer level
    address = result.get('address', {})
    if address:
        has_country = any(key in address for key in ['country'])
        has_state = any(key in address for key in ['state', 'region', 'province'])
        has_city = any(key in address for key in ['city', 'town', 'municipality'])
        has_suburb = any(key in address for key in ['suburb', 'neighbourhood', 'neighborhood', 'village'])
        has_street = any(key in address for key in ['road', 'street', 'house_number', 'house', 'building'])
        
        if has_street:
            return 'street'
        if has_suburb:
            return 'suburb'
        if has_city and not has_street:
            return 'city'
        if has_state and not has_city and not has_street:
            return 'state'
        if has_country and not has_state and not has_city and not has_street:
            return 'country'
    
    return 'unknown'


def _extract_result_location_components(result: Dict) -> Dict[str, str]:
    """
    Extract city, state, and country from a geocoded result.
    Returns a dict with 'city', 'state', 'country' keys.
    """
    components = {}
    address = result.get('address', {})
    
    if address:
        # Extract city
        components['city'] = (
            address.get('city') or 
            address.get('town') or 
            address.get('municipality') or
            address.get('city_district') or
            None
        )
        
        # Extract state
        components['state'] = (
            address.get('state') or 
            address.get('region') or 
            address.get('province') or
            address.get('state_district') or
            None
        )
        
        # Extract country
        components['country'] = address.get('country') or None
    
    # Fallback to parsing display_name if address details aren't available
    if not components.get('city') or not components.get('state'):
        display_name = result.get('display_name', '')
        if display_name:
            parsed = _extract_location_components(display_name)
            for key in ['city', 'state', 'country']:
                if not components.get(key) and parsed.get(key):
                    components[key] = parsed[key]
    
    return components


def _locations_match(result_components: Dict[str, str], target_components: Dict[str, str]) -> bool:
    """
    Check if a result location matches the target location based on city/state/country.
    Returns True if they appear to be the same location.
    """
    result_city = (result_components.get('city') or '').lower().strip()
    result_state = (result_components.get('state') or '').lower().strip()
    result_country = (result_components.get('country') or '').lower().strip()
    
    target_city = (target_components.get('city') or '').lower().strip()
    target_state = (target_components.get('state') or '').lower().strip()
    target_country = (target_components.get('country') or '').lower().strip()
    
    # If we have city info for both, they must match
    if result_city and target_city:
        if result_city != target_city:
            return False
    
    # If we have state info for both, they must match
    if result_state and target_state:
        if result_state != target_state:
            return False
    
    # Country should always match (assuming same country searches)
    if result_country and target_country:
        if result_country != target_country:
            return False
    
    return True


def _build_contextualized_query(location: str, target_components: Dict[str, str], target_display_name: str) -> List[str]:
    """
    Build geocoding queries. ALWAYS appends target area context to the location.
    This ensures locations are searched within the target area first.
    Example: "Newton" + target "Adelaide, Australia" -> "Newton, Adelaide, Australia"
    """
    if not location:
        return []
    
    location_lower = location.lower()
    queries = []
    
    # Check if location already contains sufficient context
    city = target_components.get('city', '').lower()
    state = target_components.get('state', '').lower()
    country = target_components.get('country', '').lower()
    
    has_city = city and city in location_lower
    has_state = state and state in location_lower
    has_country = country and country in location_lower
    
    # If location already has city AND (state OR country), it's already contextualized
    if has_city and (has_state or has_country):
        queries.append(location)
        return queries
    
    city_val = target_components.get('city', '')
    state_val = target_components.get('state', '')
    country_val = target_components.get('country', '')
    
    # Strategy: ALWAYS append target area context first
    # This ensures we search for the location within the target area
    
    # Query 1: Location + Full target display name (most specific)
    if target_display_name:
        queries.append(f"{location}, {target_display_name}")
    
    # Query 2: Location + City + State + Country (if we have all components)
    if city_val and state_val and country_val:
        queries.append(f"{location}, {city_val}, {state_val}, {country_val}")
    
    # Query 3: Location + City + Country
    if city_val and country_val:
        queries.append(f"{location}, {city_val}, {country_val}")
    
    # Query 4: Location + State + Country (if no city but have state)
    if state_val and country_val and not city_val:
        queries.append(f"{location}, {state_val}, {country_val}")
    
    # Query 5: Location + Country (broader search)
    if country_val:
        queries.append(f"{location}, {country_val}")
    
    # Query 6: Just location (fallback if all contextualized queries fail)
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

# Add custom CSS to make upload drop area bigger and responsive
st.markdown("""
<style>
    /* Target the actual drop zone - it's a SECTION element! */
    section[data-testid="stFileUploaderDropzone"] {
        height: 450px !important;
        min-height: 450px !important;
        padding: 40px 20px !important;
        transition: all 0.3s ease !important;
        box-sizing: border-box !important;
    }
    
    /* Hover state */
    section[data-testid="stFileUploaderDropzone"]:hover {
        background-color: rgba(31, 119, 180, 0.05) !important;
        transform: scale(1.01) !important;
    }
    
    /* Drag over state - when file is being dragged */
    section[data-testid="stFileUploaderDropzone"].drag-over {
        background-color: rgba(31, 119, 180, 0.15) !important;
        transform: scale(1.02) !important;
        box-shadow: 0 0 20px rgba(31, 119, 180, 0.3) !important;
        border-color: #1f77b4 !important;
    }
    
    /* Make sure the inner content area is properly sized and positioned */
    section[data-testid="stFileUploaderDropzone"] {
        display: flex !important;
        flex-direction: column !important;
        position: relative !important;
    }
    
    section[data-testid="stFileUploaderDropzone"] > div {
        height: 100% !important;
        min-height: 370px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        flex: 1 !important;
    }
    
    /* Position the Browse files button at the bottom center */
    section[data-testid="stFileUploaderDropzone"] > span {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-top: auto !important;
        padding-top: 20px !important;
        padding-bottom: 20px !important;
        width: 100% !important;
    }
    
    section[data-testid="stFileUploaderDropzone"] > span > button {
        margin: 0 auto !important;
    }
    
    .stTextInput > div > div > input {
        font-size: 18px !important;
        padding: 12px !important;
    }
    
    /* Style the Run button with orange color */
    button[data-testid="stBaseButton-primary"],
    button[data-testid="baseButton-primary"],
    button[kind="primary"] {
        background-color: #ff6603 !important;
        border-color: #ff6603 !important;
        color: white !important;
    }
    
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[kind="primary"]:hover {
        background-color: #e55a00 !important;
        border-color: #e55a00 !important;
    }
    
    button[data-testid="stBaseButton-primary"]:active,
    button[data-testid="baseButton-primary"]:active,
    button[kind="primary"]:active {
        background-color: #cc4f00 !important;
        border-color: #cc4f00 !important;
    }
</style>
<script>
    // Enhanced drag-over handler targeting the correct element
    function setupDragOver() {
        const dropZone = document.querySelector('section[data-testid="stFileUploaderDropzone"]');
        
        if (dropZone && !dropZone.hasAttribute('data-drag-setup')) {
            dropZone.setAttribute('data-drag-setup', 'true');
            
            // Prevent default drag behaviors
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                }, false);
            });
            
            // Add drag-over class on dragenter/dragover
            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, function(e) {
                    this.classList.add('drag-over');
                }, false);
            });
            
            // Remove drag-over class on dragleave/drop
            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, function(e) {
                    this.classList.remove('drag-over');
                }, false);
            });
        }
    }
    
    // Run immediately and on various events
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupDragOver);
    } else {
        setupDragOver();
    }
    
    // Watch for Streamlit re-renders
    const observer = new MutationObserver(function(mutations) {
        setupDragOver();
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    // Also use interval as backup
    setInterval(setupDragOver, 500);
</script>
""", unsafe_allow_html=True)

# Main input widgets
st.markdown("### Upload CSV")
uploaded = st.file_uploader(
    "Upload CSV from Google Ads",
    type=["csv"],
    help="Upload your Google Ads search terms CSV file"
)

# Inject CSS and JavaScript right after file uploader to ensure it applies
st.markdown("""
<style>
    /* Force the drop zone section to be 450px tall */
    section[data-testid="stFileUploaderDropzone"] {
        height: 450px !important;
        min-height: 450px !important;
    }
    section[data-testid="stFileUploaderDropzone"].drag-over {
        border-color: #1f77b4 !important;
        background-color: rgba(31, 119, 180, 0.15) !important;
        box-shadow: 0 0 20px rgba(31, 119, 180, 0.3) !important;
    }
</style>
<script>
    // Force resize using JavaScript as backup - target the correct element
    function forceResizeUploader() {
        const dropZone = document.querySelector('section[data-testid="stFileUploaderDropzone"]');
        if (dropZone) {
            // Set height directly via JavaScript
            dropZone.style.height = '450px';
            dropZone.style.minHeight = '450px';
        }
    }
    
    // Run immediately and repeatedly
    forceResizeUploader();
    setInterval(forceResizeUploader, 300);
    
    // Also run after a delay to catch late renders
    setTimeout(forceResizeUploader, 1000);
    setTimeout(forceResizeUploader, 2000);
</script>
""", unsafe_allow_html=True)

st.markdown("### Target Area")
target_area = st.text_input(
    "Target area (city/region/country)",
    value="City, Country",
    help="Enter the target area you want to filter locations for (e.g., 'Adelaide, Australia')"
)

# Advanced settings in sidebar
with st.sidebar.expander("⚙️ Advanced Settings"):
    throttle = st.slider("Geocode pause (seconds) to ease rate limits", 0.0, 2.0, 0.2, 0.1)

# Run button
st.markdown("### Run Analysis")
run_button = st.button("🚀 Run", type="primary", use_container_width=True)

# Inject CSS to style the Run button with orange color
st.markdown("""
<style>
    /* Target primary buttons (Run button) */
    button[data-testid="stBaseButton-primary"],
    button[data-testid="baseButton-primary"],
    button[kind="primary"],
    .stButton > button[kind="primary"] {
        background-color: #ff6603 !important;
        border-color: #ff6603 !important;
        color: white !important;
    }
    
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background-color: #e55a00 !important;
        border-color: #e55a00 !important;
    }
    
    button[data-testid="stBaseButton-primary"]:active,
    button[data-testid="baseButton-primary"]:active,
    button[kind="primary"]:active,
    .stButton > button[kind="primary"]:active {
        background-color: #cc4f00 !important;
        border-color: #cc4f00 !important;
    }
</style>
""", unsafe_allow_html=True)

# Unified Technical Details & Status Log
with st.expander("🔧 Technical Details & Status", expanded=False):
    # Build comprehensive log
    log_sections = []
    
    # ========== APPLICATION INFO ==========
    log_sections.append("## 📱 Application Information")
    log_sections.append(f"- **App Version**: {APP_VERSION}")
    log_sections.append(f"- **Python Version**: {os.sys.version.split()[0]}")
    log_sections.append(f"- **Streamlit Version**: {st.__version__}")
    log_sections.append("")
    
    # ========== GEOCODER CONFIGURATION ==========
    log_sections.append("## 🌍 Geocoder Configuration")
    log_sections.append(f"- **Geocoder URL**: {GEOCODER_URL}")
    log_sections.append(f"- **Geocoder Service**: {'geocode.maps.co' if 'geocode.maps.co' in GEOCODER_URL else 'Nominatim (OpenStreetMap)'}")
    log_sections.append(f"- **API Key Status**: {'✅ Present' if GEOCODER_API_KEY else '❌ Not set'}")
    
    if GEOCODER_API_KEY:
        log_sections.append(f"- **API Key Length**: {len(GEOCODER_API_KEY)} characters")
        if len(GEOCODER_API_KEY) >= 8:
            log_sections.append(f"- **API Key Preview**: {GEOCODER_API_KEY[:4]}...{GEOCODER_API_KEY[-4:]}")
        
        # Geocoder API Key Test (if applicable)
        if "geocode.maps.co" in GEOCODER_URL:
            log_sections.append("")
            log_sections.append("### Geocoder API Key Test")
            log_sections.append("**Troubleshooting 401 errors:**")
            log_sections.append("1. Go to https://geocode.maps.co and log into your account")
            log_sections.append("2. Check that your API key matches exactly what's shown in your account")
            log_sections.append("3. Try regenerating/copying the key again")
            log_sections.append("4. Make sure your account is active (check usage limits)")
            log_sections.append("")
            
            # Test button for geocoder
            if st.button("🧪 Test Geocoder API Key", key="test_geocoder_key"):
                test_url = f"{GEOCODER_URL}?q=Miami, FL&api_key={GEOCODER_API_KEY}"
                st.code(test_url, language=None)
                try:
                    resp = requests.get(GEOCODER_URL, params={"q": "Miami, FL", "api_key": GEOCODER_API_KEY}, timeout=10)
                    st.write(f"**Status Code:** {resp.status_code}")
                    st.write(f"**Response:** {resp.text[:500]}")
                    if resp.status_code == 200:
                        st.success("✅ Geocoder API key works!")
                    elif resp.status_code == 401:
                        st.error("❌ API key is INVALID. Please check the troubleshooting steps above.")
                    else:
                        st.error(f"❌ API key test failed: {resp.text[:200]}")
                except Exception as e:
                    st.error(f"Error testing API key: {e}")
    else:
        if "geocode.maps.co" in GEOCODER_URL:
            log_sections.append("- **⚠️ Warning**: API key required for geocode.maps.co but not found")
    
    log_sections.append("")
    
    # ========== OPENAI CONFIGURATION ==========
    log_sections.append("## 🤖 OpenAI Configuration")
    log_sections.append(f"- **Package Installed**: {'✅ Yes' if OPENAI_AVAILABLE else '❌ No'}")
    
    if not OPENAI_AVAILABLE:
        log_sections.append(f"- **Import Error**: {OPENAI_IMPORT_ERROR if 'OPENAI_IMPORT_ERROR' in globals() and OPENAI_IMPORT_ERROR else 'Package not installed'}")
        log_sections.append("")
        log_sections.append("### ⚠️ FIX REQUIRED - OpenAI Package Missing")
        log_sections.append("The `openai` package is not installed. To fix:")
        log_sections.append("1. Make sure `requirements.txt` includes `openai`")
        log_sections.append("2. Push changes to GitHub")
        log_sections.append("3. In Streamlit Cloud, go to Settings → Dependencies")
        log_sections.append("4. Click 'Reboot app' to reinstall dependencies")
        log_sections.append("")
    else:
        log_sections.append(f"- **API Key Found**: {'✅ Yes' if OPENAI_API_KEY else '❌ No'}")
        
        if OPENAI_API_KEY:
            log_sections.append(f"- **API Key Length**: {len(OPENAI_API_KEY)} characters")
            log_sections.append(f"- **API Key Preview**: {OPENAI_API_KEY[:7]}...{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 11 else '***'}")
            log_sections.append(f"- **API Key Starts With**: {OPENAI_API_KEY[:3] if len(OPENAI_API_KEY) >= 3 else 'N/A'}")
        
        log_sections.append(f"- **Client Initialized**: {'✅ Yes' if openai_client else '❌ No'}")
        
        # OpenAI Connection Test
        log_sections.append("")
        log_sections.append("### OpenAI Connection Test")
        if openai_client:
            try:
                test_response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                log_sections.append("- **Test API Call**: ✅ Success")
                log_sections.append(f"- **Model Response**: {test_response.choices[0].message.content[:50]}")
            except Exception as e:
                log_sections.append("- **Test API Call**: ❌ Failed")
                log_sections.append(f"- **Error**: {str(e)}")
                log_sections.append(f"- **Error Type**: {type(e).__name__}")
        else:
            log_sections.append("- **Test API Call**: ⏭️ Skipped (client not initialized)")
    
    log_sections.append("")
    
    # ========== CONFIGURATION SOURCES ==========
    log_sections.append("## 🔐 Configuration Sources")
    
    # Streamlit Secrets Check
    log_sections.append("### Streamlit Secrets")
    try:
        if hasattr(st, 'secrets'):
            log_sections.append("- **Available**: ✅ Yes")
            try:
                secrets_keys = list(st.secrets.keys())
                log_sections.append(f"- **Available Keys**: {', '.join(secrets_keys) if secrets_keys else 'None found'}")
                
                # Test OPENAI_API_KEY access
                try:
                    direct_key = st.secrets["OPENAI_API_KEY"]
                    log_sections.append(f"- **OPENAI_API_KEY (direct)**: ✅ Found (length: {len(str(direct_key))})")
                except KeyError:
                    log_sections.append("- **OPENAI_API_KEY (direct)**: ❌ Key not found")
                except Exception as e:
                    log_sections.append(f"- **OPENAI_API_KEY (direct)**: ❌ Error - {str(e)} ({type(e).__name__})")
                
                try:
                    get_key = st.secrets.get("OPENAI_API_KEY", None)
                    if get_key:
                        log_sections.append(f"- **OPENAI_API_KEY (get method)**: ✅ Found (length: {len(str(get_key))})")
                    else:
                        log_sections.append("- **OPENAI_API_KEY (get method)**: ❌ Returned None")
                except Exception as e:
                    log_sections.append(f"- **OPENAI_API_KEY (get method)**: ❌ Error - {str(e)} ({type(e).__name__})")
            except Exception as e:
                log_sections.append(f"- **Error reading secrets**: {str(e)} ({type(e).__name__})")
        else:
            log_sections.append("- **Available**: ❌ No")
    except Exception as e:
        log_sections.append(f"- **Error**: {str(e)} ({type(e).__name__})")
    
    log_sections.append("")
    
    # Environment Variables Check
    log_sections.append("### Environment Variables")
    env_openai = os.getenv("OPENAI_API_KEY", None)
    env_geocoder_key = os.getenv("GEOCODER_API_KEY", None)
    env_geocoder_url = os.getenv("GEOCODER_URL", None)
    
    log_sections.append(f"- **OPENAI_API_KEY**: {'✅ Found' if env_openai else '❌ Not set'}")
    if env_openai:
        log_sections.append(f"  - Length: {len(env_openai)} characters")
    
    log_sections.append(f"- **GEOCODER_API_KEY**: {'✅ Found' if env_geocoder_key else '❌ Not set'}")
    if env_geocoder_key:
        log_sections.append(f"  - Length: {len(env_geocoder_key)} characters")
    
    log_sections.append(f"- **GEOCODER_URL**: {'✅ Set' if env_geocoder_url else '❌ Using default'}")
    if env_geocoder_url:
        log_sections.append(f"  - Value: {env_geocoder_url}")
    
    log_sections.append("")
    
    # ========== DEPENDENCIES ==========
    log_sections.append("## 📦 Dependencies")
    try:
        requirements_path = "requirements.txt"
        if os.path.exists(requirements_path):
            with open(requirements_path, 'r') as f:
                requirements = f.read()
            log_sections.append("- **requirements.txt**: ✅ Found")
            log_sections.append(f"- **Contains 'openai'**: {'✅ Yes' if 'openai' in requirements.lower() else '❌ No'}")
            log_sections.append("- **Installed Packages**:")
            for line in requirements.strip().split('\n'):
                if line.strip():
                    log_sections.append(f"  - {line.strip()}")
        else:
            log_sections.append("- **requirements.txt**: ❌ Not found")
    except Exception as e:
        log_sections.append(f"- **Error reading requirements.txt**: {str(e)}")
    
    log_sections.append("")
    
    # ========== STATUS SUMMARY ==========
    log_sections.append("## 📊 Status Summary")
    
    # Determine overall status
    status_items = []
    if OPENAI_AVAILABLE and openai_client:
        status_items.append("✅ OpenAI: Ready")
    elif OPENAI_AVAILABLE and OPENAI_API_KEY:
        status_items.append("⚠️ OpenAI: Package OK but client not initialized")
    elif OPENAI_AVAILABLE:
        status_items.append("⚠️ OpenAI: Package installed but API key missing")
    else:
        status_items.append("❌ OpenAI: Package not installed")
    
    if GEOCODER_API_KEY or "nominatim" in GEOCODER_URL.lower():
        status_items.append("✅ Geocoder: Configured")
    else:
        status_items.append("⚠️ Geocoder: API key missing (required for geocode.maps.co)")
    
    for item in status_items:
        log_sections.append(f"- {item}")
    
    log_sections.append("")
    log_sections.append("---")
    log_sections.append("")
    log_sections.append("**💡 Tip**: Copy the log below if you need help debugging. It contains all the information needed to diagnose issues.")
    
    # Display the log
    log_text = "\n".join(log_sections)
    st.markdown(log_text)
    
    # Copy button and text area
    st.markdown("**📋 Complete Debug Log:**")
    
    # Copyable text area with hidden label
    log_textarea = st.text_area(
        "Log content:",
        value=log_text,
        height=500,
        key="technical_log_output",
        label_visibility="collapsed"
    )
    
    # Copy button using JavaScript
    st.markdown("""
    <script>
        function setupCopyButton() {
            // Find the textarea by its key
            const textarea = document.querySelector('textarea[data-testid="technical_log_output"]');
            if (textarea) {
                // Check if button already exists
                let copyBtn = document.getElementById('copy-log-js-btn');
                if (!copyBtn) {
                    // Create copy button
                    copyBtn = document.createElement('button');
                    copyBtn.id = 'copy-log-js-btn';
                    copyBtn.innerHTML = '📋 Copy to Clipboard';
                    copyBtn.style.cssText = 'margin-top: 10px; padding: 10px 20px; background-color: #ff6603; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 14px; width: 100%;';
                    
                    copyBtn.onclick = function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        // Get text from textarea
                        const text = textarea.value;
                        
                        // Copy to clipboard
                        if (navigator.clipboard && navigator.clipboard.writeText) {
                            navigator.clipboard.writeText(text).then(function() {
                                // Success feedback
                                copyBtn.innerHTML = '✅ Copied to Clipboard!';
                                copyBtn.style.backgroundColor = '#28a745';
                                setTimeout(function() {
                                    copyBtn.innerHTML = '📋 Copy to Clipboard';
                                    copyBtn.style.backgroundColor = '#ff6603';
                                }, 2000);
                            }).catch(function(err) {
                                // Fallback for older browsers
                                fallbackCopyTextToClipboard(text, copyBtn);
                            });
                        } else {
                            // Fallback for browsers without clipboard API
                            fallbackCopyTextToClipboard(text, copyBtn);
                        }
                    };
                    
                    // Fallback copy function for older browsers
                    function fallbackCopyTextToClipboard(text, btn) {
                        const textArea = document.createElement("textarea");
                        textArea.value = text;
                        textArea.style.top = "0";
                        textArea.style.left = "0";
                        textArea.style.position = "fixed";
                        document.body.appendChild(textArea);
                        textArea.focus();
                        textArea.select();
                        try {
                            const successful = document.execCommand('copy');
                            if (successful) {
                                btn.innerHTML = '✅ Copied to Clipboard!';
                                btn.style.backgroundColor = '#28a745';
                                setTimeout(function() {
                                    btn.innerHTML = '📋 Copy to Clipboard';
                                    btn.style.backgroundColor = '#ff6603';
                                }, 2000);
                            } else {
                                btn.innerHTML = '❌ Copy Failed';
                                btn.style.backgroundColor = '#dc3545';
                                setTimeout(function() {
                                    btn.innerHTML = '📋 Copy to Clipboard';
                                    btn.style.backgroundColor = '#ff6603';
                                }, 2000);
                            }
                        } catch (err) {
                            btn.innerHTML = '❌ Copy Failed';
                            btn.style.backgroundColor = '#dc3545';
                            setTimeout(function() {
                                btn.innerHTML = '📋 Copy to Clipboard';
                                btn.style.backgroundColor = '#ff6603';
                            }, 2000);
                        }
                        document.body.removeChild(textArea);
                    }
                    
                    // Insert button after the textarea container
                    const textareaContainer = textarea.closest('.stTextArea');
                    if (textareaContainer) {
                        textareaContainer.appendChild(copyBtn);
                    } else if (textarea.parentElement) {
                        textarea.parentElement.appendChild(copyBtn);
                    }
                }
            }
        }
        
        // Run on load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupCopyButton);
        } else {
            setupCopyButton();
        }
        
        // Watch for Streamlit re-renders
        const observer = new MutationObserver(function(mutations) {
            setupCopyButton();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // Also check periodically as backup
        setInterval(setupCopyButton, 1000);
    </script>
    """, unsafe_allow_html=True)


# Only process if Run button is clicked and inputs are provided
if run_button:
    if not uploaded:
        st.error("❌ Please upload a CSV file first.")
        st.stop()
    if not target_area.strip():
        st.error("❌ Please enter a target area.")
        st.stop()

if run_button and uploaded and target_area.strip():
    # Create single progress bar and status text for entire process
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Read and clean CSV (0-20%)
    status_text.text("Reading and cleaning CSV file...")
    progress_bar.progress(0.05)
    
    try:
        df_raw = _read_and_clean_csv(uploaded)
        progress_bar.progress(0.10)
        status_text.text(f"CSV loaded: {len(df_raw)} rows found")
        
        df_cleaned = _standardize_columns(df_raw)
        progress_bar.progress(0.20)
        status_text.text("Columns standardized")
    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Problem reading CSV: {exc}")
        st.stop()

    # Step 2: Aggregate search terms (20-40%)
    progress_bar.progress(0.25)
    status_text.text(f"Aggregating duplicate search terms... ({len(df_cleaned)} rows)")
    
    df_aggregated = _aggregate_search_terms(df_cleaned)
    duplicates_found = len(df_cleaned) - len(df_aggregated)
    progress_bar.progress(0.40)
    status_text.text(f"Aggregated: {len(df_aggregated)} unique search terms")

    # Step 3: Extract locations from search terms (40-60%)
    progress_bar.progress(0.45)
    if openai_client:
        status_text.text("Using AI to extract location names from search terms...")
    else:
        status_text.text("Extracting location names from search terms...")
    
    df_locations = _extract_locations_from_search_terms(df_aggregated)
    locations_found = len(df_locations)
    terms_without_locations = len(df_aggregated) - len(df_locations)
    
    progress_bar.progress(0.60)
    if locations_found > 0:
        status_text.text(f"Extracted {locations_found} unique locations from {len(df_aggregated)} search terms")
    else:
        progress_bar.empty()
        status_text.empty()
        st.warning(f"⚠️ No locations could be extracted from search terms. Check if search terms contain location names.")
        st.stop()

    # Step 4: Geocode target area (60-70%)
    progress_bar.progress(0.65)
    status_text.text(f"Geocoding target area '{target_area}'...")
    
    target_result, target_geom, target_warnings = geocode_target_area(target_area)
    if target_warnings:
        for w in target_warnings:
            st.warning(w)
    if target_geom is None:
        progress_bar.empty()
        status_text.empty()
        st.error("Could not geocode the target area. Try a more specific name.")
        st.stop()

    progress_bar.progress(0.70)
    status_text.text(f"Target area matched: {target_result.get('display_name')}")
    
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
    
    # Step 5: Geocode extracted locations (70-100%)
    progress_bar.progress(0.75)
    status_text.text(f"Geocoding {locations_found} extracted locations...")
    
    # Reset cache between runs if needed
    geocode.cache_clear()
    
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
        
        # Update progress (70% to 100% for geocoding step)
        location_progress = (idx + 1) / total_locations
        overall_progress = 0.70 + (location_progress * 0.30)  # 70% to 100%
        progress_bar.progress(overall_progress)
        status_text.text(f"Geocoding location {idx + 1}/{total_locations}: {extracted_location} | Matched: {geocoded_count} | Unmatched: {unmatched_count}")
        
        # Try queries in order until we get a valid result
        # Since queries are ordered with target context first, use the first successful result
        result = None
        error_messages = []
        successful_query = None
        result_admin_level = None
        result_components = None
        
        for query_idx, query in enumerate(queries):
            candidate_result, error = geocode(query, polygon=False)
            
            if throttle:
                time.sleep(throttle)
            
            if candidate_result is not None:
                # Get administrative level and location components for validation
                admin_level = _get_result_administrative_level(candidate_result)
                components = _extract_result_location_components(candidate_result)
                
                # Use the first successful result (queries are ordered with target context first)
                result = candidate_result
                successful_query = query
                result_admin_level = admin_level
                result_components = components
                audit_entry["successful_query"] = query
                audit_entry["reasoning"].append(f"Query {query_idx + 1} succeeded: '{query}' (found {admin_level} level location)")
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
        
        # Check if inside target area using geometry
        inside = is_inside(result, target_geom)
        audit_entry["geocoded_result"] = {
            "name": geocoded_name,
            "lat": geocoded_lat,
            "lon": geocoded_lon
        }
        audit_entry["inside_target"] = inside
        
        # Determine status based on geometry check
        # Since we always query with target context first, the result should be in/near the target area
        # However, for high-level administrative areas (country/state/city), we also validate location components
        # to catch edge cases where a major city name matches a street/suburb in the target area
        
        if result_admin_level in ['country', 'state', 'city']:
            # For high-level administrative areas, validate location components match target
            # This catches cases where "Adelaide" (city) might match "Adelaide Street" in target area
            if not _locations_match(result_components, target_components):
                # Different city/state - exclude even if geometry says inside
                status = "exclude"
                audit_entry["status"] = "exclude"
                result_loc = result_components.get('city') or result_components.get('state') or 'unknown'
                target_loc = target_components.get('city') or target_components.get('state') or 'unknown'
                audit_entry["reasoning"].append(f"{result_admin_level.capitalize()} level location doesn't match target area - excluding (result: {result_loc}, target: {target_loc})")
            elif inside is True:
                status = "keep"
                audit_entry["status"] = "keep"
                audit_entry["reasoning"].append(f"{result_admin_level.capitalize()} level location matches target area and is INSIDE")
            elif inside is False:
                status = "exclude"
                audit_entry["status"] = "exclude"
                audit_entry["reasoning"].append(f"{result_admin_level.capitalize()} level location is OUTSIDE target area")
            else:
                status = "unmatched"
                audit_entry["status"] = "unmatched"
                audit_entry["reasoning"].append(f"Could not determine if {result_admin_level} level location is inside/outside target area")
        else:
            # For lower-level locations (suburbs, streets), use geometry check
            # Since we queried with target context, these should be in the target area
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
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text(f"Complete! Processed {total_locations} locations ({geocoded_count} matched, {unmatched_count} unmatched)")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

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

