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


@lru_cache(maxsize=1000)
def _check_location_inside_target_with_llm(final_location: str, target_area: str) -> Tuple[bool, str]:
    """
    Use LLM to determine if a location is inside the target area.
    Returns (is_inside, reasoning)
    
    Example:
    - final_location: "adelaide melbourne australia"
    - target_area: "melbourne australia"
    - Should return False because Adelaide is not in Melbourne
    """
    if not openai_client:
        return False, "LLM not available"
    
    if not final_location or not target_area:
        return False, "Missing location or target area"
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a geography expert. Determine if a location is inside/within a target area. Consider:\n- Is the location a suburb, neighborhood, or part of the target area?\n- Is the location a different city/region that is NOT within the target area?\n- Be precise: 'adelaide melbourne australia' means Adelaide is NOT in Melbourne (they are different cities).\n- 'brighton melbourne australia' means Brighton suburb IS in Melbourne.\n\nRespond with ONLY 'YES' if the location is inside the target area, or 'NO' followed by a brief reason if it's not."
                },
                {
                    "role": "user",
                    "content": f"Location: '{final_location}'\nTarget area: '{target_area}'\n\nIs the location inside/within the target area? Answer 'YES' or 'NO' with a brief reason."
                }
            ],
            temperature=0.1,
            max_tokens=100,
        )
        
        answer = response.choices[0].message.content.strip()
        answer_upper = answer.upper()
        
        if answer_upper.startswith("YES"):
            reason = answer[3:].strip() if len(answer) > 3 else "Location is inside target area"
            return True, f"LLM: {reason}"
        elif answer_upper.startswith("NO"):
            reason = answer[2:].strip() if len(answer) > 2 else "Location is not inside target area"
            return False, f"LLM: {reason}"
        else:
            # If response format is unexpected, default to False (conservative)
            return False, f"LLM response unclear, defaulting to outside: {answer}"
        
    except Exception as e:
        # If LLM fails, default to False (conservative approach)
        return False, f"LLM error: {str(e)}"


@lru_cache(maxsize=1)
def _infer_industry_from_search_terms(search_terms_tuple: tuple) -> str:
    """
    Use LLM to infer the industry/business type from a sample of search terms.
    Returns a string describing the industry (e.g., "locksmith services", "plumbing", "restaurants").
    """
    if not openai_client:
        return "general business"
    
    if not search_terms_tuple or len(search_terms_tuple) == 0:
        return "general business"
    
    # Convert tuple back to list and take a sample (max 20 terms for efficiency)
    search_terms = list(search_terms_tuple)[:20]
    terms_text = "\n".join(f"- {term}" for term in search_terms)
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a business analyst. Analyze search terms to infer the industry or business type. Return ONLY a brief industry description (e.g., 'locksmith services', 'plumbing', 'restaurants', 'real estate'). Keep it to 1-3 words."
                },
                {
                    "role": "user",
                    "content": f"Analyze these search terms and infer the industry/business type:\n\n{terms_text}\n\nReturn ONLY the industry name (1-3 words)."
                }
            ],
            temperature=0.1,
            max_tokens=30,
        )
        
        industry = response.choices[0].message.content.strip()
        # Clean up the response
        industry = industry.strip('"').strip("'").strip()
        return industry if industry else "general business"
        
    except Exception as e:
        # If LLM fails, return default
        return "general business"


@lru_cache(maxsize=500)
def _identify_competitor_with_llm(search_term: str, industry: str) -> Optional[str]:
    """
    Use LLM to identify if a search term contains a competitor name.
    Returns the competitor name if found, None otherwise.
    
    Example:
    - search_term: "abc locksmith"
    - industry: "locksmith services"
    - Returns: "ABC Locksmith" or None
    """
    if not openai_client:
        return None
    
    if not search_term or not isinstance(search_term, str) or not search_term.strip():
        return None
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a business intelligence expert. Analyze search terms to identify competitor business names. A competitor name is a specific business/brand name (e.g., 'ABC Locksmith', 'QuickKey Services', 'Joe's Plumbing'). Generic terms like 'locksmith near me', 'plumber', 'best restaurant' are NOT competitor names. Return ONLY the competitor name if found, or 'NONE' if no competitor name is present."
                },
                {
                    "role": "user",
                    "content": f"Industry: {industry}\nSearch term: '{search_term}'\n\nDoes this search term contain a competitor business name? If yes, return ONLY the competitor name. If no, return 'NONE'."
                }
            ],
            temperature=0.1,
            max_tokens=50,
        )
        
        result = response.choices[0].message.content.strip()
        result = result.strip('"').strip("'").strip()
        
        # Handle "NONE" or empty responses
        if not result or result.upper() == "NONE" or len(result) < 2:
            return None
        
        # Capitalize properly (first letter of each word)
        competitor_name = ' '.join(word.capitalize() for word in result.split())
        
        return competitor_name
        
    except Exception as e:
        # If LLM fails, return None
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
    Build a single geocoding query: extracted location + target area.
    
    Strategy: Use only one query with target context to find the location
    within the target area. If it doesn't exist there, geocoding will fail
    or return something outside the target area.
    
    Examples:
    - "Brighton, Melbourne, Australia" -> finds Brighton suburb in Melbourne
    - "Adelaide, Melbourne, Australia" -> won't find good match or finds outside Melbourne
    """
    if not location:
        return []
    
    # Check if location already contains sufficient context
    location_lower = location.lower()
    city = target_components.get('city', '').lower()
    state = target_components.get('state', '').lower()
    country = target_components.get('country', '').lower()
    
    has_city = city and city in location_lower
    has_state = state and state in location_lower
    has_country = country and country in location_lower
    
    # If location already has city AND (state OR country), it's already contextualized
    if has_city and (has_state or has_country):
        return [location]
    
    # Single query: Location + Target area (use simple format: City, Country)
    city_val = target_components.get('city', '')
    country_val = target_components.get('country', '')
    
    # Build simple query: "Location, City, Country"
    if city_val and country_val:
        return [f"{location}, {city_val}, {country_val}"]
    elif country_val:
        return [f"{location}, {country_val}"]
    else:
        # Fallback to original target area input if components not available
        return [f"{location}, {target_display_name}"] if target_display_name else [location]


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
    # Check for summary rows that start with "Total:" (with or without colon)
    first_col_str = df.iloc[:, 0].astype(str).str.strip()
    is_total_row = first_col_str.str.lower().str.startswith("total")
    df = df[~is_total_row]
    
    # Also remove rows where first column is empty
    df = df.dropna(subset=[df.columns[0]])
    
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


def _remove_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove summary rows that start with 'Total:' or similar patterns.
    These are typically added by spreadsheet applications or report generators.
    """
    if df.empty:
        return df
    
    # Check if search_term column exists (after standardization)
    if "search_term" in df.columns:
        # Remove rows where search_term starts with "Total:" (case-insensitive)
        search_terms = df["search_term"].astype(str).str.strip()
        is_summary = search_terms.str.lower().str.startswith("total")
        df = df[~is_summary]
    else:
        # If not standardized yet, check first column
        first_col_str = df.iloc[:, 0].astype(str).str.strip()
        is_summary = first_col_str.str.lower().str.startswith("total")
        df = df[~is_summary]
    
    return df.reset_index(drop=True)


def _extract_locations_from_search_terms(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract location names from search terms and create two dataframes:
    1. Terms with locations (aggregated by location)
    2. Terms without locations (for competitor detection)
    
    Returns: (df_with_locations, df_without_locations)
    """
    results_with_location = []
    results_without_location = []
    
    for _, row in df.iterrows():
        search_term = row["search_term"]
        impressions = row["impressions"]
        
        extracted_location = _extract_location_from_search_term(search_term)
        
        if extracted_location:
            results_with_location.append({
                "original_search_term": search_term,
                "extracted_location": extracted_location,
                "impressions": impressions,
            })
        else:
            results_without_location.append({
                "original_search_term": search_term,
                "impressions": impressions,
            })
    
    # Process terms with locations
    if not results_with_location:
        df_with_locations = pd.DataFrame(columns=["original_search_term", "extracted_location", "impressions"])
    else:
        result_df = pd.DataFrame(results_with_location)
        # Aggregate by extracted location (same location might come from different search terms)
        aggregated = result_df.groupby("extracted_location", as_index=False).agg({
            "impressions": "sum",
            "original_search_term": lambda x: ", ".join(x.unique()[:3]) + ("..." if len(x.unique()) > 3 else "")
        })
        aggregated = aggregated.sort_values("impressions", ascending=False)
        df_with_locations = aggregated.reset_index(drop=True)
    
    # Process terms without locations
    if not results_without_location:
        df_without_locations = pd.DataFrame(columns=["original_search_term", "impressions"])
    else:
        df_without_locations = pd.DataFrame(results_without_location)
        df_without_locations = df_without_locations.sort_values("impressions", ascending=False).reset_index(drop=True)
    
    return df_with_locations, df_without_locations


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
    
    /* Position the Browse files button 150px below the drag and drop text */
    section[data-testid="stFileUploaderDropzone"] > span {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-top: 150px !important;
        padding-top: 0 !important;
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

# Note: Technical Details & Status has been moved to the Audit Log section below
# (Only shown when processing is complete)


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
    progress_bar.progress(0.35)
    status_text.text(f"Aggregated: {len(df_aggregated)} unique search terms")
    
    # Remove summary rows (after aggregation as requested)
    df_aggregated = _remove_summary_rows(df_aggregated)
    progress_bar.progress(0.40)
    status_text.text(f"Removed summary rows: {len(df_aggregated)} search terms remaining")

    # Step 3: Extract locations from search terms (40-50%)
    progress_bar.progress(0.45)
    if openai_client:
        status_text.text("Using AI to extract location names from search terms...")
    else:
        status_text.text("Extracting location names from search terms...")
    
    df_locations, df_without_locations = _extract_locations_from_search_terms(df_aggregated)
    locations_found = len(df_locations)
    terms_without_locations_count = len(df_without_locations)
    
    progress_bar.progress(0.50)
    if locations_found > 0:
        status_text.text(f"Extracted {locations_found} unique locations from {len(df_aggregated)} search terms")
    else:
        progress_bar.empty()
        status_text.empty()
        st.warning(f"⚠️ No locations could be extracted from search terms. Check if search terms contain location names.")
        st.stop()

    # Step 4: Validate locations using LLM (50-80%)
    progress_bar.progress(0.55)
    status_text.text(f"Validating {locations_found} locations using AI...")
    
    if not openai_client:
        progress_bar.empty()
        status_text.empty()
        st.error("❌ OpenAI API is required for location validation. Please configure your API key.")
        st.stop()
    
    all_records = []
    total_locations = len(df_locations)
    validated_count = 0
    unmatched_count = 0
    audit_log = []
    
    for idx, row in df_locations.iterrows():
        extracted_location = row["extracted_location"]
        original_terms = row["original_search_term"]
        impressions = row["impressions"]
        
        # Initialize audit log entry
        audit_entry = {
            "original_search_terms": original_terms,
            "extracted_location": extracted_location,
            "impressions": impressions,
            "final_location": None,
            "inside_target": None,
            "status": None,
            "reasoning": []
        }
        
        # Build final location: "extracted_location target_area" (space-separated)
        final_location = f"{extracted_location} {target_area}".strip()
        audit_entry["final_location"] = final_location
        
        # Update progress (55% to 80%)
        location_progress = (idx + 1) / total_locations
        overall_progress = 0.55 + (location_progress * 0.25)
        progress_bar.progress(overall_progress)
        status_text.text(f"Validating location {idx + 1}/{total_locations}: {extracted_location} | Inside: {validated_count} | Outside: {unmatched_count}")
        
        # Small delay to avoid rate limiting
        if throttle:
            time.sleep(throttle)
        
        # Ask LLM if the location is inside the target area
        is_inside, reasoning = _check_location_inside_target_with_llm(final_location, target_area)
        audit_entry["inside_target"] = is_inside
        audit_entry["reasoning"].append(f"Final location: '{final_location}'")
        audit_entry["reasoning"].append(f"LLM check: {reasoning}")
        
        # Determine status
        if is_inside:
            status = "keep"
            audit_entry["status"] = "keep"
            validated_count += 1
        else:
            status = "exclude"
            audit_entry["status"] = "exclude"
            unmatched_count += 1
        
        audit_log.append(audit_entry)
        all_records.append(
            {
                "extracted_location": extracted_location,
                "original_search_terms": original_terms,
                "impressions": impressions,
                "final_location": final_location,
                "status": status,
            }
        )

    results_df = pd.DataFrame(all_records)
    
    # Step 5: Detect competitors from terms without locations (80-100%)
    competitor_records = []
    if not df_without_locations.empty and openai_client:
        progress_bar.progress(0.80)
        status_text.text("Inferring industry from search terms...")
        
        # Infer industry from all search terms (sample from aggregated)
        sample_terms = df_aggregated["search_term"].head(20).tolist()
        industry = _infer_industry_from_search_terms(tuple(sample_terms))
        
        progress_bar.progress(0.85)
        status_text.text(f"Inferred industry: {industry}. Detecting competitors from {terms_without_locations_count} terms...")
        
        total_terms = len(df_without_locations)
        competitor_count = 0
        
        for idx, row in df_without_locations.iterrows():
            search_term = row["original_search_term"]
            impressions = row["impressions"]
            
            # Update progress (85% to 100%)
            term_progress = (idx + 1) / total_terms
            overall_progress = 0.85 + (term_progress * 0.15)
            progress_bar.progress(overall_progress)
            status_text.text(f"Analyzing term {idx + 1}/{total_terms} for competitors... | Found: {competitor_count}")
            
            # Small delay to avoid rate limiting
            if throttle:
                time.sleep(throttle)
            
            # Identify competitor
            competitor_name = _identify_competitor_with_llm(search_term, industry)
            
            if competitor_name:
                competitor_records.append({
                    "competitor_name": competitor_name,
                    "original_search_term": search_term,
                    "impressions": impressions,
                })
                competitor_count += 1
                # Update status with new count
                status_text.text(f"Analyzing term {idx + 1}/{total_terms} for competitors... | Found: {competitor_count}")
    
    competitor_df = pd.DataFrame(competitor_records) if competitor_records else pd.DataFrame(columns=["competitor_name", "original_search_term", "impressions"])
    
    # Aggregate competitors by name
    if not competitor_df.empty:
        agg_competitors = competitor_df.groupby("competitor_name", as_index=False).agg({
            "impressions": "sum",
            "original_search_term": lambda x: ", ".join(x.unique()[:3]) + ("..." if len(x.unique()) > 3 else "")
        })
        agg_competitors = agg_competitors.sort_values("impressions", ascending=False).reset_index(drop=True)
    else:
        agg_competitors = pd.DataFrame(columns=["competitor_name", "original_search_term", "impressions"])
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text(f"Complete! Processed {total_locations} locations ({validated_count} inside target, {unmatched_count} outside target) | Found {len(agg_competitors)} competitors")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    keep_df = results_df[results_df["status"] == "keep"]
    exclude_df = results_df[results_df["status"] == "exclude"]
    unmatched_df = results_df[results_df["status"] == "unmatched"]

    agg_keep = (
        keep_df.groupby("extracted_location", dropna=True)["impressions"].sum().reset_index()
        if not keep_df.empty
        else pd.DataFrame(columns=["extracted_location", "impressions"])
    )
    agg_exclude = (
        exclude_df.groupby("extracted_location", dropna=True)["impressions"].sum().reset_index()
        if not exclude_df.empty
        else pd.DataFrame(columns=["extracted_location", "impressions"])
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
    
    if not unmatched_df.empty:
        st.warning(
            f"⚠️ {len(unmatched_df)} extracted locations were marked as unmatched. "
            "These may need manual review."
        )
        with st.expander("View unmatched locations"):
            st.dataframe(unmatched_df[["extracted_location", "original_search_terms", "impressions"]])

    st.subheader("📋 Detailed results (all locations)")
    st.dataframe(results_df.sort_values("impressions", ascending=False))
    download_link("Download full results", results_df, "results.csv", "full-results")

    # Competitors section
    st.subheader("🏢 Potential Competitors Detected")
    if not agg_competitors.empty:
        st.dataframe(agg_competitors.sort_values("impressions", ascending=False))
        download_link("Download competitors (aggregated)", agg_competitors, "competitors.csv", "competitors-agg")
        st.caption(f"Total impressions for competitors: {agg_competitors['impressions'].sum():,.0f}")
    else:
        st.info("No competitor names detected in search terms without locations.")

    # Comprehensive Audit Log with Technical Details
    st.subheader("🔧 Technical Details & Status")
    st.info(
        "Complete workflow audit log and technical configuration details. Use this to audit the process, debug issues, and verify system configuration."
    )
    
    # Build comprehensive log combining technical details and audit log
    log_sections = []
    
    # ========== PROCESSING SUMMARY ==========
    log_sections.append("=" * 80)
    log_sections.append("PROCESSING SUMMARY")
    log_sections.append("=" * 80)
    log_sections.append(f"Target Area: {target_area}")
    log_sections.append(f"Total Search Terms Processed: {len(df_aggregated)}")
    log_sections.append(f"Locations Extracted: {total_locations}")
    log_sections.append(f"  ✅ Inside Target Area: {validated_count}")
    log_sections.append(f"  ❌ Outside Target Area: {unmatched_count}")
    if not competitor_df.empty:
        log_sections.append(f"Competitors Detected: {len(agg_competitors)}")
        log_sections.append(f"  Total Competitor Impressions: {agg_competitors['impressions'].sum():,.0f}")
    log_sections.append("")
    
    # ========== APPLICATION INFO ==========
    log_sections.append("=" * 80)
    log_sections.append("APPLICATION INFORMATION")
    log_sections.append("=" * 80)
    log_sections.append(f"App Version: {APP_VERSION}")
    log_sections.append(f"Python Version: {os.sys.version.split()[0]}")
    log_sections.append(f"Streamlit Version: {st.__version__}")
    log_sections.append("")
    
    # ========== CONFIGURATION STATUS ==========
    log_sections.append("=" * 80)
    log_sections.append("CONFIGURATION STATUS")
    log_sections.append("=" * 80)
    
    # OpenAI Status
    if OPENAI_AVAILABLE and openai_client:
        log_sections.append("✅ OpenAI: Ready")
        if OPENAI_API_KEY:
            log_sections.append(f"  API Key: {OPENAI_API_KEY[:7]}...{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 11 else '***'}")
    elif OPENAI_AVAILABLE and OPENAI_API_KEY:
        log_sections.append("⚠️ OpenAI: Package OK but client not initialized")
    elif OPENAI_AVAILABLE:
        log_sections.append("❌ OpenAI: Package installed but API key missing")
    else:
        log_sections.append("❌ OpenAI: Package not installed")
        log_sections.append(f"  Import Error: {OPENAI_IMPORT_ERROR if 'OPENAI_IMPORT_ERROR' in globals() and OPENAI_IMPORT_ERROR else 'Package not installed'}")
    
    # Geocoder Status
    if GEOCODER_API_KEY or "nominatim" in GEOCODER_URL.lower():
        log_sections.append("✅ Geocoder: Configured")
        log_sections.append(f"  Service: {'geocode.maps.co' if 'geocode.maps.co' in GEOCODER_URL else 'Nominatim (OpenStreetMap)'}")
        log_sections.append(f"  URL: {GEOCODER_URL}")
        if GEOCODER_API_KEY:
            log_sections.append(f"  API Key: {'Present' if GEOCODER_API_KEY else 'Not set'}")
    else:
        log_sections.append("⚠️ Geocoder: API key missing (required for geocode.maps.co)")
    
    log_sections.append("")
    
    # ========== CONFIGURATION SOURCES ==========
    log_sections.append("=" * 80)
    log_sections.append("CONFIGURATION SOURCES")
    log_sections.append("=" * 80)
    
    # Streamlit Secrets
    try:
        if hasattr(st, 'secrets'):
            log_sections.append("Streamlit Secrets: ✅ Available")
            try:
                secrets_keys = list(st.secrets.keys())
                log_sections.append(f"  Available Keys: {', '.join(secrets_keys) if secrets_keys else 'None found'}")
            except Exception as e:
                log_sections.append(f"  Error reading secrets: {str(e)}")
        else:
            log_sections.append("Streamlit Secrets: ❌ Not available")
    except Exception as e:
        log_sections.append(f"Streamlit Secrets: ❌ Error - {str(e)}")
    
    # Environment Variables
    env_openai = os.getenv("OPENAI_API_KEY", None)
    env_geocoder_key = os.getenv("GEOCODER_API_KEY", None)
    env_geocoder_url = os.getenv("GEOCODER_URL", None)
    
    log_sections.append(f"Environment Variables:")
    log_sections.append(f"  OPENAI_API_KEY: {'✅ Found' if env_openai else '❌ Not set'}")
    log_sections.append(f"  GEOCODER_API_KEY: {'✅ Found' if env_geocoder_key else '❌ Not set'}")
    log_sections.append(f"  GEOCODER_URL: {'✅ Set' if env_geocoder_url else '❌ Using default'}")
    log_sections.append("")
    
    # ========== LOCATION VALIDATION AUDIT LOG ==========
    log_sections.append("=" * 80)
    log_sections.append("LOCATION VALIDATION AUDIT LOG")
    log_sections.append("=" * 80)
    log_sections.append("")
    
    if audit_log:
        for idx, entry in enumerate(audit_log, 1):
            log_sections.append(f"LOCATION #{idx}")
            log_sections.append("-" * 80)
            log_sections.append(f"Original Search Terms: {entry['original_search_terms']}")
            log_sections.append(f"Extracted Location: {entry['extracted_location']}")
            log_sections.append(f"Impressions: {entry['impressions']}")
            log_sections.append(f"Final Location: {entry['final_location']}")
            log_sections.append(f"Inside Target Area: {entry['inside_target']}")
            log_sections.append("")
            log_sections.append("LLM Reasoning:")
            for reason in entry['reasoning']:
                log_sections.append(f"  • {reason}")
            log_sections.append("")
            status_emoji = {"keep": "✅", "exclude": "❌", "unmatched": "⚠️"}.get(entry['status'], "❓")
            log_sections.append(f"Final Status: {status_emoji} {entry['status'].upper()}")
            log_sections.append("")
            log_sections.append("=" * 80)
            log_sections.append("")
    else:
        log_sections.append("No locations were processed.")
        log_sections.append("")
    
    # ========== COMPETITOR DETECTION LOG ==========
    if not competitor_df.empty:
        log_sections.append("=" * 80)
        log_sections.append("COMPETITOR DETECTION LOG")
        log_sections.append("=" * 80)
        log_sections.append(f"Inferred Industry: {industry if 'industry' in locals() else 'N/A'}")
        log_sections.append(f"Terms Analyzed: {terms_without_locations_count}")
        log_sections.append(f"Competitors Found: {len(agg_competitors)}")
        log_sections.append("")
        
        for idx, row in agg_competitors.iterrows():
            log_sections.append(f"COMPETITOR #{idx + 1}")
            log_sections.append("-" * 80)
            log_sections.append(f"Competitor Name: {row['competitor_name']}")
            log_sections.append(f"Search Terms: {row['original_search_term']}")
            log_sections.append(f"Total Impressions: {row['impressions']:,}")
            log_sections.append("")
            log_sections.append("=" * 80)
            log_sections.append("")
    
    # ========== DEPENDENCIES ==========
    log_sections.append("=" * 80)
    log_sections.append("DEPENDENCIES")
    log_sections.append("=" * 80)
    try:
        requirements_path = "requirements.txt"
        if os.path.exists(requirements_path):
            with open(requirements_path, 'r') as f:
                requirements = f.read()
            log_sections.append("requirements.txt: ✅ Found")
            for line in requirements.strip().split('\n'):
                if line.strip():
                    log_sections.append(f"  - {line.strip()}")
        else:
            log_sections.append("requirements.txt: ❌ Not found")
    except Exception as e:
        log_sections.append(f"Error reading requirements.txt: {str(e)}")
    
    log_sections.append("")
    log_sections.append("=" * 80)
    log_sections.append("END OF AUDIT LOG")
    log_sections.append("=" * 80)
    
    # Combine all sections
    complete_log_text = "\n".join(log_sections)
    
    # Display in expandable section with copy functionality
    with st.expander("📋 View Complete Audit Log", expanded=False):
        st.markdown("**💡 Tip**: Copy the log below if you need help debugging. It contains all processing details and system configuration.")
        
        log_textarea = st.text_area(
            "Complete audit log:",
            value=complete_log_text,
            height=600,
            key="complete_audit_log_display",
            label_visibility="collapsed"
        )
        
        # Copy button using JavaScript
        st.markdown("""
        <script>
            function setupAuditCopyButton() {
                const textarea = document.querySelector('textarea[data-testid="complete_audit_log_display"]');
                if (textarea) {
                    let copyBtn = document.getElementById('copy-audit-log-js-btn');
                    if (!copyBtn) {
                        copyBtn = document.createElement('button');
                        copyBtn.id = 'copy-audit-log-js-btn';
                        copyBtn.innerHTML = '📋 Copy to Clipboard';
                        copyBtn.style.cssText = 'margin-top: 10px; padding: 10px 20px; background-color: #ff6603; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 14px; width: 100%;';
                        
                        copyBtn.onclick = function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            const text = textarea.value;
                            
                            if (navigator.clipboard && navigator.clipboard.writeText) {
                                navigator.clipboard.writeText(text).then(function() {
                                    copyBtn.innerHTML = '✅ Copied to Clipboard!';
                                    copyBtn.style.backgroundColor = '#28a745';
                                    setTimeout(function() {
                                        copyBtn.innerHTML = '📋 Copy to Clipboard';
                                        copyBtn.style.backgroundColor = '#ff6603';
                                    }, 2000);
                                }).catch(function(err) {
                                    copyBtn.innerHTML = '❌ Copy Failed';
                                    copyBtn.style.backgroundColor = '#dc3545';
                                    setTimeout(function() {
                                        copyBtn.innerHTML = '📋 Copy to Clipboard';
                                        copyBtn.style.backgroundColor = '#ff6603';
                                    }, 2000);
                                });
                            } else {
                                const textArea = document.createElement('textarea');
                                textArea.value = text;
                                textArea.style.position = 'fixed';
                                textArea.style.opacity = '0';
                                document.body.appendChild(textArea);
                                textArea.focus();
                                textArea.select();
                                try {
                                    const successful = document.execCommand('copy');
                                    if (successful) {
                                        copyBtn.innerHTML = '✅ Copied to Clipboard!';
                                        copyBtn.style.backgroundColor = '#28a745';
                                        setTimeout(function() {
                                            copyBtn.innerHTML = '📋 Copy to Clipboard';
                                            copyBtn.style.backgroundColor = '#ff6603';
                                        }, 2000);
                                    } else {
                                        copyBtn.innerHTML = '❌ Copy Failed';
                                        copyBtn.style.backgroundColor = '#dc3545';
                                        setTimeout(function() {
                                            copyBtn.innerHTML = '📋 Copy to Clipboard';
                                            copyBtn.style.backgroundColor = '#ff6603';
                                        }, 2000);
                                    }
                                } catch (err) {
                                    copyBtn.innerHTML = '❌ Copy Failed';
                                    copyBtn.style.backgroundColor = '#dc3545';
                                    setTimeout(function() {
                                        copyBtn.innerHTML = '📋 Copy to Clipboard';
                                        copyBtn.style.backgroundColor = '#ff6603';
                                    }, 2000);
                                }
                                document.body.removeChild(textArea);
                            }
                        };
                        
                        const textareaContainer = textarea.closest('.stTextArea');
                        if (textareaContainer) {
                            textareaContainer.appendChild(copyBtn);
                        } else if (textarea.parentElement) {
                            textarea.parentElement.appendChild(copyBtn);
                        }
                    }
                }
            }
            
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', setupAuditCopyButton);
            } else {
                setupAuditCopyButton();
            }
            
            const observer = new MutationObserver(function(mutations) {
                setupAuditCopyButton();
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            
            setInterval(setupAuditCopyButton, 1000);
        </script>
        """, unsafe_allow_html=True)
