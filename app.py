import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
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
            timeout=30.0,  # 30 second timeout to prevent hanging
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
            timeout=30.0,  # 30 second timeout to prevent hanging
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
            timeout=30.0,  # 30 second timeout to prevent hanging
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
            timeout=30.0,  # 30 second timeout to prevent hanging
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


def _read_and_clean_csv(uploaded_file) -> pd.DataFrame:
    """Read CSV or XLSX file, handling Google Ads format with metadata rows."""
    # Determine file type from extension or content type
    file_name = ""
    if hasattr(uploaded_file, 'name'):
        file_name = uploaded_file.name.lower()
    elif hasattr(uploaded_file, 'type'):
        # Check content type as fallback
        content_type = uploaded_file.type.lower()
        if 'spreadsheet' in content_type or 'excel' in content_type:
            file_name = "file.xlsx"
    
    is_xlsx = file_name.endswith('.xlsx') or file_name.endswith('.xls')
    
    # Try multiple encodings for CSV files (common encodings for Excel/Google Ads exports)
    encodings_to_try = ['utf-8', 'utf-16', 'utf-16-le', 'latin-1', 'cp1252']
    
    df = None
    last_error = None
    
    # Strategy: Try reading with skiprows=2 first (most common for Google Ads files)
    # If that fails, try reading normally and detect metadata rows
    if is_xlsx:
        # For Excel files, try with skiprows first
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, engine='openpyxl', skiprows=2)
        except Exception:
            # If skiprows fails, try without skiprows
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            # Check if first row is metadata
            if len(df) > 0 and len(df.columns) > 0:
                first_cell = str(df.iloc[0, 0]).lower() if len(df) > 0 else ""
                first_col = str(df.columns[0]).lower()
                if "report" in first_col or "report" in first_cell or first_cell == "all time":
                    # Re-read with skiprows=2
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file, engine='openpyxl', skiprows=2)
    else:
        # For CSV files, try multiple strategies
        # Strategy 1: Try reading with skiprows=2 (most common case)
        for encoding in encodings_to_try:
            try:
                uploaded_file.seek(0)
                # Try with on_bad_lines for newer pandas, fallback to error_bad_lines for older
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding, skiprows=2, on_bad_lines='skip')
                except TypeError:
                    # Older pandas version, use error_bad_lines
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, skiprows=2, error_bad_lines=False, warn_bad_lines=False)
                # Verify we got a reasonable dataframe (has columns and data)
                if df is not None and len(df.columns) > 0 and len(df) > 0:
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                last_error = e
                continue
        
        # Strategy 2: If skiprows=2 didn't work, try reading normally and detect metadata
        if df is None or len(df.columns) == 0:
            df = None
            for encoding in encodings_to_try:
                try:
                    uploaded_file.seek(0)
                    # Try with on_bad_lines for newer pandas, fallback to error_bad_lines for older
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')
                    except TypeError:
                        # Older pandas version
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding, error_bad_lines=False, warn_bad_lines=False)
                    # Check if first row looks like metadata
                    if df is not None and len(df) > 0 and len(df.columns) > 0:
                        first_cell = str(df.iloc[0, 0]).lower() if len(df) > 0 else ""
                        first_col = str(df.columns[0]).lower()
                        # If first row/column suggests metadata, re-read with skiprows
                        if "report" in first_col or "report" in first_cell or first_cell == "all time" or len(df.columns) == 1:
                            uploaded_file.seek(0)
                            try:
                                df = pd.read_csv(uploaded_file, encoding=encoding, skiprows=2, on_bad_lines='skip')
                            except TypeError:
                                uploaded_file.seek(0)
                                df = pd.read_csv(uploaded_file, encoding=encoding, skiprows=2, error_bad_lines=False, warn_bad_lines=False)
                    if df is not None and len(df.columns) > 0:
                        break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as e:
                    last_error = e
                    continue
        
        if df is None or len(df.columns) == 0:
            raise ValueError(f"Could not read CSV file. Tried encodings: {', '.join(encodings_to_try)}. Last error: {str(last_error) if last_error else 'Unknown error'}")
    
    # Remove rows that are totals or empty
    if len(df) > 0:
        # Check all columns for "Total:" rows (not just first column)
        # A row is a total row if ANY column starts with "Total:"
        is_total_row = pd.Series([False] * len(df))
        for col in df.columns:
            if df[col].dtype == 'object':  # Only check string columns
                col_str = df[col].astype(str).str.strip()
                is_total_row = is_total_row | col_str.str.lower().str.startswith("total")
        
        df = df[~is_total_row]
        
        # Remove rows where the first column is empty or NaN
        if len(df.columns) > 0:
            df = df.dropna(subset=[df.columns[0]])
            # Also remove rows where first column is empty string
            first_col = df.columns[0]
            df = df[df[first_col].astype(str).str.strip() != ""]
    
    # Defensive: Ensure all column names are strings to prevent type errors downstream
    if len(df.columns) > 0:
        df.columns = [str(c).strip() for c in df.columns]
    
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and extract search_term and impressions."""
    # Defensive: Ensure all column names are strings to prevent type errors
    df.columns = [str(c).strip() for c in df.columns]
    
    # Create normalized lookup dictionary (strip and lower for flexible matching)
    lowered = {c: str(c).strip().lower() for c in df.columns}
    col_map: Dict[str, str] = {}

    # Find search term column (with flexible matching - strip whitespace from candidates too)
    for candidate in ["search_term", "search term", "term", "query", "search terms", "searchterm"]:
        candidate_normalized = candidate.strip().lower()
        match = next((c for c in df.columns if lowered[c] == candidate_normalized), None)
        if match:
            col_map[match] = "search_term"
            break
    
    # Find impressions column (with flexible matching - strip whitespace from candidates too)
    for candidate in ["impressions", "impr", "impression", "impr.", "imp"]:
        candidate_normalized = candidate.strip().lower()
        match = next((c for c in df.columns if lowered[c] == candidate_normalized), None)
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


def _extract_locations_from_search_terms(df: pd.DataFrame, progress_callback=None, status_callback=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract location names from search terms and create two dataframes:
    1. Terms with locations (aggregated by location)
    2. Terms without locations (for competitor detection)
    
    Args:
        df: DataFrame with search terms
        progress_callback: Optional function(progress: float) to update progress
        status_callback: Optional function(message: str) to update status
    
    Returns: (df_with_locations, df_without_locations)
    """
    results_with_location = []
    results_without_location = []
    
    total_terms = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        search_term = row["search_term"]
        impressions = row["impressions"]
        
        # Update progress if callback provided
        if progress_callback:
            progress = 0.45 + ((idx / total_terms) * 0.05)  # 45% to 50%
            progress_callback(progress)
        
        if status_callback:
            status_callback(f"Extracting locations: {idx}/{total_terms} terms processed | Found: {len(results_with_location)} locations")
        
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
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        position: relative !important;
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
    
    /* Instructions container - center content vertically */
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }
    
    /* The inner text container */
    div[data-testid="stFileUploaderDropzoneInstructions"] > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 8px !important;
    }
    
    /* Hide the original "Limit 200MB..." text and replace with custom text */
    section[data-testid="stFileUploaderDropzone"] .e16n7gab4 {
        font-size: 0 !important;
        color: transparent !important;
    }
    section[data-testid="stFileUploaderDropzone"] .e16n7gab4::after {
        content: 'Supported Files: CSV, XLSX' !important;
        font-size: 14px !important;
        color: #6b7280 !important;
    }
    
    /* Position the button span below the text with proper spacing */
    section[data-testid="stFileUploaderDropzone"] > .e16n7gab6 {
        margin-top: 20px !important;
        position: relative !important;
        order: 3 !important;
    }
    
    .stTextInput > div > div > input {
        font-size: 18px !important;
        padding: 12px !important;
    }
    
    /* Style the Run button with orange color */
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

# Main input widgets
st.markdown("### Upload File")
uploaded = st.file_uploader(
    "",
    type=["csv", "xlsx"],
    label_visibility="collapsed"
)

st.markdown("### Target Area")
target_area = st.text_input(
    "Target area (city/region/country)",
    value="City, Country",
    help="Enter the target area you want to filter locations for (e.g., 'Adelaide, Australia')"
)

# Advanced settings in sidebar
with st.sidebar.expander("⚙️ Advanced Settings"):
    throttle = st.slider("LLM API pause (seconds) to ease rate limits", 0.0, 2.0, 0.2, 0.1)

# Run button
st.markdown("### Run Analysis")
run_button = st.button("🚀 Run", type="primary", use_container_width=True)

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
    
    # Create real-time log display
    st.subheader("📋 Processing Log")
    log_container = st.empty()
    log_entries = []
    
    def add_log_entry(message: str, level: str = "info"):
        """Add an entry to the real-time log"""
        timestamp = time.strftime("%H:%M:%S")
        icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(level, "ℹ️")
        log_entries.append(f"[{timestamp}] {icon} {message}")
        # Keep only last 50 entries to avoid overwhelming the display
        if len(log_entries) > 50:
            log_entries.pop(0)
        # Update the log display
        log_container.text_area("", value="\n".join(log_entries), height=200, label_visibility="collapsed")
    
    # Step 1: Read and clean CSV (0-20%)
    status_text.text("Reading and cleaning CSV file...")
    add_log_entry("Starting CSV file processing...", "info")
    progress_bar.progress(0.05)
    
    try:
        df_raw = _read_and_clean_csv(uploaded)
        progress_bar.progress(0.10)
        status_text.text(f"CSV loaded: {len(df_raw)} rows found")
        add_log_entry(f"CSV loaded successfully: {len(df_raw)} rows found", "success")
        
        df_cleaned = _standardize_columns(df_raw)
        progress_bar.progress(0.20)
        status_text.text("Columns standardized")
        add_log_entry("Columns standardized and validated", "success")
    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        add_log_entry(f"Error reading CSV: {str(exc)}", "error")
        st.error(f"❌ Problem reading CSV: {exc}")
        st.stop()

    # Step 2: Aggregate search terms (20-40%)
    progress_bar.progress(0.25)
    status_text.text(f"Aggregating duplicate search terms... ({len(df_cleaned)} rows)")
    add_log_entry(f"Aggregating {len(df_cleaned)} search terms...", "info")
    
    df_aggregated = _aggregate_search_terms(df_cleaned)
    duplicates_found = len(df_cleaned) - len(df_aggregated)
    progress_bar.progress(0.35)
    status_text.text(f"Aggregated: {len(df_aggregated)} unique search terms")
    add_log_entry(f"Aggregated to {len(df_aggregated)} unique search terms ({duplicates_found} duplicates removed)", "success")
    
    # Remove summary rows (after aggregation as requested)
    df_aggregated = _remove_summary_rows(df_aggregated)
    progress_bar.progress(0.40)
    status_text.text(f"Removed summary rows: {len(df_aggregated)} search terms remaining")
    add_log_entry(f"Removed summary rows: {len(df_aggregated)} search terms remaining", "success")

    # Step 3: Extract locations from search terms (40-50%)
    progress_bar.progress(0.45)
    if openai_client:
        status_text.text("Using AI to extract location names from search terms...")
        add_log_entry(f"Starting AI-powered location extraction from {len(df_aggregated)} search terms...", "info")
    else:
        status_text.text("Extracting location names from search terms...")
        add_log_entry(f"Starting location extraction from {len(df_aggregated)} search terms (fallback mode)...", "info")
    
    # Pass progress callbacks to the extraction function
    df_locations, df_without_locations = _extract_locations_from_search_terms(
        df_aggregated,
        progress_callback=lambda p: progress_bar.progress(p),
        status_callback=lambda msg: status_text.text(msg)
    )
    locations_found = len(df_locations)
    terms_without_locations_count = len(df_without_locations)
    
    progress_bar.progress(0.50)
    if locations_found > 0:
        status_text.text(f"Extracted {locations_found} unique locations from {len(df_aggregated)} search terms")
        add_log_entry(f"Location extraction complete: {locations_found} unique locations found, {terms_without_locations_count} terms without locations", "success")
    else:
        progress_bar.empty()
        status_text.empty()
        add_log_entry("No locations could be extracted from search terms", "warning")
        st.warning(f"⚠️ No locations could be extracted from search terms. Check if search terms contain location names.")
        st.stop()

    # Step 4: Validate locations using LLM (50-80%)
    progress_bar.progress(0.55)
    status_text.text(f"Validating {locations_found} locations using AI...")
    add_log_entry(f"Starting location validation: {locations_found} locations to validate against target area '{target_area}'", "info")
    
    if not openai_client:
        progress_bar.empty()
        status_text.empty()
        add_log_entry("OpenAI API is required for location validation but is not configured", "error")
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
        location_progress = (idx + 1) / total_locations if total_locations > 0 else 1.0
        overall_progress = min(0.55 + (location_progress * 0.25), 0.80)
        progress_bar.progress(overall_progress)
        status_text.text(f"Validating location {idx + 1}/{total_locations}: {extracted_location} | Inside: {validated_count} | Outside: {unmatched_count}")
        
        # Small delay to avoid rate limiting
        if throttle:
            time.sleep(throttle)
        
        # Ask LLM if the location is inside the target area
        try:
            is_inside, reasoning = _check_location_inside_target_with_llm(final_location, target_area)
            audit_entry["inside_target"] = is_inside
            audit_entry["reasoning"].append(f"Final location: '{final_location}'")
            audit_entry["reasoning"].append(f"LLM check: {reasoning}")
            
            # Determine status
            if is_inside:
                status = "keep"
                audit_entry["status"] = "keep"
                validated_count += 1
                add_log_entry(f"Location '{extracted_location}' is INSIDE target area '{target_area}' (Keep)", "success")
            else:
                status = "exclude"
                audit_entry["status"] = "exclude"
                unmatched_count += 1
                add_log_entry(f"Location '{extracted_location}' is OUTSIDE target area '{target_area}' (Exclude)", "warning")
        except Exception as e:
            # Handle LLM errors gracefully
            add_log_entry(f"Error validating location '{extracted_location}': {str(e)}", "error")
            status = "exclude"
            audit_entry["status"] = "exclude"
            audit_entry["inside_target"] = False
            audit_entry["reasoning"].append(f"Error: {str(e)}")
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
    industry = "N/A"  # Initialize to default value to avoid scope issues
    if not df_without_locations.empty and openai_client:
        progress_bar.progress(0.80)
        status_text.text("Inferring industry from search terms...")
        add_log_entry("Inferring industry from search terms...", "info")
        
        # Infer industry from all search terms (sample from aggregated)
        sample_terms = df_aggregated["search_term"].head(20).tolist()
        try:
            industry = _infer_industry_from_search_terms(tuple(sample_terms))
            add_log_entry(f"Inferred industry: {industry}", "success")
        except Exception as e:
            add_log_entry(f"Error inferring industry: {str(e)}", "error")
            industry = "general business"
        
        progress_bar.progress(0.85)
        status_text.text(f"Inferred industry: {industry}. Detecting competitors from {terms_without_locations_count} terms...")
        add_log_entry(f"Starting competitor detection from {terms_without_locations_count} terms without locations...", "info")
        
        total_terms = len(df_without_locations)
        competitor_count = 0
        
        for idx, row in df_without_locations.iterrows():
            search_term = row["original_search_term"]
            impressions = row["impressions"]
            
            # Update progress (85% to 100%)
            term_progress = (idx + 1) / total_terms if total_terms > 0 else 1.0
            overall_progress = min(0.85 + (term_progress * 0.15), 1.0)
            progress_bar.progress(overall_progress)
            status_text.text(f"Analyzing term {idx + 1}/{total_terms} for competitors... | Found: {competitor_count}")
            
            # Small delay to avoid rate limiting
            if throttle:
                time.sleep(throttle)
            
            # Identify competitor
            try:
                competitor_name = _identify_competitor_with_llm(search_term, industry)
                
                if competitor_name:
                    competitor_records.append({
                        "competitor_name": competitor_name,
                        "original_search_term": search_term,
                        "impressions": impressions,
                    })
                    competitor_count += 1
                    add_log_entry(f"Found competitor: '{competitor_name}' in search term '{search_term}'", "success")
                    # Update status with new count
                    status_text.text(f"Analyzing term {idx + 1}/{total_terms} for competitors... | Found: {competitor_count}")
            except Exception as e:
                add_log_entry(f"Error analyzing term '{search_term}' for competitors: {str(e)}", "error")
    
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
    add_log_entry(f"Processing complete! Locations: {validated_count} inside, {unmatched_count} outside. Competitors: {len(agg_competitors)}", "success")
    
    # Clear progress indicators but keep log visible
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
    
    log_sections.append(f"Environment Variables:")
    log_sections.append(f"  OPENAI_API_KEY: {'✅ Found' if env_openai else '❌ Not set'}")
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
        log_sections.append(f"Inferred Industry: {industry}")
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
        
        # Note: Users can manually select and copy text from the textarea above
        st.caption("💡 Tip: Select all text in the box above (Ctrl+A / Cmd+A) and copy (Ctrl+C / Cmd+C) to copy the audit log.")
