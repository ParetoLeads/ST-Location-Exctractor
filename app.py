import streamlit as st
import pandas as pd
import tempfile
import os
from location_analyzer import LocationAnalyzer
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="KMZ Location Scraper",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #262730;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🗺️ KMZ Location Scraper</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888; margin-top: -1rem; margin-bottom: 0.5rem;">Developed with 💡 by Paretoleads.com</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header" style="color: #ffffff;">Extract locations from KMZ files and estimate populations using OpenStreetMap and GPT</p>', unsafe_allow_html=True)

# Hardcoded configuration
PRIMARY_TYPES = ['city', 'town', 'district', 'county', 'municipality', 'borough', 'suburb']
ADDITIONAL_TYPES = ['neighbourhood', 'village', 'locality']
SPECIAL_TYPES = ['commercial_area']
ALL_TYPES = PRIMARY_TYPES + ADDITIONAL_TYPES + SPECIAL_TYPES

# GPT settings - always enabled
USE_GPT = True
ENABLE_WEB_BROWSING = True
CHUNK_SIZE = 10
MAX_LOCATIONS = 0  # No limit
VERBOSE = False  # Verbose output shows detailed progress messages

# Main content area
uploaded_file = st.file_uploader(
    "Upload KMZ File (Max 1MB)",
    type=['kmz'],
    help="Select a KMZ file containing the boundary polygon. Maximum file size: 1MB",
    accept_multiple_files=False
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'excel_data' not in st.session_state:
    st.session_state.excel_data = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'progress_messages' not in st.session_state:
    st.session_state.progress_messages = []
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []

# Process button
if uploaded_file is not None:
    # Check file size (1MB limit)
    file_size = uploaded_file.size
    max_size = 1 * 1024 * 1024  # 1MB in bytes
    
    if file_size > max_size:
        st.error(f"❌ File size ({file_size / 1024 / 1024:.2f} MB) exceeds the 1MB limit. Please upload a smaller KMZ file.")
    else:
        if st.button("🚀 Start Analysis", type="primary", disabled=st.session_state.processing):
            st.session_state.processing = True
            st.session_state.results = None
            st.session_state.excel_data = None
            
            # Create temporary file for KMZ
            with tempfile.NamedTemporaryFile(delete=False, suffix='.kmz') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_kmz_path = tmp_file.name
            
            try:
                # Get API key from secrets
                api_key = st.secrets.get("OPENAI_API_KEY", "")
                
                if not api_key:
                    st.error("⚠️ OpenAI API key not found in secrets. Please configure it in Streamlit Cloud secrets.")
                    st.session_state.processing = False
                else:
                    # Create progress containers
                    progress_container = st.container()
                    status_container = st.container()
                    
                    with progress_container:
                        # Enhanced progress display
                        stage_text = st.empty()
                        main_progress = st.progress(0)
                        progress_metrics = st.empty()
                        status_text = st.empty()
                    
                    # Progress tracking
                    progress_messages = []
                    status_messages = []
                    progress_state = {
                        "current_stage": "",
                        "stage_progress": 0,
                        "locations_found": 0,
                        "gpt_batches_completed": 0,
                        "gpt_batches_total": 0,
                        "hierarchy_batches_completed": 0,
                        "hierarchy_batches_total": 0
                    }
                    total_stages = 5  # KMZ, OSM, Hierarchy, GPT, Excel
                    
                    def progress_callback(msg):
                        progress_messages.append(msg)
                        
                        # Extract metrics from messages
                        if "boundary points" in msg:
                            # Extract number of boundary points
                            import re
                            match = re.search(r'(\d+)\s+boundary points', msg)
                            if match:
                                progress_state["boundary_points"] = int(match.group(1))
                        
                        if "locations found" in msg.lower() or "Added" in msg and "locations" in msg or "Total unique OSM locations found" in msg:
                            import re
                            # Match patterns like "Total unique OSM locations found: 500" or "Added 5 locations"
                            # First try to match "found: X" pattern
                            match = re.search(r'found:\s*(\d+)', msg)
                            if match:
                                progress_state["locations_found"] = int(match.group(1))
                            else:
                                # Fall back to "X locations" pattern
                                match = re.search(r'(\d+)\s+locations', msg, re.IGNORECASE)
                                if match:
                                    progress_state["locations_found"] = int(match.group(1))
                        
                        if "Retrieving hierarchy batch" in msg or "Processing hierarchy batch" in msg:
                            import re
                            match = re.search(r'batch (\d+)/(\d+)', msg)
                            if match:
                                progress_state["hierarchy_batches_completed"] = int(match.group(1))
                                progress_state["hierarchy_batches_total"] = int(match.group(2))
                        
                        if "Processing GPT batch" in msg or "Calculating population for batch" in msg:
                            import re
                            match = re.search(r'batch (\d+)/(\d+)', msg)
                            if match:
                                progress_state["gpt_batches_completed"] = int(match.group(1))
                                progress_state["gpt_batches_total"] = int(match.group(2))
                        
                        # Update stage based on message content
                        if "Extracting boundary" in msg or "KMZ" in msg or "boundary points" in msg or "Parsing KMZ" in msg:
                            progress_state["current_stage"] = "Parsing boundary coordinates from KMZ file"
                            progress_state["stage_progress"] = 1
                        elif "Finding OSM Locations" in msg or ("OSM" in msg and "Processing" not in msg) or "Discovering locations" in msg:
                            progress_state["current_stage"] = "Discovering locations within boundary"
                            progress_state["stage_progress"] = 2
                        elif "Administrative Hierarchy" in msg or "hierarchy" in msg.lower() or "Retrieving administrative" in msg:
                            progress_state["current_stage"] = "Retrieving administrative boundaries"
                            progress_state["stage_progress"] = 3
                        elif "GPT" in msg or "population" in msg.lower() or "Estimating" in msg or "Calculating population" in msg:
                            progress_state["current_stage"] = "Calculating population estimates"
                            progress_state["stage_progress"] = 4
                        elif "Excel" in msg or "Saved" in msg or "Compiling results" in msg:
                            progress_state["current_stage"] = "Compiling results into Excel export"
                            progress_state["stage_progress"] = 5
                        
                        # Update progress bar (0-100%)
                        progress_value = (progress_state["stage_progress"] / total_stages)
                        main_progress.progress(progress_value)
                        
                        # Enhanced progress display with metrics
                        with stage_text:
                            st.markdown(f"### {progress_state['current_stage']}")
                        
                        # Show detailed metrics
                        with progress_metrics:
                            metrics_parts = []
                            if progress_state.get("locations_found", 0) > 0:
                                metrics_parts.append(f"{progress_state['locations_found']} locations found")
                            if progress_state.get("hierarchy_batches_total", 0) > 0:
                                metrics_parts.append(f"Hierarchy: {progress_state['hierarchy_batches_completed']}/{progress_state['hierarchy_batches_total']} batches")
                            if progress_state.get("gpt_batches_total", 0) > 0:
                                metrics_parts.append(f"GPT: {progress_state['gpt_batches_completed']}/{progress_state['gpt_batches_total']} batches")
                            
                            if metrics_parts:
                                st.caption(" | ".join(metrics_parts))
                        
                        with status_text:
                            st.text(msg)
                    
                    def status_callback(msg):
                        status_messages.append(msg)
                        progress_messages.append(msg)  # Also add to progress messages for log
                    
                    # Initialize analyzer
                    try:
                        analyzer = LocationAnalyzer(
                            kmz_file=tmp_kmz_path,
                            verbose=VERBOSE,
                            openai_api_key=api_key,
                            use_gpt=USE_GPT,
                            chunk_size=CHUNK_SIZE,
                            max_locations=MAX_LOCATIONS,
                            pause_before_gpt=False,  # Not used in Streamlit
                            enable_web_browsing=ENABLE_WEB_BROWSING,
                            primary_place_types=PRIMARY_TYPES,
                            additional_place_types=ADDITIONAL_TYPES,
                            special_place_types=SPECIAL_TYPES,
                            progress_callback=progress_callback,
                            status_callback=status_callback
                        )
                        
                        # Run analysis
                        results = analyzer.run()
                        
                        if results:
                            # Update progress to 100%
                            main_progress.progress(1.0)
                            stage_text.markdown("### Analysis complete!")
                            progress_metrics.empty()
                            status_text.text("Analysis complete!")
                            
                            # Save to Excel
                            excel_data = analyzer.save_to_excel(results)
                            
                            if excel_data:
                                st.session_state.results = results
                                st.session_state.excel_data = excel_data
                                st.session_state.progress_messages = progress_messages
                                st.session_state.status_messages = status_messages
                                st.success(f"✅ Successfully processed {len(results)} locations!")
                            else:
                                st.warning("⚠️ Analysis completed but Excel export failed. Results are still available below.")
                                st.session_state.results = results
                                st.session_state.progress_messages = progress_messages
                                st.session_state.status_messages = status_messages
                        else:
                            st.error("❌ Analysis failed. Check the status messages above.")
                            st.session_state.progress_messages = progress_messages
                            st.session_state.status_messages = status_messages
                            main_progress.progress(0)
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.session_state.progress_messages = progress_messages
                        st.session_state.status_messages = status_messages
                        import traceback
                        error_traceback = traceback.format_exc()
                        with st.expander("Error Details"):
                            st.code(error_traceback)
                        progress_messages.append(f"ERROR: {str(e)}")
                        progress_messages.append(error_traceback)
                        main_progress.progress(0)
                    
                    finally:
                        # Clean up temp file
                        if 'tmp_kmz_path' in locals() and os.path.exists(tmp_kmz_path):
                            os.unlink(tmp_kmz_path)
                        st.session_state.processing = False
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
                st.session_state.processing = False
                if 'tmp_kmz_path' in locals() and os.path.exists(tmp_kmz_path):
                    os.unlink(tmp_kmz_path)

# Display results
if st.session_state.results is not None:
    st.divider()
    st.header("📊 Results")
    
    # Convert results to DataFrame
    df = pd.json_normalize(st.session_state.results, sep='_')
    
    # Select and reorder columns for display
    display_columns = [
        'name',
        'type',
        'latitude',
        'longitude',
        'gpt_population',
        'gpt_confidence',
        'final_population',
        'population_source'
    ]
    
    # Ensure all columns exist
    for col in display_columns:
        if col not in df.columns:
            df[col] = None
    
    # Select available columns
    available_columns = [col for col in display_columns if col in df.columns]
    df_display = df[available_columns].copy()
    
    # Clean column names
    df_display.columns = [col.replace('admin_hierarchy_', '').replace('_', ' ').title() for col in df_display.columns]
    
    # Format numeric columns
    if 'Gpt Population' in df_display.columns:
        df_display['Gpt Population'] = df_display['Gpt Population'].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "-")
    if 'Final Population' in df_display.columns:
        df_display['Final Population'] = df_display['Final Population'].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "-")
    if 'Latitude' in df_display.columns:
        df_display['Latitude'] = df_display['Latitude'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "-")
    if 'Longitude' in df_display.columns:
        df_display['Longitude'] = df_display['Longitude'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "-")
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Locations", len(df_display))
    
    with col2:
        gpt_count = df['gpt_population'].notna().sum() if 'gpt_population' in df.columns else 0
        st.metric("GPT Population Data", gpt_count)
    
    with col3:
        final_count = (df['final_population'] > 0).sum() if 'final_population' in df.columns else 0
        st.metric("Population Assigned", final_count)
    
    with col4:
        clean_count = ((df['final_population'] > 10000).sum() if 'final_population' in df.columns else 0)
        st.metric("Locations > 10K", clean_count)
    
    # Add search/filter
    st.subheader("Location Data")
    search_term = st.text_input("Search locations", placeholder="Type to filter by name...")
    
    if search_term:
        df_display = df_display[df_display['Name'].str.contains(search_term, case=False, na=False)]
    
    # Display table with sorting
    st.dataframe(
        df_display,
        width='stretch',
        height=400,
        hide_index=True
    )
    
    # Download button
    if st.session_state.excel_data is not None:
        st.download_button(
            label="📥 Download Excel File",
            data=st.session_state.excel_data,
            file_name=f"{uploaded_file.name.replace('.kmz', '')}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch'
        )

# Always visible log section at bottom
st.divider()
st.subheader("Processing Log")
all_messages = st.session_state.progress_messages + st.session_state.status_messages
if all_messages:
    log_text = "\n".join(all_messages)
    st.code(log_text, language=None)
    st.caption("Copy this log if you need to report any issues")
else:
    st.info("Log will appear here once processing starts.")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem;'>
        <p>Built with Streamlit | Uses OpenStreetMap and OpenAI GPT</p>
    </div>
""", unsafe_allow_html=True)
