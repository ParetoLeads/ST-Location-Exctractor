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
    initial_sidebar_state="expanded"
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
st.markdown('<p class="sub-header">Extract locations from KMZ files and estimate populations using OpenStreetMap and GPT</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Place types configuration
    st.subheader("Place Types")
    
    primary_types_input = st.text_input(
        "Primary Types (comma-separated)",
        value="city, town, district, county, municipality, borough, suburb",
        help="Primary location types to search for"
    )
    
    additional_types_input = st.text_input(
        "Additional Types (comma-separated)",
        value="neighbourhood, village, locality",
        help="Additional location types to search for"
    )
    
    special_types_input = st.text_input(
        "Special Types (comma-separated)",
        value="",
        help="Special location types (e.g., commercial_area)"
    )
    
    st.divider()
    
    # GPT Configuration
    st.subheader("GPT Settings")
    
    use_gpt = st.checkbox("Enable GPT Population Estimation", value=True)
    
    chunk_size = st.number_input(
        "Chunk Size",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of locations per GPT batch"
    )
    
    max_locations = st.number_input(
        "Max Locations (0 = no limit)",
        min_value=0,
        value=0,
        help="Maximum number of locations to process"
    )
    
    enable_web_browsing = st.checkbox(
        "Enable Web Browsing for GPT",
        value=False,
        help="Requires GPT-4 with web browsing capabilities"
    )
    
    st.divider()
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        verbose = st.checkbox("Verbose Output", value=False)

# Main content area
uploaded_file = st.file_uploader(
    "Upload KMZ File",
    type=['kmz'],
    help="Select a KMZ file containing the boundary polygon"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'excel_data' not in st.session_state:
    st.session_state.excel_data = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Process button
if uploaded_file is not None:
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        st.session_state.processing = True
        st.session_state.results = None
        st.session_state.excel_data = None
        
        # Create temporary file for KMZ
        with tempfile.NamedTemporaryFile(delete=False, suffix='.kmz') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_kmz_path = tmp_file.name
        
        try:
            # Parse place types
            primary_types = [t.strip() for t in primary_types_input.split(',') if t.strip()]
            additional_types = [t.strip() for t in additional_types_input.split(',') if t.strip()]
            special_types = [t.strip() for t in special_types_input.split(',') if t.strip()]
            
            # Get API key from secrets
            api_key = st.secrets.get("OPENAI_API_KEY", "") if use_gpt else ""
            
            if use_gpt and not api_key:
                st.error("⚠️ OpenAI API key not found in secrets. Please configure it in Streamlit Cloud secrets.")
                st.session_state.processing = False
            else:
                # Create progress containers
                progress_container = st.container()
                status_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Progress callback functions
                progress_messages = []
                status_messages = []
                
                def progress_callback(msg):
                    progress_messages.append(msg)
                    with status_text:
                        st.text(msg)
                
                def status_callback(msg):
                    status_messages.append(msg)
                
                # Initialize analyzer
                analyzer = LocationAnalyzer(
                    kmz_file=tmp_kmz_path,
                    verbose=verbose,
                    openai_api_key=api_key,
                    use_gpt=use_gpt,
                    chunk_size=chunk_size,
                    max_locations=max_locations,
                    pause_before_gpt=False,  # Not used in Streamlit
                    enable_web_browsing=enable_web_browsing,
                    primary_place_types=primary_types,
                    additional_place_types=additional_types,
                    special_place_types=special_types,
                    progress_callback=progress_callback,
                    status_callback=status_callback
                )
                
                # Run analysis
                with st.spinner("Processing... This may take several minutes."):
                    results = analyzer.run()
                    
                    if results:
                        # Update progress
                        progress_bar.progress(1.0)
                        status_text.text("✅ Analysis complete!")
                        
                        # Save to Excel
                        excel_data = analyzer.save_to_excel(results)
                        
                        st.session_state.results = results
                        st.session_state.excel_data = excel_data
                        
                        st.success(f"✅ Successfully processed {len(results)} locations!")
                    else:
                        st.error("❌ Analysis failed. Check the status messages above.")
                
                # Clean up temp file
                os.unlink(tmp_kmz_path)
                st.session_state.processing = False
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state.processing = False
            if os.path.exists(tmp_kmz_path):
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
        'admin_hierarchy_parent_name',
        'admin_hierarchy_level_4_name',
        'osm_population_tag',
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
    df_display = df[available_columns]
    
    # Clean column names
    df_display.columns = [col.replace('admin_hierarchy_', '').replace('_', ' ').title() for col in df_display.columns]
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Locations", len(df_display))
    
    with col2:
        osm_count = df['osm_population_tag'].notna().sum() if 'osm_population_tag' in df.columns else 0
        st.metric("OSM Population Data", osm_count)
    
    with col3:
        gpt_count = df['gpt_population'].notna().sum() if 'gpt_population' in df.columns else 0
        st.metric("GPT Population Data", gpt_count)
    
    with col4:
        final_count = (df['final_population'] > 0).sum() if 'final_population' in df.columns else 0
        st.metric("Final Population Assigned", final_count)
    
    # Display table
    st.dataframe(df_display, use_container_width=True, height=400)
    
    # Download button
    if st.session_state.excel_data is not None:
        st.download_button(
            label="📥 Download Excel File",
            data=st.session_state.excel_data,
            file_name=f"{uploaded_file.name.replace('.kmz', '')}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem;'>
        <p>Built with Streamlit | Uses OpenStreetMap and OpenAI GPT</p>
    </div>
""", unsafe_allow_html=True)
