# KMZ Location Scraper

A Streamlit web application that extracts locations from KMZ boundary files and estimates populations using OpenStreetMap (OSM) and GPT-4.

## Features

- **KMZ Boundary Extraction**: Parses KMZ/KML files to extract polygon boundaries
- **OSM Location Discovery**: Queries OpenStreetMap to find locations within or near the boundary
- **Administrative Hierarchy**: Retrieves administrative boundaries for location disambiguation
- **GPT Population Estimation**: Uses GPT-4 Turbo to estimate population for discovered locations
- **Interactive Visualization**: Displays locations on an interactive map with boundary overlay
- **Excel Export**: Exports results to Excel with full data and filtered clean data sheets

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT population estimation)

### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Streamlit secrets (for Streamlit Cloud deployment):
   - Create a `.streamlit/secrets.toml` file (or configure in Streamlit Cloud)
   - Add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

### Local Development

Run the application locally:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Configuration

Configuration is managed through the `config.py` module and can be customized via environment variables.

### Environment Variables

- `OSM_OVERPASS_URL`: OpenStreetMap Overpass API URL (default: `http://overpass-api.de/api/interpreter`)
- `OSM_API_TIMEOUT`: API timeout in seconds (default: `30`)
- `MAX_RETRY_ATTEMPTS`: Maximum retry attempts for API calls (default: `12`)
- `DEFAULT_CHUNK_SIZE`: Number of locations per GPT batch (default: `10`)
- `DEFAULT_MAX_LOCATIONS`: Maximum locations to process, 0 for no limit (default: `0`)
- `POLYGON_BUFFER_KM`: Buffer distance in km for point-in-polygon checks (default: `2.0`)
- `USE_GPT`: Enable GPT population estimation (default: `True`)
- `GPT_MODEL`: GPT model to use (default: `gpt-4-turbo`)
- `MAX_FILE_SIZE_MB`: Maximum KMZ file size in MB (default: `1`)
- `ENABLE_CACHE`: Enable caching for OSM queries (default: `False`)
- `CACHE_TTL_SECONDS`: Cache time-to-live in seconds (default: `3600`)
- `VERBOSE`: Enable verbose logging (default: `False`)

### Place Types

The application searches for different types of locations:

**Primary Types**: city, town, district, county, municipality, borough, suburb

**Additional Types**: neighbourhood, village, locality

**Special Types**: commercial_area

These can be customized in `config.py` or by passing custom lists to `LocationAnalyzer`.

## Usage

1. **Upload KMZ File**: Click "Upload KMZ File" and select a KMZ file containing your boundary polygon (max 1MB)

2. **Start Analysis**: Click "🚀 Start Analysis" to begin processing

3. **Monitor Progress**: Watch the progress bar and status messages as the application:
   - Extracts boundary coordinates from the KMZ file
   - Discovers locations within the boundary using OpenStreetMap
   - Retrieves administrative hierarchy information
   - Estimates populations using GPT-4
   - Compiles results into Excel format

4. **View Results**: Once complete, you can:
   - View location statistics
   - Explore locations on an interactive map
   - Search and filter locations
   - Download results as an Excel file

## Output Format

The Excel export contains two sheets:

1. **Full Data**: All discovered locations with:
   - Name, type, coordinates
   - GPT population estimate and confidence
   - Administrative hierarchy information

2. **Clean Data**: Filtered to locations with population > 10,000, sorted by population (descending)

## Architecture

### Project Structure

```
location_scraper/
├── app.py                 # Streamlit UI
├── location_analyzer.py   # Core analysis logic
├── config.py              # Configuration management
├── utils/
│   ├── __init__.py
│   ├── exceptions.py      # Custom exception classes
│   ├── retry_handler.py   # Retry logic for API calls
│   ├── progress_tracker.py # Progress tracking utilities
│   ├── validators.py       # Validation functions
│   └── cache.py           # Caching utilities
├── requirements.txt
└── README.md
```

### Key Components

- **LocationAnalyzer**: Main class that orchestrates the analysis process
- **ProgressTracker**: Manages progress state and UI updates
- **RetryHandler**: Handles API retries with exponential backoff
- **Validators**: Validates KMZ files, API keys, and geometry
- **Cache**: Optional caching for OSM queries and hierarchy lookups

## Error Handling

The application includes comprehensive error handling:

- **KMZParseError**: Invalid or corrupted KMZ files
- **OSMQueryError**: OpenStreetMap API errors
- **GPTAPIError**: OpenAI API errors
- **ValidationError**: Configuration or input validation errors

All errors are logged and displayed to the user with actionable error messages.

## Performance Considerations

- **Caching**: Enable caching via `ENABLE_CACHE=true` to reduce API calls for repeated queries
- **Batch Processing**: Locations are processed in batches to optimize API usage
- **Rate Limiting**: Built-in delays and retry logic handle API rate limits gracefully
- **Parallelization**: Can be enabled via `ENABLE_PARALLEL_OSM_QUERIES=true` (experimental)

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Ensure your API key is configured in Streamlit secrets
   - Check that the key is valid and has sufficient credits

2. **"File size exceeds limit"**
   - Reduce the KMZ file size or increase `MAX_FILE_SIZE_MB` in config

3. **"Rate limited" errors**
   - The application automatically retries with exponential backoff
   - For frequent use, consider enabling caching

4. **"No locations found"**
   - Verify the KMZ file contains a valid polygon
   - Check that the boundary area contains populated places in OpenStreetMap

## Development

### Adding New Place Types

Edit `config.py` to add new place types:

```python
PRIMARY_PLACE_TYPES = ['city', 'town', 'your_new_type']
```

### Customizing Retry Logic

Modify retry behavior in `utils/retry_handler.py` or adjust `MAX_RETRY_ATTEMPTS` and `RETRY_DELAY_BASE` in config.

### Extending Validation

Add new validation functions in `utils/validators.py` and use them in `app.py`.

## License

This project is developed by Paretoleads.com

## Support

For issues or questions, please check the processing log in the application or review the error details in the expandable error sections.
