"""
Validation utilities for KMZ Location Scraper.
"""
import zipfile
import os
from typing import List, Tuple
from utils.exceptions import ValidationError, KMZParseError
from config import config


def validate_kmz_file(file_path: str) -> None:
    """
    Validate that a file is a valid KMZ file.
    
    Args:
        file_path: Path to the KMZ file
        
    Raises:
        ValidationError: If the file is invalid
        KMZParseError: If the file cannot be parsed
    """
    if not os.path.exists(file_path):
        raise ValidationError(f"File not found: {file_path}")
    
    if not file_path.lower().endswith('.kmz'):
        raise ValidationError(f"File must have .kmz extension: {file_path}")
    
    try:
        with zipfile.ZipFile(file_path, 'r') as kmz:
            # Check if it's a valid zip file
            if kmz.testzip() is not None:
                raise KMZParseError(f"KMZ file is corrupted or invalid: {file_path}")
            
            # Check if it contains at least one KML file
            kml_files = [f for f in kmz.namelist() if f.endswith('.kml')]
            if not kml_files:
                raise KMZParseError(f"No KML file found in KMZ archive: {file_path}")
                
    except zipfile.BadZipFile:
        raise KMZParseError(f"File is not a valid ZIP/KMZ archive: {file_path}")
    except Exception as e:
        raise KMZParseError(f"Error validating KMZ file: {str(e)}")


def validate_file_size(file_size: int, max_size: int = None) -> None:
    """
    Validate that a file size is within the allowed limit.
    
    Args:
        file_size: Size of the file in bytes
        max_size: Maximum allowed size in bytes (defaults to config value)
        
    Raises:
        ValidationError: If the file is too large
    """
    if max_size is None:
        max_size = config.MAX_FILE_SIZE_BYTES
    
    if file_size > max_size:
        max_size_mb = max_size / (1024 * 1024)
        file_size_mb = file_size / (1024 * 1024)
        raise ValidationError(
            f"File size ({file_size_mb:.2f} MB) exceeds the maximum allowed size "
            f"({max_size_mb:.2f} MB). Please upload a smaller file."
        )


def validate_polygon_points(polygon_points: List[Tuple[float, float]], 
                            min_points: int = 3) -> None:
    """
    Validate polygon geometry.
    
    Args:
        polygon_points: List of (lon, lat) tuples
        min_points: Minimum number of points required (default: 3)
        
    Raises:
        ValidationError: If the polygon is invalid
    """
    if not polygon_points:
        raise ValidationError("Polygon must contain at least one point")
    
    if len(polygon_points) < min_points:
        raise ValidationError(
            f"Polygon must contain at least {min_points} points, "
            f"but found {len(polygon_points)}"
        )
    
    # Validate coordinate ranges
    for i, (lon, lat) in enumerate(polygon_points):
        if not (-180 <= lon <= 180):
            raise ValidationError(
                f"Invalid longitude at point {i}: {lon} (must be between -180 and 180)"
            )
        if not (-90 <= lat <= 90):
            raise ValidationError(
                f"Invalid latitude at point {i}: {lat} (must be between -90 and 90)"
            )
    
    # Check for duplicate consecutive points
    for i in range(len(polygon_points) - 1):
        if polygon_points[i] == polygon_points[i + 1]:
            raise ValidationError(
                f"Duplicate consecutive points detected at index {i}"
            )


def validate_api_key(api_key: str, api_type: str = "OpenAI") -> None:
    """
    Validate that an API key is provided and not empty.
    
    Args:
        api_key: The API key to validate
        api_type: Type of API (for error messages)
        
    Raises:
        ValidationError: If the API key is missing or invalid
    """
    if not api_key:
        raise ValidationError(
            f"{api_type} API key is required but not provided. "
            f"Please configure it in Streamlit Cloud secrets or environment variables."
        )
    
    if not isinstance(api_key, str) or len(api_key.strip()) < 10:
        raise ValidationError(
            f"{api_type} API key appears to be invalid (too short or not a string)."
        )


def validate_coordinates(latitude: float, longitude: float) -> None:
    """
    Validate that coordinates are within valid ranges.
    
    Args:
        latitude: Latitude value
        longitude: Longitude value
        
    Raises:
        ValidationError: If coordinates are invalid
    """
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValidationError("Coordinates must be numeric values")
    
    if not (-90 <= latitude <= 90):
        raise ValidationError(f"Invalid latitude: {latitude} (must be between -90 and 90)")
    
    if not (-180 <= longitude <= 180):
        raise ValidationError(f"Invalid longitude: {longitude} (must be between -180 and 180)")
