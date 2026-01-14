import pandas as pd
import requests
import json
import os
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Callable
import time
from openai import OpenAI
from math import ceil
from bs4 import BeautifulSoup
import re
import traceback
import math
from io import BytesIO

# Constants
DEFAULT_CHUNK_SIZE = 10
DEFAULT_API_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_LOCATIONS = 0
DEFAULT_PAUSE_BEFORE_GPT = False
DEFAULT_ENABLE_WEB_BROWSING = False

class LocationAnalyzer:
    def __init__(self, kmz_file: str, output_excel: str = None, 
                 verbose: bool = False,
                 openai_api_key: str = "", use_gpt: bool = True, 
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 max_locations: int = DEFAULT_MAX_LOCATIONS,
                 pause_before_gpt: bool = DEFAULT_PAUSE_BEFORE_GPT,
                 enable_web_browsing: bool = DEFAULT_ENABLE_WEB_BROWSING,
                 primary_place_types: List[str] = None,
                 additional_place_types: List[str] = None,
                 special_place_types: List[str] = None,
                 progress_callback: Optional[Callable[[str], None]] = None,
                 status_callback: Optional[Callable[[str], None]] = None):
        """Initialize the location analyzer with the KMZ file path.
        
        Args:
            kmz_file: Path to the KMZ file containing the boundary
            output_excel: Path to save the output Excel file (if None, auto-generated from KMZ filename)
            verbose: Enable verbose output
            openai_api_key: OpenAI API key for GPT population estimation
            use_gpt: Enable GPT population estimation
            chunk_size: Number of locations to process per GPT request
            max_locations: Maximum number of locations to process (0 for no limit)
            pause_before_gpt: Pause before GPT population estimation for review (not used in Streamlit)
            enable_web_browsing: Enable web browsing capabilities for GPT
            primary_place_types: List of primary place types to search for
            additional_place_types: List of additional place types to search for
            special_place_types: List of special place types to search for
            progress_callback: Callback function for progress updates (takes message string)
            status_callback: Callback function for status updates (takes message string)
        """
        self.kmz_file = kmz_file
        self.output_excel = output_excel if output_excel else self._generate_output_filename(kmz_file)
        self.verbose = verbose
        self.openai_api_key = openai_api_key
        self.use_gpt = use_gpt
        self.chunk_size = chunk_size
        self.max_locations = max_locations
        self.pause_before_gpt = pause_before_gpt
        self.enable_web_browsing = enable_web_browsing
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        
        # Progress callbacks
        self.progress_callback = progress_callback or (lambda msg: None)
        self.status_callback = status_callback or (lambda msg: None)
        
        # Place types
        self.primary_place_types = primary_place_types or ['city', 'town', 'district', 'county', 'municipality', 'borough', 'suburb']
        self.additional_place_types = additional_place_types or ['neighbourhood', 'village', 'locality']
        self.special_place_types = special_place_types or []
        
        self.primary_types_pattern = "|".join(self.primary_place_types)
        self.additional_types_pattern = "|".join(self.additional_place_types)
        self.special_types_pattern = "|".join(self.special_place_types) if self.special_place_types else "^$"
        
        # Initialize OpenAI client if using GPT
        self.gpt_client = None
        if self.use_gpt and self.openai_api_key:
            self.gpt_client = OpenAI(api_key=self.openai_api_key)
            self.gpt_model = "gpt-4-turbo"
            self.status_callback(f"GPT population estimation enabled using model: {self.gpt_model}")
        elif self.use_gpt and not self.openai_api_key:
            self.status_callback("Warning: GPT usage enabled, but no OpenAI API key found. Skipping GPT estimation.")
            self.use_gpt = False
        
        # Check if KMZ file exists
        if not os.path.exists(kmz_file):
            raise FileNotFoundError(f"KMZ file not found: {kmz_file}")
    
    def _log(self, message: str):
        """Log a message using callbacks."""
        self.progress_callback(message)
        if self.verbose:
            self.status_callback(message)
    
    def _generate_output_filename(self, kmz_file: str) -> str:
        """Generate output Excel filename based on KMZ filename."""
        base_name = os.path.splitext(os.path.basename(kmz_file))[0]
        return f"{base_name}_gpt_results.xlsx"
    
    def extract_boundary_from_kmz(self) -> List[Tuple[float, float]]:
        """Extract boundary coordinates from KMZ file."""
        self._log(f"Extracting boundary from KMZ file: {self.kmz_file}")
        
        try:
            with zipfile.ZipFile(self.kmz_file, 'r') as kmz:
                kml_data = None
                for filename in kmz.namelist():
                    if filename.endswith('.kml'):
                        kml_data = kmz.read(filename)
                        break
                
                if not kml_data:
                    raise ValueError(f"No KML file found in the KMZ archive")
                
                try:
                    root = ET.fromstring(kml_data)
                except ET.ParseError as e:
                    raise ValueError(f"Invalid KML format: {str(e)}")
                
                namespace = None
                if '{' in root.tag:
                    namespace = root.tag.split('}')[0][1:]
                
                def find_coordinates(elem, namespace=None):
                    if elem.tag.endswith('coordinates') or (namespace and elem.tag == f'{{{namespace}}}coordinates'):
                        return elem.text.strip() if elem.text else None
                    
                    for child in elem:
                        if child.tag.endswith('Polygon') or (namespace and child.tag == f'{{{namespace}}}Polygon'):
                            for grand in child:
                                if grand.tag.endswith('outerBoundaryIs') or (namespace and grand.tag == f'{{{namespace}}}outerBoundaryIs'):
                                    for great in grand:
                                        if great.tag.endswith('LinearRing') or (namespace and great.tag == f'{{{namespace}}}LinearRing'):
                                            for great_great in great:
                                                if great_great.tag.endswith('coordinates') or (namespace and great_great.tag == f'{{{namespace}}}coordinates'):
                                                    return great_great.text.strip() if great_great.text else None
                    
                    for child in elem:
                        result = find_coordinates(child, namespace)
                        if result:
                            return result
                    
                    return None
                
                coord_text = find_coordinates(root, namespace)
                
                if not coord_text:
                    for elem in root.findall('.//*'):
                        if elem.tag.endswith('coordinates') or (namespace and elem.tag == f'{{{namespace}}}coordinates'):
                            if elem.text and elem.text.strip():
                                coord_text = elem.text.strip()
                                break
                
                if not coord_text:
                    raise ValueError("No coordinate elements found in KML")
                
                geo_points = []
                for coord in coord_text.split():
                    if coord:
                        parts = coord.split(',')
                        if len(parts) >= 2:
                            try:
                                lon, lat = float(parts[0]), float(parts[1])
                                geo_points.append((lon, lat))
                            except ValueError:
                                continue
                
                if not geo_points:
                    raise ValueError("No valid boundary points found in coordinates")
                
                self._log(f"Successfully extracted {len(geo_points)} boundary points")
                return geo_points
                
        except Exception as e:
            self._log(f"Error extracting KMZ file: {str(e)}")
            raise
    
    def is_point_in_polygon(self, point: Tuple[float, float], polygon_points: List[Tuple[float, float]]) -> bool:
        """Check if a point is inside the polygon using ray casting algorithm."""
        x, y = point
        inside = False
        
        j = len(polygon_points) - 1
        for i in range(len(polygon_points)):
            if ((polygon_points[i][1] > y) != (polygon_points[j][1] > y)) and \
               (x < (polygon_points[j][0] - polygon_points[i][0]) * (y - polygon_points[i][1]) / \
                (polygon_points[j][1] - polygon_points[i][1]) + polygon_points[i][0]):
                inside = not inside
            j = i
        
        return inside
    
    def haversine(self, lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance between two points on the Earth."""
        R = 6371  # Earth radius in kilometers
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2)**2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def point_segment_distance(self, p_lat, p_lon, lat1, lon1, lat2, lon2):
        """Calculate the shortest distance from a point to a line segment using Haversine."""
        dist_p1 = self.haversine(p_lat, p_lon, lat1, lon1)
        dist_p2 = self.haversine(p_lat, p_lon, lat2, lon2)
        segment_len = self.haversine(lat1, lon1, lat2, lon2)

        if segment_len < 1e-9:
            return dist_p1

        dot_product = ((p_lon - lon1) * (lon2 - lon1) + (p_lat - lat1) * (lat2 - lat1))
        squared_len = (lon2 - lon1)**2 + (lat2 - lat1)**2

        t = 0 if squared_len == 0 else max(0, min(1, dot_product / squared_len))

        closest_lon = lon1 + t * (lon2 - lon1)
        closest_lat = lat1 + t * (lat2 - lat1)

        dist_to_closest_point = self.haversine(p_lat, p_lon, closest_lat, closest_lon)

        if t == 0:
            return dist_p1
        elif t == 1:
            return dist_p2
        else:
            return dist_to_closest_point

    def is_point_near_polygon(self, point_lat, point_lon, polygon_points, buffer_km=2.0):
        """Check if a point is within buffer_km of any segment of the polygon boundary."""
        min_dist = float('inf')
        num_points = len(polygon_points)
        if num_points < 2:
            return False

        for i in range(num_points):
            p1_lon, p1_lat = polygon_points[i]
            p2_lon, p2_lat = polygon_points[(i + 1) % num_points] 
            
            dist_seg = self.point_segment_distance(point_lat, point_lon, p1_lat, p1_lon, p2_lat, p2_lon)
            min_dist = min(min_dist, dist_seg)
            
            if min_dist <= buffer_km:
                return True

        return False
    
    def _get_bounding_box(self, geo_points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """Get the bounding box of a polygon with buffer."""
        lons, lats = zip(*geo_points)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        buffer = 0.04
        min_lon -= buffer
        max_lon += buffer
        min_lat -= buffer
        max_lat += buffer
        
        return min_lat, min_lon, max_lat, max_lon
    
    def _create_osm_query(self, min_lat, min_lon, max_lat, max_lon, place_types_pattern):
        """Create a common OSM Overpass query for different place types."""
        return f"""
        [out:json][timeout:60];
        (
          node["place"~"{place_types_pattern}"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["place"~"{place_types_pattern}"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["place"~"{place_types_pattern}"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """
    
    def _create_primary_osm_query(self, min_lat, min_lon, max_lat, max_lon):
        """Create query for primary locations."""
        return self._create_osm_query(min_lat, min_lon, max_lat, max_lon, self.primary_types_pattern)

    def _create_additional_osm_query(self, min_lat, min_lon, max_lat, max_lon):
        """Create query for additional places."""
        return self._create_osm_query(min_lat, min_lon, max_lat, max_lon, self.additional_types_pattern)

    def _create_special_osm_query(self, min_lat, min_lon, max_lat, max_lon):
        """Create query for special locations."""
        return f"""
        [out:json][timeout:60];
        (
          node["landuse"="commercial"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["landuse"="commercial"]({min_lat},{min_lon},{max_lat},{max_lon});
          relation["landuse"="commercial"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """
    
    def _process_osm_results(self, response_data, place_type_filter=None, skip_city_block=True):
        """Process OSM query results, store selected tags, and filter by geometry."""
        locations = []
        if "elements" not in response_data:
            return locations
        
        osm_elements = response_data["elements"]
        self._log(f"Processing {len(osm_elements)} OSM elements...")
        
        for element in osm_elements:
            tags = element.get("tags", {})
            element_type = element.get("type")
            element_id = element.get("id")

            location_type = None
            if "place" in tags:
                place_type = tags["place"]
                if place_type_filter and place_type not in place_type_filter:
                    continue
                if skip_city_block and place_type == "city_block":
                    continue
                location_type = place_type
            elif tags.get("landuse") == "commercial":
                if place_type_filter is None or "commercial_area" in place_type_filter: 
                    location_type = "commercial_area"
                else:
                    continue
            elif tags.get("boundary") == "administrative":
                if place_type_filter is None: 
                    location_type = "administrative"
                else:
                    continue
            else:
                continue

            name = tags.get("name", "")
            if not name:
                continue
                    
            lat, lon = 0, 0
            if element_type == "node":
                lat, lon = element.get("lat", 0), element.get("lon", 0)
            elif "center" in element:
                lat, lon = element["center"].get("lat", 0), element["center"].get("lon", 0)
            elif "lat" in element and "lon" in element:
                lat, lon = element.get("lat", 0), element.get("lon", 0)
            else:
                continue
                
            is_inside = self.is_point_in_polygon((lon, lat), self.polygon_points)
            is_near = False
            if not is_inside:
                is_near = self.is_point_near_polygon(lat, lon, self.polygon_points, buffer_km=2.0)

            if not (is_inside or is_near):
                continue

            osm_population_tag = tags.get('population', None)
            try:
                if osm_population_tag is not None:
                     osm_population_tag = int(str(osm_population_tag).replace(',','').replace(' ',''))
                else:
                     osm_population_tag = None
            except (ValueError, TypeError):
                 osm_population_tag = None
                 
            osm_population_date = tags.get('population:date', None)
            wikidata_id = tags.get('wikidata', None)
            wikipedia_link = tags.get('wikipedia', None)
            
            location_data = {
                "name": name,
                "type": location_type,
                "latitude": lat,
                "longitude": lon,
                "osm_population_tag": osm_population_tag,
                "osm_population_date": osm_population_date,
                "wikidata_id": wikidata_id,
                "wikipedia_link": wikipedia_link,
            }
            locations.append(location_data)
        
        self._log(f"Finished processing OSM elements. Added {len(locations)} locations.")
        return locations

    def _find_osm_locations(self, polygon_points, query_type, place_types=None, skip_city_block=True):
        """Generic function to find OSM locations of various types."""
        min_lat, min_lon, max_lat, max_lon = self._get_bounding_box(polygon_points)
        
        if query_type == "primary":
            query = self._create_primary_osm_query(min_lat, min_lon, max_lat, max_lon)
        elif query_type == "additional":
            query = self._create_additional_osm_query(min_lat, min_lon, max_lat, max_lon)
        elif query_type == "special":
            query = self._create_special_osm_query(min_lat, min_lon, max_lat, max_lon)
        else:
            raise ValueError(f"Invalid query_type: {query_type}")
        
        try:
            response = requests.post(self.overpass_url, data=query)
            response.raise_for_status()
            data = response.json()
            
            return self._process_osm_results(data, place_types, skip_city_block)
        except Exception as e:
            self._log(f"Error querying OpenStreetMap for {query_type} locations: {str(e)}")
            return []
    
    def clean_and_deduplicate_locations(self, locations: List[Dict]) -> List[Dict]:
        """Clean location names and remove duplicates."""
        for location in locations:
            if 'name' in location:
                location['name'] = location['name'].replace('-', ' ')
                
        seen_names = set()
        unique_locations = []
        
        for location in locations:
            location_name = location.get('name', '')
            if location_name and location_name not in seen_names:
                seen_names.add(location_name)
                unique_locations.append(location)
                
        return unique_locations
    
    def _fetch_admin_hierarchy_batch(self, locations: List[Dict]) -> List[Dict]:
        """Get administrative hierarchy for multiple locations in a single batch request."""
        if not locations:
            return locations

        query_parts = []
        for i, location in enumerate(locations):
            lat = location.get('latitude')
            lon = location.get('longitude')
            if lat is not None and lon is not None:
                query_parts.append(f"is_in({lat},{lon}) -> .a{i};")
                query_parts.append(f"area.a{i}[admin_level=\"8\"]; area.a{i}[admin_level=\"6\"]; area.a{i}[admin_level=\"4\"]; area.a{i}[admin_level=\"2\"];")

        if not query_parts:
            return locations

        query = f"""
        [out:json][timeout:{DEFAULT_API_TIMEOUT}];
        {" ".join(query_parts)}
        out tags;
        """
        
        try:
            response = requests.post(self.overpass_url, data=query, timeout=DEFAULT_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            for i, location in enumerate(locations):
                lat = location.get('latitude')
                lon = location.get('longitude')
                if lat is None or lon is None:
                    continue

                hierarchy = {f'level_{level}_name': None for level in [8, 6, 4, 2]}
                hierarchy['containing_level'] = None
                hierarchy['parent_level'] = None
                hierarchy['parent_name'] = None

                found_levels = {}
                for element in data.get('elements', []):
                    tags = element.get('tags', {})
                    level_str = tags.get('admin_level')
                    name = tags.get('name') or tags.get('name:en')
                    if level_str and name and level_str.isdigit():
                        level = int(level_str)
                        level_key = f'level_{level}_name'
                        if level_key in hierarchy:
                            hierarchy[level_key] = name
                            found_levels[level] = name

                sorted_found_levels = sorted(found_levels.keys(), reverse=True)
                if sorted_found_levels:
                    hierarchy['containing_level'] = sorted_found_levels[0]
                    if len(sorted_found_levels) > 1:
                        hierarchy['parent_level'] = sorted_found_levels[1]
                        hierarchy['parent_name'] = found_levels.get(sorted_found_levels[1])

                location['admin_hierarchy'] = hierarchy

            return locations
            
        except requests.exceptions.RequestException as e:
            self._log(f"Warning: Failed to fetch hierarchy batch: {str(e)}")
            return locations
        except Exception as e:
            self._log(f"Warning: Unexpected error fetching hierarchy batch: {str(e)}")
            return locations
    
    def _get_osm_population(self, location):
        """Get population from OpenStreetMap tags if available."""
        try:
            pop_str = location.get("tags", {}).get("population") 
            if pop_str:
                try:
                    return int(pop_str.replace(',', '').replace(' ', ''))
                except ValueError:
                    return None
            return None
        except Exception as e:
            return None
    
    def _prepare_batch_gpt_prompt(self, locations_chunk: List[Dict], start_index: int) -> str:
        """Creates the prompt for a batch of locations for GPT population estimation."""
        prompt_header = ( 
            "You are an AI assistant specializing in accurately retrieving demographic data based on geographic context. "
            "Your task is to estimate the most recent known residential population for each of the locations listed below.\n\n"
            "For each location, you are provided with:\n"
            "- 'index': An identifier for the location within the overall list (starting from {start_index}).\n"
            "- 'name': The name of the specific location.\n"
            "- 'type': The type of the location (e.g., city, suburb, neighbourhood).\n"
            "- 'parent_name': The name of the administrative area directly containing the location.\n"
            "- 'level_4_name': The name of the higher-level administrative area (e.g., province or state).\n"
            "- 'level_2_name': The name of the country.\n\n"
            "**Crucially, use the provided 'type', 'parent_name', 'level_4_name', and 'level_2_name' to disambiguate the location 'name' "
            "and ensure you are retrieving the population for the correct place.**\n\n"
            "Provide your answer ONLY as a single, valid JSON **array**. Each object in the array must correspond to one location from the input list below, respecting the original order.\n"
            "Do not include any introductory text, explanations, or summaries before or after the JSON array.\n"
            "Each object in the JSON array must contain:\n"
            "- 'index': The original integer index provided for the location.\n"
            "- 'population': The estimated population as an integer. If the population is unknown or cannot be reliably estimated, use `null`.\n"
            "- 'confidence': Your confidence in the estimate as a string: 'High', 'Medium', or 'Low'.\n\n"
            "**Estimation Guidance:** For locations with type 'suburb' or 'neighbourhood', if a direct, reliable population figure isn't found in your knowledge, "
            "provide a reasonable **estimate** based on the context (parent_name, level_4_name) and typical population densities for such areas. "
            "Assign **'Medium'** confidence to these well-informed estimates. Assign 'Low' confidence only if even estimation is highly uncertain.\n\n"
            "Example Response Format (for a batch of 3 locations with indices 0, 1, 2):\n"
            "[\n"
            "  {\"index\": 0, \"population\": 186948, \"confidence\": \"High\"},\n"
            "  {\"index\": 1, \"population\": 25000, \"confidence\": \"Medium\"},\n"
            "  {\"index\": 2, \"population\": null, \"confidence\": \"Low\"}\n"
            "]\n\n"
            "--- START LOCATIONS ---"
        )
        
        locations_data = []
        for i, loc in enumerate(locations_chunk):
            original_index = start_index + i
            admin_hierarchy = loc.get('admin_hierarchy', {}) 
            
            location_info = {
                "index": original_index,
                "name": loc.get('name', 'Unknown'),
                "type": loc.get('type', 'unknown'), 
                "parent_name": admin_hierarchy.get('parent_name'), 
                "level_4_name": admin_hierarchy.get('level_4_name'),
                "level_2_name": admin_hierarchy.get('level_2_name')
            }
            locations_data.append(location_info)
            
        locations_json_str = json.dumps(locations_data, indent=2, ensure_ascii=False)

        prompt_footer = (
            "\n--- END LOCATIONS ---\n\n"
            "Provide ONLY the JSON array containing one object for each location listed above, matching the provided indices."
        )

        return f"{prompt_header}\n{locations_json_str}\n{prompt_footer}"
    
    def _get_gpt_populations_batch(self, locations_chunk: List[Dict], start_index: int) -> Dict[int, Dict]:
        """Sends a batch of locations to GPT and parses the JSON array response."""
        prompt = self._prepare_batch_gpt_prompt(locations_chunk, start_index)
        
        try:
            response = self.gpt_client.chat.completions.create(
                model=self.gpt_model, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, 
            )
            raw_response_content = response.choices[0].message.content
                
        except Exception as e:
            self._log(f"Error calling OpenAI API: {str(e)}")
            results_map = {}
            for i in range(len(locations_chunk)):
                original_idx = start_index + i
                results_map[original_idx] = {"population": None, "confidence": "API Error"}
            return results_map

        parsed_results_map = {}
        try:
            gpt_response_json = json.loads(raw_response_content)
            
            if not isinstance(gpt_response_json, list):
                match = re.search(r'\[\s*\{.*?\}\s*\]', raw_response_content, re.DOTALL)
                if match:
                    json_text = match.group(0)
                    try:
                        gpt_response_json = json.loads(json_text)
                        if not isinstance(gpt_response_json, list):
                             raise ValueError("Regex extracted content is not a list.")
                    except Exception as e_regex:
                        raise ValueError("Response is not a valid JSON array, even after regex.")

            processed_indices = set()
            for item_data in gpt_response_json:
                try:
                    if not isinstance(item_data, dict):
                        continue

                    original_idx = item_data.get('index')
                    population = item_data.get('population')
                    confidence = item_data.get('confidence')

                    if not isinstance(original_idx, int): 
                        continue
                         
                    if not (start_index <= original_idx < start_index + len(locations_chunk)):
                        continue
                    
                    pop_value = None
                    if population is not None:
                        try:
                            pop_float = float(population)
                            if pd.notna(pop_float):
                                pop_value = int(pop_float) 
                        except (ValueError, TypeError):
                            pass
                            
                    conf_value = str(confidence) if confidence in ["High", "Medium", "Low"] else None

                    if original_idx in processed_indices:
                         continue
                    parsed_results_map[original_idx] = {
                        "population": pop_value, 
                        "confidence": conf_value
                    }
                    processed_indices.add(original_idx)
                
                except Exception as e:
                     pass

            for i in range(len(locations_chunk)):
                expected_idx = start_index + i
                if expected_idx not in parsed_results_map:
                    parsed_results_map[expected_idx] = {"population": None, "confidence": "Missing"}

        except json.JSONDecodeError as e:
            self._log(f"Error: Failed to decode GPT JSON response: {str(e)}")
            for i in range(len(locations_chunk)):
                 original_idx = start_index + i
                 parsed_results_map[original_idx] = {"population": None, "confidence": "Parse Error"}
        except ValueError as e:
             self._log(f"Error processing GPT response structure: {e}")
             for i in range(len(locations_chunk)):
                 original_idx = start_index + i
                 parsed_results_map[original_idx] = {"population": None, "confidence": "Format Error"}
        except Exception as e:
            self._log(f"Error during GPT response parsing: {str(e)}")
            for i in range(len(locations_chunk)):
                 original_idx = start_index + i
                 parsed_results_map[original_idx] = {"population": None, "confidence": "Parse Error"}

        return parsed_results_map
    
    def estimate_populations(self, locations: List[Dict]) -> List[Dict]:
        """Get population estimates from OSM and GPT."""
        self._log(f"\n--- Starting Population Estimation Phase for {len(locations)} locations ---")
        
        self._log("\n>>> Stage: Fetching population data from OpenStreetMap...")
        updated_locations_osm = []
        for i, location in enumerate(locations):
            result = location.copy()  
            result["osm_population"] = None
            
            try:
                pop_str = location.get("osm_population_tag")
                if pop_str:
                    result["osm_population"] = int(str(pop_str).replace(',', '').replace(' ', '')) if pop_str else None
            except Exception as e:
                pass
            
            updated_locations_osm.append(result)
        
        self._log("<<< Stage: Finished fetching OSM data.")

        final_locations = updated_locations_osm
        if self.use_gpt and self.gpt_client:
            self._log(f"\n>>> Stage: Fetching population data from GPT ({self.gpt_model}) in batches of {self.chunk_size}...")
            
            locations_for_gpt = updated_locations_osm 
            
            for loc in locations_for_gpt:
                loc["gpt_population"] = None
                loc["gpt_confidence"] = None

            num_batches = ceil(len(locations_for_gpt) / self.chunk_size)
            
            for i in range(num_batches):
                start_index = i * self.chunk_size
                end_index = start_index + self.chunk_size
                batch_locations = locations_for_gpt[start_index:end_index]
                
                self._log(f"  Processing GPT batch {i+1}/{num_batches} (Indices {start_index}-{min(end_index, len(locations_for_gpt))-1})...")

                try:
                    gpt_results_map = self._get_gpt_populations_batch(batch_locations, start_index)
                    
                    for original_idx, result_data in gpt_results_map.items():
                        if 0 <= original_idx < len(locations_for_gpt): 
                            locations_for_gpt[original_idx]["gpt_population"] = result_data.get("population")
                            locations_for_gpt[original_idx]["gpt_confidence"] = result_data.get("confidence")
                except Exception as e:
                    self._log(f"Error processing GPT batch {i+1}: {str(e)}")
                    for idx in range(start_index, min(end_index, len(locations_for_gpt))):
                         if 0 <= idx < len(locations_for_gpt):
                            locations_for_gpt[idx]["gpt_confidence"] = "Error"

                self._log(f"  Finished GPT batch {i+1}/{num_batches}.")
                time.sleep(1)

            self._log("<<< Stage: Finished fetching GPT data.")
            final_locations = locations_for_gpt

        else:
             self._log("\n>>> Stage: Skipping GPT population fetch (use_gpt is False or client not initialized).")
             for loc in updated_locations_osm:
                loc["gpt_population"] = None
                loc["gpt_confidence"] = None
             final_locations = updated_locations_osm

        self._log("\n>>> Stage: Assigning Final Population (OSM vs GPT)...")
        assigned_count = 0
        for i, loc in enumerate(final_locations):
            osm_pop = loc.get("osm_population")
            gpt_pop = loc.get("gpt_population")
            gpt_conf = loc.get("gpt_confidence")
            
            loc["final_population"] = 0
            loc["population_source"] = "None"
            
            if osm_pop is not None and osm_pop > 0:
                loc["final_population"] = osm_pop
                loc["population_source"] = "OSM"
                assigned_count += 1
            elif gpt_pop is not None and gpt_pop > 0 and gpt_conf in ["High", "Medium"]:
                loc["final_population"] = gpt_pop
                loc["population_source"] = "GPT"
                assigned_count += 1
        
        self._log(f"<<< Stage: Finished assigning final population values. Assigned population for {assigned_count}/{len(final_locations)} locations.")
        
        self._log("--- Population Estimation Phase Complete ---")

        return final_locations
    
    def save_to_excel(self, all_locations=None, filename=None) -> BytesIO:
        """Save final analysis results to Excel and return as BytesIO."""
        if all_locations is None or not all_locations:
            self._log("No locations to save.")
            return None
        
        if not filename:
            filename = self.output_excel
        
        try:
            for loc in all_locations:
                 if 'admin_hierarchy' not in loc: loc['admin_hierarchy'] = {}
                 
            df = pd.json_normalize(all_locations, sep='_')
            
            final_columns = [
                'name',
                'type',
                'latitude',
                'longitude',
                'admin_hierarchy_containing_level',
                'admin_hierarchy_parent_name',
                'admin_hierarchy_level_4_name',
                'osm_population_tag',
                'osm_population_date',
                'gpt_population',
                'gpt_confidence',
                'final_population',
                'population_source'
            ]
            
            for col in final_columns:
                if col not in df.columns:
                    df[col] = None
            
            df = df[final_columns]
            df.columns = [col.replace('admin_hierarchy_', '') for col in df.columns]
            
            numeric_cols = ['osm_population_tag', 'gpt_population', 'containing_level', 'final_population']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Float64')
            
            output = BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            
            self._log(f"\nSaved {len(all_locations)} locations to Excel file.")
            return output
            
        except Exception as e:
            self._log(f"Error saving to Excel: {str(e)}")
            traceback.print_exc()
            return None
    
    def run(self):
        """Run the full analysis process."""
        try:
            self.polygon_points = self.extract_boundary_from_kmz()
            
            self._log("\n--- Finding OSM Locations ---")
            primary_locations = self._find_osm_locations(self.polygon_points, "primary", self.primary_place_types)
            
            additional_places = []
            if self.additional_place_types:
                additional_places = self._find_osm_locations(self.polygon_points, "additional", self.additional_place_types)
            
            special_locations = []
            if self.special_place_types: 
                special_locations = self._find_osm_locations(self.polygon_points, "special", self.special_place_types)
            
            administrative_areas = [] 
            
            all_locations = primary_locations + additional_places + special_locations + administrative_areas
            all_locations = self.clean_and_deduplicate_locations(all_locations)
            
            self._log("\n--- Fetching Administrative Hierarchy ---")
            batch_size = 10
            for i in range(0, len(all_locations), batch_size):
                batch = all_locations[i:i + batch_size]
                self._fetch_admin_hierarchy_batch(batch)
            
            self._log("Finished fetching hierarchy.")

            self._log(f"\n--- OSM Location Summary (Post-Hierarchy) ---")
            self._log(f"Total unique OSM locations found: {len(all_locations)}")

            if self.max_locations > 0 and len(all_locations) > self.max_locations:
                self._log(f"\nLimiting results to {self.max_locations} locations (out of {len(all_locations)} found)")
                prioritized = []
                prioritized.extend(primary_locations)
                remaining = self.max_locations - len(prioritized)
                if remaining > 0: prioritized.extend(additional_places[:remaining])
                remaining = self.max_locations - len(prioritized)
                if remaining > 0: prioritized.extend(special_locations[:remaining])
                remaining = self.max_locations - len(prioritized)
                if remaining > 0: prioritized.extend(administrative_areas[:remaining])
                all_locations = prioritized[:self.max_locations]
                self._log(f"Processing {len(all_locations)} locations after limiting.")

            self._log("\n--- Starting Population Estimation (OSM/GPT) ---")
            all_locations = self.estimate_populations(all_locations)
            
            return all_locations
            
        except Exception as e:
            self._log(f"An error occurred during analysis: {str(e)}")
            traceback.print_exc()
            return None
