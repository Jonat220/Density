# Building Density Calculator - Technical Documentation

## Overview
The Building Density Calculator is a Streamlit web application that analyzes urban building density by counting OpenStreetMap (OSM) buildings within a specified radius and computing density metrics per area unit.

## Core Functionality
- **Geocoding**: Convert addresses to coordinates using Nominatim
- **Data Fetching**: Query OSM building data via Overpass API
- **Geometry Processing**: Convert OSM elements to valid polygons
- **Density Calculation**: Compute buildings per square kilometer/mile
- **Height Analysis**: Extract and analyze building heights from OSM tags
- **Visualization**: Interactive map with building overlays and search radius

## Architecture Overview

### Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **Geocoding**: Nominatim (OpenStreetMap geocoding service)
- **Data Source**: Overpass API (OpenStreetMap query interface)
- **Geometry**: Shapely (geometric operations)
- **Mapping**: Folium (interactive maps)
- **HTTP**: Requests (API communication)

### Dependencies
```
streamlit>=1.36.0      # Web UI framework
requests>=2.31.0       # HTTP client for API calls
shapely>=2.0.0         # Geometric operations
pyproj>=3.6.0          # Coordinate system transformations
geopy>=2.4.1           # Geocoding services
folium>=0.16.0         # Interactive maps
streamlit-folium>=0.20.0  # Streamlit-Folium integration
```

## Code Structure Breakdown

### 1. Imports and Configuration
```python
import streamlit as st
import requests
from geopy.geocoders import Nominatim
from shapely.geometry import Polygon
import folium
from streamlit_folium import st_folium

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter", 
    "https://overpass.openstreetmap.ru/api/interpreter",
]
```
**Purpose**: Import required libraries and define Overpass API endpoints with fallback mirrors.

### 2. Utility Functions

#### Unit Conversion Helpers
```python
def miles_to_meters(miles: float) -> float:
    return miles * 1609.344

def kilometers_to_meters(km: float) -> float:
    return km * 1000.0

def meters_to_sq_km(m2: float) -> float:
    return m2 / 1_000_000.0

def meters_to_sq_miles(m2: float) -> float:
    return m2 / 2_589_988.110336
```
**Purpose**: Convert between different distance and area units for density calculations.

#### Geocoding Functions
```python
def geocode_location(query: str) -> Tuple[float, float]:
    geolocator = Nominatim(user_agent="building-density-app")
    location = geolocator.geocode(query)
    if not location:
        raise ValueError("Location not found. Try a more specific address or coordinates.")
    return (location.latitude, location.longitude)

def parse_lat_lon_from_string(s: str) -> Tuple[float, float] | None:
    """Parse coordinates from text input like 'lat, lon'"""
    # Validates coordinate ranges and formats
```
**Purpose**: Convert addresses to coordinates and parse coordinate strings from user input.

### 3. OpenStreetMap Data Fetching

#### Query Builder
```python
def build_overpass_query(lat: float, lon: float, radius_m: float, timeout_s: int) -> str:
    query = f"""
    [out:json][timeout:{timeout_s}];
    (
        way["building"](around:{radius_m},{lat},{lon});
        rel["building"](around:{radius_m},{lat},{lon});
    );
    out tags geom;
    """
    return query
```
**Purpose**: Construct Overpass QL query to fetch building features within specified radius.

#### Data Fetcher with Retry Logic
```python
def fetch_buildings(lat, lon, radius_m, endpoint, timeout_s, retries) -> List[Dict[str, Any]]:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                endpoint,
                data={"data": build_overpass_query(lat, lon, radius_m, timeout_s)},
                timeout=timeout_s + 30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("elements", [])
        except Exception as e:
            last_err = e
            # Exponential backoff
            time.sleep(min(2 ** attempt, 5))
    if last_err:
        raise last_err
    return []
```
**Purpose**: Execute Overpass query with retry mechanism and exponential backoff for reliability.

### 4. Geometry Processing

#### OSM Element to Polygon Conversion
```python
def element_to_polygon(element: Dict[str, Any]) -> Polygon | None:
    geom = element.get("geometry")
    if not geom:
        return None
    coords = [(pt["lon"], pt["lat"]) for pt in geom]
    if len(coords) >= 3 and (coords[0] != coords[-1]):
        coords.append(coords[0])  # Close polygon
    try:
        polygon = Polygon(coords)
        if polygon.is_valid and polygon.area > 0:
            return polygon
    except Exception:
        return None
    return None
```
**Purpose**: Convert OSM geometry elements to valid Shapely polygons with validation.

#### Building Counting and Polygon Collection
```python
def count_buildings_and_polygons(elements: List[Dict[str, Any]]) -> Tuple[int, List[Polygon]]:
    polygons: List[Polygon] = []
    for el in elements:
        if el.get("type") in {"way", "relation"}:
            poly = element_to_polygon(el)
            if poly is not None:
                polygons.append(poly)
    return len(polygons), polygons
```
**Purpose**: Filter valid building polygons and count them for density calculations.

### 5. Density Calculations

#### Core Density Computation
```python
def compute_density(num_buildings: int, radius_m: float) -> Dict[str, float]:
    circle_area_m2 = math.pi * (radius_m ** 2)
    return {
        "per_sq_km": num_buildings / meters_to_sq_km(circle_area_m2) if circle_area_m2 > 0 else 0.0,
        "per_sq_mile": num_buildings / meters_to_sq_miles(circle_area_m2) if circle_area_m2 > 0 else 0.0,
        "area_sq_km": meters_to_sq_km(circle_area_m2),
        "area_sq_miles": meters_to_sq_miles(circle_area_m2),
    }
```
**Purpose**: Calculate building density per square kilometer and square mile using circle area.

### 6. Height Analysis

#### Building Height Extraction
```python
def extract_building_height(element: Dict[str, Any]) -> float | None:
    """Extract building height from OSM tags. Returns height in meters."""
    tags = element.get("tags", {})
    
    # Check for height tag (most common)
    if "height" in tags:
        # Handle different height formats (m, ft, raw numbers)
        # Convert feet to meters if needed
    
    # Check for building:levels tag and estimate height
    if "building:levels" in tags:
        # Estimate 3 meters per floor
    
    # Check for max_height tag
    if "max_height" in tags:
        # Handle max_height with unit conversion
```
**Purpose**: Extract building height information from various OSM tag formats and convert to meters.

#### Height Statistics Computation
```python
def compute_height_statistics(heights: List[float]) -> Dict[str, float]:
    """Compute height statistics from a list of building heights."""
    # Calculate average, median, min, max heights
    # Track buildings with height data vs total buildings
```
**Purpose**: Calculate comprehensive height statistics including average, median, and range.

### 7. Map Visualization

#### Interactive Map Generation
```python
def make_map(lat: float, lon: float, radius_m: float, building_polygons: List[Polygon]) -> folium.Map:
    m = folium.Map(location=[lat, lon], zoom_start=14, control_scale=True)
    
    # Draw search radius circle
    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        color="#1f77b4",
        fill=True,
        fill_opacity=0.05,
        weight=2,
    ).add_to(m)
    
    # Overlay building polygons
    for poly in building_polygons:
        try:
            folium.GeoJson(
                data=poly.__geo_interface__,
                style_function=lambda x: {
                    "color": "#d62728", 
                    "weight": 1, 
                    "fillColor": "#ff9896", 
                    "fillOpacity": 0.5
                },
            ).add_to(m)
        except Exception:
            pass
    
    # Add center marker
    folium.Marker([lat, lon], icon=folium.Icon(color="blue", icon="info-sign"), tooltip="Center").add_to(m)
    return m
```
**Purpose**: Create interactive map with search radius, building overlays, and center marker.

### 7. User Interface

#### Custom Styling
```python
def inject_styles() -> None:
    st.markdown("""
        <style>
            /* Modern UI styling for cards, buttons, sidebar, metrics */
            .card {
                background: var(--card-bg);
                border: 1px solid rgba(2,6,23,0.06);
                box-shadow: 0 8px 24px rgba(2,6,23,0.06);
                border-radius: 14px;
                padding: 18px 16px;
            }
            /* Additional styling for buttons, inputs, metrics */
        </style>
    """, unsafe_allow_html=True)
```
**Purpose**: Apply modern, professional styling to the Streamlit interface.

#### Main Application Flow
```python
def main() -> None:
    # 1. Page Setup
    st.set_page_config(page_title="Building Density Calculator", layout="wide")
    inject_styles()
    
    # 2. Sidebar Inputs
    with st.sidebar:
        input_mode = st.radio("Input Mode", ["Address", "Coordinates"])
        # Address or coordinate inputs
        units = st.selectbox("Radius Units", ["kilometers", "miles"])
        radius_value = st.number_input("Radius", min_value=0.1, max_value=50.0)
        query_btn = st.button("Calculate", type="primary")
    
    # 3. Input Processing
    # Geocode address or parse coordinates
    
    # 4. Live Map Preview
    # Show map with current inputs
    
    # 5. Calculation Trigger
    if query_btn:
        # Fetch data, process, compute density, cache results
    
    # 6. Results Display
    # Show metrics and final map with overlays
```
**Purpose**: Orchestrate the complete user experience from input to results.

## Data Flow Architecture

```
User Input (Address/Coordinates + Radius)
           ↓
    Geocoding (if needed)
           ↓
    Overpass Query Construction
           ↓
    OSM Data Fetching (with retries)
           ↓
    Geometry Processing (OSM → Polygons)
           ↓
    Density Calculation (Count/Area)
           ↓
    Results Caching
           ↓
    Map Visualization + Metrics Display
```

## Key Features

### 1. Caching System
- **Cache Key**: `{endpoint}|{lat}|{lon}|{radius}|{timeout}`
- **Storage**: Streamlit session state
- **Benefit**: Instant repeat queries, reduced API load

### 2. Error Handling
- **Geocoding**: Graceful fallback for invalid addresses
- **API Calls**: Retry mechanism with exponential backoff
- **Geometry**: Validation of polygon integrity
- **User Feedback**: Clear error messages

### 3. Performance Optimizations
- **Lazy Loading**: Map updates only when needed
- **Session State**: Persist results across interactions
- **Efficient Queries**: Optimized Overpass QL syntax

### 4. User Experience
- **Dual Input Modes**: Address or direct coordinates
- **Live Preview**: Map updates as user adjusts parameters
- **Responsive Design**: Modern UI with custom styling
- **Comprehensive Metrics**: Multiple density units and metadata

## Usage Instructions

### Installation
```bash
cd /path/to/density/app
pip install -r requirements.txt
streamlit run app.py
```

### Operation
1. **Input Location**: Choose Address or Coordinates mode
2. **Set Parameters**: Enter location and radius (km/miles)
3. **Calculate**: Click "Calculate" button
4. **View Results**: Examine metrics and interactive map
5. **Interpret**: Consider OSM data completeness for your area

### Output Metrics
- **Building Count**: Total buildings found in radius
- **Search Area**: Circle area in sq km and sq miles
- **Density**: Buildings per sq km and per sq mile
- **Metadata**: Query details, timestamp, cache status

## Limitations and Considerations

### Data Quality
- **OSM Coverage**: Building data completeness varies by region
- **Community Sourced**: Accuracy depends on local contributors
- **Update Frequency**: Data may not reflect recent development

### Technical Constraints
- **API Limits**: Overpass has rate limiting and timeout restrictions
- **Geometry Complexity**: Very large areas may timeout
- **Coordinate Precision**: Results depend on input accuracy

### Best Practices
- **Start Small**: Use smaller radii for initial testing
- **Verify Results**: Cross-reference with known areas
- **Consider Context**: Urban vs rural areas have different coverage
- **Cache Wisely**: Results persist in session for efficiency

## Future Enhancements

### Potential Improvements
- **Multiple Data Sources**: Integrate additional building datasets
- **Advanced Analytics**: Height data, building types, temporal analysis
- **Export Features**: CSV/GeoJSON export of results
- **Batch Processing**: Multiple location analysis
- **Custom Areas**: Polygon-based instead of circular search

### Technical Upgrades
- **Database Integration**: Persistent caching across sessions
- **API Optimization**: Parallel queries for large areas
- **Enhanced Visualization**: 3D building models, heat maps
- **Mobile Optimization**: Responsive design improvements

## Troubleshooting

### Common Issues
1. **No Results**: Check OSM coverage for your area
2. **Timeout Errors**: Reduce radius or try different endpoint
3. **Geocoding Failures**: Use coordinates mode for precise locations
4. **Map Display Issues**: Check browser compatibility

### Debug Information
- **Run Metadata**: Available in expandable section
- **Cache Status**: Shows if results were cached
- **Query Details**: Endpoint, timeout, retry information
- **Error Messages**: Specific failure reasons

---

*This documentation provides a comprehensive technical overview of the Building Density Calculator application, suitable for presentations, development reference, and user guidance.*
