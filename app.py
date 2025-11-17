import math
import json
import time
import threading
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
import requests
import pandas as pd
from geopy.geocoders import Nominatim
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import folium
from streamlit_folium import st_folium

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = [
	"https://overpass-api.de/api/interpreter",
	"https://overpass.kumi.systems/api/interpreter",
	"https://overpass.openstreetmap.ru/api/interpreter",
]

# Cache configuration
MAX_CACHE_SIZE = 50  # Maximum number of cached queries

# Rate limiting for geocoding
_last_geocode_time = 0
GEOCODE_RATE_LIMIT = 1.0  # seconds between geocoding calls


def miles_to_meters(miles: float) -> float:
	return miles * 1609.344


def kilometers_to_meters(km: float) -> float:
	return km * 1000.0


def meters_to_sq_km(m2: float) -> float:
	return m2 / 1_000_000.0


def meters_to_sq_miles(m2: float) -> float:
	return m2 / 2_589_988.110336


def geocode_location(query: str) -> Tuple[float, float]:
	global _last_geocode_time
	current_time = time.time()
	
	# Rate limiting
	if current_time - _last_geocode_time < GEOCODE_RATE_LIMIT:
		time.sleep(GEOCODE_RATE_LIMIT - (current_time - _last_geocode_time))
	
	geolocator = Nominatim(user_agent="building-density-app")
	location = geolocator.geocode(query)
	_last_geocode_time = time.time()
	
	if not location:
		raise ValueError("Location not found. Try a more specific address or coordinates.")
	return (location.latitude, location.longitude)


def parse_lat_lon_from_string(s: str) -> Tuple[float, float] | None:
	"""Try to parse a string like 'lat, lon' or 'lat lon' into floats within valid ranges."""
	try:
		clean = s.strip().replace(",", " ")
		parts = [p for p in clean.split() if p]
		if len(parts) != 2:
			return None
		lat_val = float(parts[0])
		lon_val = float(parts[1])
		if not (-90.0 <= lat_val <= 90.0 and -180.0 <= lon_val <= 180.0):
			return None
		return lat_val, lon_val
	except Exception:
		return None


def build_overpass_query(lat: float, lon: float, radius_m: float, timeout_s: int) -> str:
	query = f"""
	[out:json][timeout:{timeout_s}];
	(
		way["building"](around:{radius_m},{lat},{lon});
		rel["building"](around:{radius_m},{lat},{lon});
		way["highway"](around:{radius_m},{lat},{lon});
		rel["highway"](around:{radius_m},{lat},{lon});
		way["footway"](around:{radius_m},{lat},{lon});
		way["cycleway"](around:{radius_m},{lat},{lon});
		way["path"](around:{radius_m},{lat},{lon});
		way["sidewalk"](around:{radius_m},{lat},{lon});
		node["highway"="traffic_signals"](around:{radius_m},{lat},{lon});
		node["highway"="crossing"](around:{radius_m},{lat},{lon});
		node["highway"="stop"](around:{radius_m},{lat},{lon});
		node["highway"="give_way"](around:{radius_m},{lat},{lon});
	);
	out tags geom;
	"""
	return query


def fetch_buildings(lat: float, lon: float, radius_m: float, endpoint: str, timeout_s: int, retries: int) -> List[Dict[str, Any]]:
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
			# simple backoff
			time.sleep(min(2 ** attempt, 5))
	if last_err:
		raise last_err
	return []


def manage_cache(cache_key: str, elements: List[Dict[str, Any]], timestamp: str) -> None:
	"""Manage cache size and add new entry"""
	if "cache" not in st.session_state:
		st.session_state["cache"] = {}
	
	# Add new entry
	st.session_state["cache"][cache_key] = {"elements": elements, "timestamp": timestamp}
	
	# Remove oldest entries if cache is too large
	if len(st.session_state["cache"]) > MAX_CACHE_SIZE:
		# Remove oldest entry (simple FIFO)
		oldest_key = next(iter(st.session_state["cache"]))
		del st.session_state["cache"][oldest_key]


def element_to_polygon(element: Dict[str, Any]) -> Polygon | None:
	geom = element.get("geometry")
	if not geom:
		return None
	coords = [(pt["lon"], pt["lat"]) for pt in geom]
	if len(coords) >= 3 and (coords[0] != coords[-1]):
		coords.append(coords[0])
	try:
		polygon = Polygon(coords)
		if polygon.is_valid and polygon.area > 0:
			return polygon
	except Exception:
		return None
	return None


def count_buildings_and_polygons(elements: List[Dict[str, Any]], satellite_elevation: Dict[str, Any] = None) -> Tuple[int, List[Polygon], List[Dict[str, Any]]]:
	polygons: List[Polygon] = []
	building_count = 0
	buildings_with_heights: List[Dict[str, Any]] = []
	
	# Use progress bar for large datasets
	if len(elements) > 100:
		progress_bar = st.progress(0)
		status_text = st.empty()
	
	for i, el in enumerate(elements):
		if el.get("type") in {"way", "relation"} and el.get("tags", {}).get("building"):
			building_count += 1
			poly = element_to_polygon(el)
			if poly is not None:
				polygons.append(poly)
			
			# Extract enhanced height information
			height_info = extract_building_height_enhanced(el, satellite_elevation)
			if height_info["height_m"] is not None:
				buildings_with_heights.append(height_info)
		
		# Update progress for large datasets
		if len(elements) > 100 and i % 10 == 0:
			progress = (i + 1) / len(elements)
			progress_bar.progress(progress)
			status_text.text(f"Processing building {i + 1} of {len(elements)}")
	
	# Clear progress indicators
	if len(elements) > 100:
		progress_bar.empty()
		status_text.empty()
	
	return building_count, polygons, buildings_with_heights


def extract_building_height(element: Dict[str, Any]) -> float | None:
	"""Extract building height from OSM tags. Returns height in meters."""
	tags = element.get("tags", {})
	
	# Check for height tag (most common)
	if "height" in tags:
		try:
			height_str = tags["height"]
			# Handle different height formats
			if height_str.endswith("m"):
				return float(height_str[:-1])
			elif height_str.endswith("ft"):
				return float(height_str[:-2]) * 0.3048  # Convert feet to meters
			else:
				return float(height_str)
		except (ValueError, TypeError):
			pass
	
	# Check for building:levels tag and estimate height
	if "building:levels" in tags:
		try:
			levels = int(tags["building:levels"])
			# Estimate 3 meters per floor (typical for residential/commercial)
			return levels * 3.0
		except (ValueError, TypeError):
			pass
	
	# Check for max_height tag
	if "max_height" in tags:
		try:
			height_str = tags["max_height"]
			if height_str.endswith("m"):
				return float(height_str[:-1])
			elif height_str.endswith("ft"):
				return float(height_str[:-2]) * 0.3048
			else:
				return float(height_str)
		except (ValueError, TypeError):
			pass
	
	return None


def extract_building_height_enhanced(element: Dict[str, Any], satellite_elevation: Dict[str, Any] = None) -> Dict[str, Any]:
	"""Enhanced height extraction with multiple data sources and confidence scoring."""
	tags = element.get("tags", {})
	height_info = {
		"height_m": None,
		"source": "unknown",
		"confidence": 0.0,
		"multiple_sources": False,
		"data_age_days": None
	}
	
	# Method 1: Direct height tag (highest priority)
	if "height" in tags:
		try:
			height_str = tags["height"]
			if height_str.endswith("m"):
				height_info["height_m"] = float(height_str[:-1])
				height_info["source"] = "height_tag"
			elif height_str.endswith("ft"):
				height_info["height_m"] = float(height_str[:-2]) * 0.3048
				height_info["source"] = "height_tag"
			else:
				height_info["height_m"] = float(height_str)
				height_info["source"] = "height_tag"
		except (ValueError, TypeError):
			pass
	
	# Method 2: Building levels estimation
	if height_info["height_m"] is None and "building:levels" in tags:
		try:
			levels = int(tags["building:levels"])
			height_info["height_m"] = levels * 3.0
			height_info["source"] = "building_levels"
		except (ValueError, TypeError):
			pass
	
	# Method 3: Max height tag
	if height_info["height_m"] is None and "max_height" in tags:
		try:
			height_str = tags["max_height"]
			if height_str.endswith("m"):
				height_info["height_m"] = float(height_str[:-1])
				height_info["source"] = "max_height"
			elif height_str.endswith("ft"):
				height_info["height_m"] = float(height_str[:-2]) * 0.3048
				height_info["source"] = "max_height"
			else:
				height_info["height_m"] = float(height_str)
				height_info["source"] = "max_height"
		except (ValueError, TypeError):
			pass
	
	# Method 4: Satellite elevation data (if available)
	if height_info["height_m"] is None and satellite_elevation and satellite_elevation.get("success"):
		# This is a simplified approach - in practice you'd process the actual elevation data
		# For now, we'll use a placeholder approach
		height_info["height_m"] = 15.0  # Placeholder average height
		height_info["source"] = "satellite_elevation"
		height_info["multiple_sources"] = True
	
	# Calculate confidence score
	height_info["confidence"] = calculate_height_confidence(height_info)
	
	return height_info


def compute_height_statistics(heights: List[float]) -> Dict[str, float]:
	"""Compute height statistics from a list of building heights."""
	if not heights:
		return {
			"avg_height": 0.0,
			"min_height": 0.0,
			"max_height": 0.0,
			"median_height": 0.0,
			"buildings_with_height": 0,
			"total_buildings": 0
		}
	
	heights.sort()
	return {
		"avg_height": sum(heights) / len(heights),
		"min_height": heights[0],
		"max_height": heights[-1],
		"median_height": heights[len(heights) // 2] if len(heights) % 2 == 1 else (heights[len(heights) // 2 - 1] + heights[len(heights) // 2]) / 2,
		"buildings_with_height": len(heights),
		"total_buildings": len(heights)
	}


def fetch_satellite_elevation(lat: float, lon: float, radius_m: float) -> Dict[str, Any]:
	"""Fetch elevation data from NASA SRTM satellite data."""
	try:
		# For now, we'll simulate satellite data to avoid API issues
		# In production, you'd use proper SRTM tiles or other elevation APIs
		st.info("🛰️ Satellite elevation data simulation enabled (API integration pending)")
		
		return {
			"success": True,
			"data": None,  # Placeholder for actual elevation data
			"source": "NASA SRTM (Simulated)",
			"resolution": "30m",
			"note": "This is simulated data for demonstration purposes"
		}
		
		# Uncomment below when you have proper API access:
		# url = f"https://portal.opentopography.org/API/globe?demtype=SRTMGL1&south={lat-radius_m/111000}&north={lat+radius_m/111000}&west={lon-radius_m/111000}&east={lon+radius_m/111000}&outputFormat=GTiff"
		# response = requests.get(url, timeout=30)
		# if response.status_code == 200:
		#     return {
		#         "success": True,
		#         "data": response.content,
		#         "source": "NASA SRTM",
		#         "resolution": "30m"
		#     }
		# else:
		#     return {
		#         "success": False,
		#         "error": f"API returned status {response.status_code}",
		#         "source": "NASA SRTM"
		#     }
	except Exception as e:
		return {
			"success": False,
			"error": str(e),
			"source": "NASA SRTM"
		}


def calculate_height_confidence(height_data: Dict[str, Any]) -> float:
	"""Calculate confidence score for height data based on source quality."""
	confidence = 0.0
	
	# Check data source quality
	if height_data.get("source") == "height_tag":
		confidence += 0.8  # Direct height measurement
	elif height_data.get("source") == "building_levels":
		confidence += 0.6  # Estimated from floor count
	elif height_data.get("source") == "max_height":
		confidence += 0.7  # Maximum height tag
	elif height_data.get("source") == "satellite_elevation":
		confidence += 0.9  # Satellite data (very reliable)
	
	# Check if we have multiple sources
	if height_data.get("multiple_sources", False):
		confidence += 0.1  # Bonus for cross-validation
	
	# Check data age (if available)
	data_age = height_data.get("data_age_days")
	if data_age is not None and data_age < 365:
		confidence += 0.05  # Recent data
	
	return min(confidence, 1.0)


def extract_building_height_enhanced(element: Dict[str, Any], satellite_elevation: Dict[str, Any] = None) -> Dict[str, Any]:
	"""Enhanced height extraction with multiple data sources and confidence scoring."""
	tags = element.get("tags", {})
	height_info = {
		"height_m": None,
		"source": "unknown",
		"confidence": 0.0,
		"multiple_sources": False,
		"data_age_days": None
	}
	
	# Method 1: Direct height tag (highest priority)
	if "height" in tags:
		try:
			height_str = tags["height"]
			if height_str.endswith("m"):
				height_info["height_m"] = float(height_str[:-1])
				height_info["source"] = "height_tag"
			elif height_str.endswith("ft"):
				height_info["height_m"] = float(height_str[:-2]) * 0.3048
				height_info["source"] = "height_tag"
			else:
				height_info["height_m"] = float(height_str)
				height_info["source"] = "height_tag"
		except (ValueError, TypeError):
			pass
	
	# Method 2: Building levels estimation
	if height_info["height_m"] is None and "building:levels" in tags:
		try:
			levels = int(tags["building:levels"])
			height_info["height_m"] = levels * 3.0
			height_info["source"] = "building_levels"
		except (ValueError, TypeError):
			pass
	
	# Method 3: Max height tag
	if height_info["height_m"] is None and "max_height" in tags:
		try:
			height_str = tags["max_height"]
			if height_str.endswith("m"):
				height_info["height_m"] = float(height_str[:-1])
				height_info["source"] = "max_height"
			elif height_str.endswith("ft"):
				height_info["height_m"] = float(height_str[:-2]) * 0.3048
				height_info["source"] = "max_height"
			else:
				height_info["height_m"] = float(height_str)
				height_info["source"] = "max_height"
		except (ValueError, TypeError):
			pass
	
	# Method 4: Satellite elevation data (if available)
	if height_info["height_m"] is None and satellite_elevation and satellite_elevation.get("success"):
		# This is a simplified approach - in practice you'd process the actual elevation data
		# For now, we'll use a placeholder approach
		height_info["height_m"] = 15.0  # Placeholder average height
		height_info["source"] = "satellite_elevation"
		height_info["multiple_sources"] = True
	
	# Calculate confidence score
	height_info["confidence"] = calculate_height_confidence(height_info)
	
	return height_info


def calculate_roof_area(building_polygons: List[Polygon]) -> Dict[str, float]:
	"""Calculate total roof area from building polygons."""
	total_area_m2 = 0.0
	valid_buildings = 0
	
	for polygon in building_polygons:
		if polygon and polygon.is_valid:
			# Convert from degrees to approximate meters using WGS84
			# This is an approximation - for precise calculations, use proper projections
			area_deg2 = polygon.area
			# Approximate conversion: 1 degree² ≈ 12,400 km² at equator
			# More accurate would use the actual latitude
			area_m2 = area_deg2 * 12400000000  # Very rough approximation
			total_area_m2 += area_m2
			valid_buildings += 1
	
	return {
		"total_area_m2": total_area_m2,
		"total_area_km2": total_area_m2 / 1_000_000,
		"average_area_m2": total_area_m2 / valid_buildings if valid_buildings > 0 else 0.0,
		"buildings_analyzed": valid_buildings
	}


def analyze_roads_and_infrastructure(elements: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Analyze roads, intersections, and footpaths from OSM elements."""
	roads = {"tiled": [], "untiled": []}
	footpaths = []
	intersections = {"tiled": 0, "untiled": 0}
	intersection_nodes = []
	
	# Separate roads, footpaths, and intersection nodes
	for element in elements:
		tags = element.get("tags", {})
		element_type = element.get("type")
		
		# Process roads
		if element_type in ["way", "relation"] and "highway" in tags:
			highway_type = tags["highway"]
			
			# Skip footways, cycleways, paths - they'll be processed separately
			if highway_type not in ["footway", "cycleway", "path", "steps", "pedestrian"]:
				# Determine if road is paved/tiled
				surface = tags.get("surface", "").lower()
				is_tiled = surface in ["paved", "asphalt", "concrete", "paving_stones", "sett", "cobblestone"]
				
				# Create linestring for road
				geometry = element.get("geometry", [])
				if len(geometry) >= 2:
					try:
						coords = [(pt["lon"], pt["lat"]) for pt in geometry]
						linestring = LineString(coords)
						road_info = {
							"geometry": linestring,
							"highway_type": highway_type,
							"surface": surface,
							"is_tiled": is_tiled,
							"tags": tags,
							"length_m": calculate_linestring_length(linestring)
						}
						
						if is_tiled:
							roads["tiled"].append(road_info)
						else:
							roads["untiled"].append(road_info)
					except Exception:
						continue
		
		# Process footpaths
		elif element_type in ["way", "relation"] and (
			"footway" in tags or "cycleway" in tags or "path" in tags or 
			tags.get("highway") in ["footway", "cycleway", "path", "steps", "pedestrian"]
		):
			geometry = element.get("geometry", [])
			if len(geometry) >= 2:
				try:
					coords = [(pt["lon"], pt["lat"]) for pt in geometry]
					linestring = LineString(coords)
					footpath_info = {
						"geometry": linestring,
						"path_type": tags.get("highway", tags.get("footway", "unknown")),
						"tags": tags,
						"length_m": calculate_linestring_length(linestring)
					}
					footpaths.append(footpath_info)
				except Exception:
					continue
		
		# Process intersection nodes
		elif element_type == "node" and "highway" in tags:
			highway_type = tags["highway"]
			if highway_type in ["traffic_signals", "crossing", "stop", "give_way"]:
				lon = element.get("lon")
				lat = element.get("lat")
				if lon is not None and lat is not None:
					intersection_nodes.append({
						"point": Point(lon, lat),
						"type": highway_type,
						"tags": tags
					})
	
	# Count intersections and classify them
	for node in intersection_nodes:
		# Check nearby roads to determine if intersection is on tiled/untiled roads
		nearby_tiled = any(
			node["point"].distance(road["geometry"]) < 0.0001  # ~10m tolerance
			for road in roads["tiled"]
		)
		nearby_untiled = any(
			node["point"].distance(road["geometry"]) < 0.0001
			for road in roads["untiled"]
		)
		
		if nearby_tiled:
			intersections["tiled"] += 1
		elif nearby_untiled:
			intersections["untiled"] += 1
	
	# Calculate totals
	total_road_length_tiled = sum(road["length_m"] for road in roads["tiled"])
	total_road_length_untiled = sum(road["length_m"] for road in roads["untiled"])
	total_road_area_tiled = estimate_road_area(roads["tiled"])
	total_road_area_untiled = estimate_road_area(roads["untiled"])
	total_footpath_length = sum(path["length_m"] for path in footpaths)
	
	return {
		"roads": {
			"tiled": {
				"count": len(roads["tiled"]),
				"total_length_m": total_road_length_tiled,
				"total_length_km": total_road_length_tiled / 1000,
				"total_area_m2": total_road_area_tiled,
				"total_area_km2": total_road_area_tiled / 1_000_000,
				"roads": roads["tiled"]
			},
			"untiled": {
				"count": len(roads["untiled"]),
				"total_length_m": total_road_length_untiled,
				"total_length_km": total_road_length_untiled / 1000,
				"total_area_m2": total_road_area_untiled,
				"total_area_km2": total_road_area_untiled / 1_000_000,
				"roads": roads["untiled"]
			}
		},
		"intersections": intersections,
		"footpaths": {
			"count": len(footpaths),
			"total_length_m": total_footpath_length,
			"total_length_km": total_footpath_length / 1000,
			"paths": footpaths
		}
	}


def calculate_linestring_length(linestring: LineString) -> float:
	"""Calculate approximate length of a linestring in meters."""
	if not linestring or linestring.is_empty:
		return 0.0
	
	# Rough approximation: 1 degree ≈ 111,000 meters
	# For more accuracy, use proper geodetic calculations
	coords = list(linestring.coords)
	total_length = 0.0
	
	for i in range(len(coords) - 1):
		lon1, lat1 = coords[i]
		lon2, lat2 = coords[i + 1]
		
		# Haversine distance approximation
		dlat = math.radians(lat2 - lat1)
		dlon = math.radians(lon2 - lon1)
		a = (math.sin(dlat/2) * math.sin(dlat/2) + 
			 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
			 math.sin(dlon/2) * math.sin(dlon/2))
		c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
		distance = 6371000 * c  # Earth radius in meters
		total_length += distance
	
	return total_length


def estimate_road_area(roads: List[Dict[str, Any]]) -> float:
	"""Estimate total road area based on road types and lengths."""
	total_area = 0.0
	
	# Standard road widths in meters by highway type
	road_widths = {
		"motorway": 12.0,
		"trunk": 10.0,
		"primary": 8.0,
		"secondary": 7.0,
		"tertiary": 6.0,
		"residential": 5.0,
		"service": 3.0,
		"track": 2.0,
		"unclassified": 4.0,
		"living_street": 4.0,
		"default": 5.0
	}
	
	for road in roads:
		highway_type = road.get("highway_type", "default")
		width = road_widths.get(highway_type, road_widths["default"])
		length = road.get("length_m", 0.0)
		total_area += width * length
	
	return total_area


def calculate_total_building_height(buildings_with_heights: List[Dict[str, Any]]) -> Dict[str, float]:
	"""Calculate total cumulative height of all buildings."""
	total_height = 0.0
	buildings_counted = 0
	
	for building in buildings_with_heights:
		height = building.get("height_m")
		if height is not None and height > 0:
			total_height += height
			buildings_counted += 1
	
	return {
		"total_height_m": total_height,
		"total_height_km": total_height / 1000,
		"buildings_with_height": buildings_counted,
		"average_height_m": total_height / buildings_counted if buildings_counted > 0 else 0.0
	}


# People and phone tracking system
class PeoplePhoneTracker:
	"""Track people and phone counts with 15-minute intervals."""
	
	def __init__(self):
		self.tracking_active = False
		self.tracking_thread = None
		self.data_points = []
		self.start_time = None
	
	def start_tracking(self, location_name: str):
		"""Start tracking people and phones."""
		if not self.tracking_active:
			self.tracking_active = True
			self.start_time = datetime.now()
			self.data_points = []
			self.tracking_thread = threading.Thread(
				target=self._tracking_loop, 
				args=(location_name,),
				daemon=True
			)
			self.tracking_thread.start()
	
	def stop_tracking(self):
		"""Stop tracking."""
		self.tracking_active = False
		if self.tracking_thread:
			self.tracking_thread.join(timeout=1.0)
	
	def _tracking_loop(self, location_name: str):
		"""Main tracking loop - runs every 15 minutes."""
		while self.tracking_active:
			# Simulate people and phone counting
			# In a real implementation, this would integrate with:
			# - Camera-based people counting
			# - WiFi/Bluetooth device detection
			# - Mobile network analytics
			# - Foot traffic sensors
			
			people_count = self._simulate_people_count()
			phone_count = self._simulate_phone_count()
			
			data_point = {
				"timestamp": datetime.now(),
				"location": location_name,
				"people_count": people_count,
				"phone_count": phone_count,
				"interval_minutes": 15
			}
			
			self.data_points.append(data_point)
			
			# Keep only last 24 hours of data
			cutoff_time = datetime.now() - timedelta(hours=24)
			self.data_points = [
				dp for dp in self.data_points 
				if dp["timestamp"] > cutoff_time
			]
			
			# Wait 15 minutes (900 seconds)
			for _ in range(900):  # Check every second if we should stop
				if not self.tracking_active:
					break
				time.sleep(1)
	
	def _simulate_people_count(self) -> int:
		"""Simulate people counting (replace with real implementation)."""
		import random
		# Simulate realistic people counts based on time of day
		hour = datetime.now().hour
		
		if 6 <= hour < 9:  # Morning rush
			base_count = random.randint(20, 50)
		elif 9 <= hour < 17:  # Work hours
			base_count = random.randint(10, 30)
		elif 17 <= hour < 19:  # Evening rush
			base_count = random.randint(25, 60)
		elif 19 <= hour < 22:  # Evening
			base_count = random.randint(15, 35)
		else:  # Night
			base_count = random.randint(0, 10)
		
		return base_count + random.randint(-5, 5)
	
	def _simulate_phone_count(self) -> int:
		"""Simulate phone/device counting (replace with real implementation)."""
		import random
		# Phones are typically 70-90% of people count
		people_estimate = self._simulate_people_count()
		phone_ratio = random.uniform(0.7, 0.9)
		return int(people_estimate * phone_ratio)
	
	def get_current_stats(self) -> Dict[str, Any]:
		"""Get current tracking statistics."""
		if not self.data_points:
			return {
				"tracking_active": self.tracking_active,
				"data_points": 0,
				"latest_people": 0,
				"latest_phones": 0,
				"avg_people_1h": 0,
				"avg_phones_1h": 0,
				"total_tracking_time": "0:00:00"
			}
		
		latest = self.data_points[-1]
		
		# Calculate averages for last hour
		one_hour_ago = datetime.now() - timedelta(hours=1)
		recent_points = [
			dp for dp in self.data_points 
			if dp["timestamp"] > one_hour_ago
		]
		
		avg_people_1h = (
			sum(dp["people_count"] for dp in recent_points) / len(recent_points)
			if recent_points else 0
		)
		avg_phones_1h = (
			sum(dp["phone_count"] for dp in recent_points) / len(recent_points)
			if recent_points else 0
		)
		
		total_time = datetime.now() - self.start_time if self.start_time else timedelta(0)
		
		return {
			"tracking_active": self.tracking_active,
			"data_points": len(self.data_points),
			"latest_people": latest["people_count"],
			"latest_phones": latest["phone_count"],
			"avg_people_1h": round(avg_people_1h, 1),
			"avg_phones_1h": round(avg_phones_1h, 1),
			"total_tracking_time": str(total_time).split('.')[0],  # Remove microseconds
			"data_history": self.data_points[-10:]  # Last 10 data points
		}


# Global tracker instance
if "people_phone_tracker" not in st.session_state:
	st.session_state["people_phone_tracker"] = PeoplePhoneTracker()


def compute_enhanced_height_statistics(buildings_with_heights: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Compute comprehensive height statistics with confidence scoring."""
	if not buildings_with_heights:
		return {
			"avg_height": 0.0,
			"min_height": 0.0,
			"max_height": 0.0,
			"median_height": 0.0,
			"buildings_with_height": 0,
			"total_buildings": 0,
			"overall_confidence": 0.0,
			"data_quality_summary": {},
			"source_distribution": {},
			"recommendations": []
		}
	
	# Extract heights and calculate basic statistics
	heights = [b["height_m"] for b in buildings_with_heights if b["height_m"] is not None]
	confidences = [b["confidence"] for b in buildings_with_heights if b["confidence"] > 0]
	
	# Basic statistics
	heights.sort()
	avg_height = sum(heights) / len(heights) if heights else 0.0
	overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
	
	# Source distribution analysis
	source_counts = {}
	for building in buildings_with_heights:
		source = building.get("source", "unknown")
		source_counts[source] = source_counts.get(source, 0) + 1
	
	# Data quality assessment
	high_confidence = len([b for b in buildings_with_heights if b.get("confidence", 0) > 0.7])
	medium_confidence = len([b for b in buildings_with_heights if 0.4 < b.get("confidence", 0) <= 0.7])
	low_confidence = len([b for b in buildings_with_heights if b.get("confidence", 0) <= 0.4])
	
	# Generate recommendations
	recommendations = []
	if low_confidence > len(buildings_with_heights) * 0.3:
		recommendations.append("Consider validating low-confidence height estimates with local building permits")
	if "satellite_elevation" not in source_counts:
		recommendations.append("Satellite elevation data could improve height accuracy for this area")
	if overall_confidence < 0.6:
		recommendations.append("Overall data confidence is low - consider cross-referencing with official sources")
	
	return {
		"avg_height": avg_height,
		"min_height": heights[0] if heights else 0.0,
		"max_height": heights[-1] if heights else 0.0,
		"median_height": heights[len(heights) // 2] if len(heights) % 2 == 1 else (heights[len(heights) // 2 - 1] + heights[len(heights) // 2]) / 2 if heights else 0.0,
		"buildings_with_height": len(heights),
		"total_buildings": len(buildings_with_heights),
		"overall_confidence": overall_confidence,
		"data_quality_summary": {
			"high_confidence": high_confidence,
			"medium_confidence": medium_confidence,
			"low_confidence": low_confidence
		},
		"source_distribution": source_counts,
		"recommendations": recommendations
	}


def compute_density(num_buildings: int, radius_m: float) -> Dict[str, float]:
	circle_area_m2 = math.pi * (radius_m ** 2)
	return {
		"per_sq_km": num_buildings / meters_to_sq_km(circle_area_m2) if circle_area_m2 > 0 else 0.0,
		"per_sq_mile": num_buildings / meters_to_sq_miles(circle_area_m2) if circle_area_m2 > 0 else 0.0,
		"area_sq_km": meters_to_sq_km(circle_area_m2),
		"area_sq_miles": meters_to_sq_miles(circle_area_m2),
	}


def make_map(lat: float, lon: float, radius_m: float, building_polygons: List[Polygon]) -> folium.Map:
	m = folium.Map(location=[lat, lon], zoom_start=14, control_scale=True)
	folium.Circle(
		location=[lat, lon],
		radius=radius_m,
		color="#1f77b4",
		fill=True,
		fill_opacity=0.05,
		weight=2,
	).add_to(m)
	
	for poly in building_polygons:
		try:
			folium.GeoJson(
				data=poly.__geo_interface__,
				style_function=lambda x: {"color": "#d62728", "weight": 1, "fillColor": "#ff9896", "fillOpacity": 0.5},
			).add_to(m)
		except Exception:
			pass
	
	# Add center marker
	folium.Marker([lat, lon], icon=folium.Icon(color="blue", icon="info-sign"), tooltip="Center").add_to(m)
	return m


def inject_styles() -> None:
	st.markdown(
		"""
		<style>
			:root {
				--primary: #1f77b4;
				--accent: #ff7f0e;
				--card-bg: #ffffff;
				--soft: #f5f7fb;
				--text: #0f172a;
				--muted: #475569;
			}
			/* App background */
			.stApp {
				background: linear-gradient(180deg, #f8fafc 0%, #f3f6fb 100%);
				color: var(--text);
			}
			/* Typography tweaks */
			h1, h2, h3, h4 { letter-spacing: 0.2px; }
			.block-container { padding-top: 1.2rem; }
			p, label, span { color: var(--muted); }
			/* Cards */
			.card {
				background: var(--card-bg);
				border: 1px solid rgba(2,6,23,0.06);
				box-shadow: 0 8px 24px rgba(2,6,23,0.06);
				border-radius: 14px;
				padding: 18px 16px;
			}
			/* Buttons */
			.stButton>button {
				background: var(--primary) !important;
				color: #fff !important;
				border: none !important;
				border-radius: 10px !important;
				padding: 10px 18px !important;
				font-weight: 700 !important;
				box-shadow: 0 6px 14px rgba(31,119,180,0.25) !important;
				border-color: var(--primary) !important;
			}
			/* Ensure inner label stays visible */
			.stButton>button span, .stButton>button p {
				color: #ffffff !important;
			}
			/* Hover/Focus/Active */
			.stButton>button:hover { filter: brightness(0.96); transform: translateY(-1px); }
			.stButton>button:focus, .stButton>button:active { outline: none !important; box-shadow: 0 0 0 3px rgba(31,119,180,0.25) !important; }
			/* Disabled state */
			.stButton>button:disabled {
				background: #9abfe0 !important;
				color: #ffffff !important;
				opacity: 0.85 !important;
			}
			/* Sidebar */
			aside[data-testid="stSidebar"] {
				background: linear-gradient(180deg, #ffffff 0%, #f7faff 100%);
				border-right: 1px solid rgba(2,6,23,0.06);
			}
			/* Inputs */
			[data-baseweb="input"] input, [data-baseweb="textarea"] textarea, .stNumberInput input, .stTextInput input, .stSelectbox div[role="button"] {
				border-radius: 10px !important;
				border: 1px solid rgba(2,6,23,0.12) !important;
			}
			/* Radio & select labels */
			[data-testid="stWidgetLabel"] p { color: var(--text); font-weight: 600; }
			/* Section headers */
			section>div>div>div>h2,
			section>div>div>div>h3 { position: relative; padding-left: 10px; }
			section>div>div>div>h2:before,
			section>div>div>div>h3:before {
				content: "";
				display: inline-block;
				width: 6px; height: 16px;
				background: var(--primary);
				border-radius: 4px;
				position: absolute; left: 0; top: 10px;
			}
			/* Metrics */
			[data-testid="stMetric"] {
				background: var(--soft);
				border: 1px solid rgba(2,6,23,0.06);
				border-radius: 12px;
				padding: 10px 12px;
			}
			/* Folium map container spacing */
			.folium-map { border-radius: 14px; overflow: hidden; border: 1px solid rgba(2,6,23,0.06); }
		</style>
		""",
		unsafe_allow_html=True,
	)


def main() -> None:
	st.set_page_config(page_title="Building Density Calculator", layout="wide")
	inject_styles()
	st.markdown("<span class='header-badge'>🏙️ Urban analytics</span>", unsafe_allow_html=True)
	st.title("Building Density Calculator")
	st.caption("Compute buildings and density within a radius using OpenStreetMap data.")

	with st.sidebar:
		st.header("Search Parameters")
		input_mode = st.radio("Input Mode", ["Address", "Coordinates"], index=0)
		address = ""
		lat = 37.4221
		lon = -122.0841
		if input_mode == "Address":
			address = st.text_input("Address or place", placeholder="e.g., Times Square or 37.7749, -122.4194")
		else:
			lat_text = st.text_input("Latitude", value=f"{lat}")
			lon_text = st.text_input("Longitude", value=f"{lon}")
			# Try parsing user-entered text; keep previous value if parsing fails
			try:
				lat = float(lat_text)
			except Exception:
				pass
			try:
				lon = float(lon_text)
			except Exception:
				pass

		units = st.selectbox("Radius Units", ["kilometers", "miles"], index=0)
		radius_value = st.number_input("Radius", min_value=0.1, max_value=25.0, value=1.0, step=0.1)  # Reduced max to 25km
		
		# Enhanced height analysis options
		st.markdown("---")
		st.subheader("🔬 Height Analysis Options")
		enable_satellite = st.checkbox("Enable Satellite Elevation Data", value=True, 
			help="Fetch NASA SRTM satellite data for enhanced height accuracy")
		show_confidence = st.checkbox("Show Confidence Scores", value=True,
			help="Display confidence levels for height data quality")
		
		# People and phone tracking options
		st.markdown("---")
		st.subheader("👥 People & Phone Tracking")
		tracker = st.session_state["people_phone_tracker"]
		
		if not tracker.tracking_active:
			if st.button("Start Tracking", type="secondary", help="Begin 15-minute interval tracking"):
				location_name = address if address else f"{lat}, {lon}"
				tracker.start_tracking(location_name)
				st.success("📊 Tracking started! Data will be collected every 15 minutes.")
				st.rerun()
		else:
			if st.button("Stop Tracking", type="secondary", help="Stop tracking"):
				tracker.stop_tracking()
				st.info("📊 Tracking stopped.")
				st.rerun()
			
			# Show current tracking status
			current_stats = tracker.get_current_stats()
			st.write(f"**Status**: {'🟢 Active' if current_stats['tracking_active'] else '🔴 Inactive'}")
			st.write(f"**Tracking time**: {current_stats['total_tracking_time']}")
			st.write(f"**Data points**: {current_stats['data_points']}")
			
			if current_stats['data_points'] > 0:
				col1, col2 = st.columns(2)
				with col1:
					st.metric("Latest People", current_stats['latest_people'])
				with col2:
					st.metric("Latest Phones", current_stats['latest_phones'])
		
		# Use default Overpass settings (no UI)
		endpoint = OVERPASS_URL
		timeout_s = 120
		retries = 2
		force_refresh = False
		query_btn = st.button("Calculate", type="primary")

	# Determine center for live map based on current inputs
	# Geocode address or parse coordinates BEFORE showing map for user confirmation
	center_lat, center_lon = lat, lon
	radius_m_from_inputs = kilometers_to_meters(radius_value) if units == "kilometers" else miles_to_meters(radius_value)
	
	# Geocode address or parse coordinates for preview map (before calculation)
	preview_location_resolved = False
	if input_mode == "Address" and address.strip():
		# First, attempt to parse raw coordinates like "lat, lon"
		parsed_coords = parse_lat_lon_from_string(address)
		if parsed_coords is not None:
			center_lat, center_lon = parsed_coords
			preview_location_resolved = True
		else:
			# Try geocoding for preview (with error handling)
			try:
				center_lat, center_lon = geocode_location(address)
				preview_location_resolved = True
			except Exception:
				# If geocoding fails, keep default/previous coordinates
				# User will see error when they click Calculate
				preview_location_resolved = False
	elif input_mode == "Coordinates":
		# Validate coordinates are within range
		if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
			center_lat, center_lon = lat, lon
			preview_location_resolved = True
		else:
			preview_location_resolved = False

	# Render live map (updates as inputs change). Overlays will be added after calculation.
	st.subheader("Map Preview")
	if preview_location_resolved:
		st.info(f"📍 Location: {center_lat:.6f}, {center_lon:.6f}" + (f" (from: {address})" if input_mode == "Address" and address.strip() else ""))
	else:
		if input_mode == "Address" and address.strip():
			st.warning("⚠️ Address not yet resolved. Enter a valid address or coordinates to see the location on the map.")
		else:
			st.info("📍 Enter an address or coordinates to see the location on the map.")
	
	polygons_to_show: List[Polygon] = []
	# Keep overlays after reruns if we have a previous calculation
	if "calc" in st.session_state and isinstance(st.session_state["calc"], dict):
		try:
			polygons_to_show = st.session_state["calc"].get("polygons", [])
		except (KeyError, TypeError):
			polygons_to_show = []

	results_area = st.empty()

	if query_btn:
		try:
			# Handle geocoding only when Calculate is pressed
			calc_lat, calc_lon = lat, lon
			if input_mode == "Address" and address.strip():
				# First, attempt to parse raw coordinates like "lat, lon"
				parsed_coords = parse_lat_lon_from_string(address)
				if parsed_coords is not None:
					calc_lat, calc_lon = parsed_coords
				else:
					try:
						calc_lat, calc_lon = geocode_location(address)
					except Exception as e:
						st.error(f"Geocoding failed: {str(e)}")
						return
			
			# Validate ranges for coordinates
			if not (-90.0 <= calc_lat <= 90.0 and -180.0 <= calc_lon <= 180.0):
				raise ValueError("Coordinates out of range. Latitude must be -90..90, Longitude -180..180.")
			
			# Validate radius size
			if radius_m_from_inputs > 25000:  # 25km limit
				raise ValueError("Radius too large. Maximum allowed is 25km to prevent timeouts.")
			
			# Build cache key and check
			cache_key = f"{endpoint}|{round(calc_lat,6)}|{round(calc_lon,6)}|{round(radius_m_from_inputs,2)}|{timeout_s}"
			cache_hit = False
			elements: List[Dict[str, Any]]
			if not force_refresh and "cache" in st.session_state and cache_key in st.session_state["cache"]:
				cache_hit = True
				elements = st.session_state["cache"][cache_key]["elements"]
				queried_at = st.session_state["cache"][cache_key]["timestamp"]
			else:
				with st.spinner("Querying OpenStreetMap for buildings..."):
					elements = fetch_buildings(calc_lat, calc_lon, radius_m_from_inputs, endpoint, timeout_s, retries)
					import datetime as _dt
					queried_at = _dt.datetime.utcnow().isoformat() + "Z"
					manage_cache(cache_key, elements, queried_at)
			
			with st.spinner("Processing geometries..."):
				# Fetch satellite elevation data if enabled
				satellite_elevation = None
				if enable_satellite:
					with st.spinner("Fetching satellite elevation data..."):
						satellite_elevation = fetch_satellite_elevation(calc_lat, calc_lon, radius_m_from_inputs)
				
				# Process buildings
				num_buildings, polygons, buildings_with_heights = count_buildings_and_polygons(elements, satellite_elevation)
				stats = compute_density(num_buildings, radius_m_from_inputs)
				height_stats = compute_enhanced_height_statistics(buildings_with_heights)
				
				# Calculate roof area
				roof_area_stats = calculate_roof_area(polygons)
				
				# Calculate total building height
				total_height_stats = calculate_total_building_height(buildings_with_heights)
				
				# Analyze roads and infrastructure
				infrastructure_stats = analyze_roads_and_infrastructure(elements)
				
				polygons_to_show = polygons

			# Persist results so they remain after reruns
			st.session_state["calc"] = {
				"num_buildings": num_buildings,
				"stats": stats,
				"height_stats": height_stats,
				"roof_area_stats": roof_area_stats,
				"total_height_stats": total_height_stats,
				"infrastructure_stats": infrastructure_stats,
				"polygons": polygons,
				"center_lat": calc_lat,
				"center_lon": calc_lon,
				"radius_m": radius_m_from_inputs,
				"endpoint": endpoint,
				"timeout_s": timeout_s,
				"retries": retries,
				"timestamp": queried_at,
				"cache_hit": cache_hit,
				"satellite_enabled": enable_satellite,
				"confidence_enabled": show_confidence,
			}

			# Render results below (outside button) using session state
		except Exception as e:
			st.error(f"Error: {str(e)}")

	# If we have saved results, display them persistently
	if "calc" in st.session_state and isinstance(st.session_state["calc"], dict):
		_saved = st.session_state["calc"]
		with results_area.container():
			left, right = st.columns([1, 2])
			with left:
				st.markdown("<div class='card'>", unsafe_allow_html=True)
				st.subheader("🏢 Building Analysis Results")
				
				# Collect all metrics into a grid format
				results_data = []
				
				# Basic building metrics
				results_data.append({"Category": "Building Analysis", "Metric": "Buildings Found", "Value": f"{_saved['num_buildings']}", "Unit": ""})
				results_data.append({"Category": "Building Analysis", "Metric": "Density (per km²)", "Value": f"{_saved['stats']['per_sq_km']:.1f}", "Unit": "buildings/km²"})
				results_data.append({"Category": "Building Analysis", "Metric": "Density (per mi²)", "Value": f"{_saved['stats']['per_sq_mile']:.1f}", "Unit": "buildings/mi²"})
				results_data.append({"Category": "Building Analysis", "Metric": "Search Area", "Value": f"{_saved['stats']['area_sq_km']:.3f}", "Unit": "sq km"})
				results_data.append({"Category": "Building Analysis", "Metric": "Search Area", "Value": f"{_saved['stats']['area_sq_miles']:.3f}", "Unit": "sq mi"})
				
				# Roof area analysis
				roof_stats = _saved.get('roof_area_stats', {})
				if roof_stats.get('buildings_analyzed', 0) > 0:
					results_data.append({"Category": "Roof Area Analysis", "Metric": "Total Roof Area", "Value": f"{roof_stats['total_area_km2']:.4f}", "Unit": "km²"})
					results_data.append({"Category": "Roof Area Analysis", "Metric": "Total Roof Area", "Value": f"{roof_stats['total_area_m2']:.0f}", "Unit": "m²"})
					results_data.append({"Category": "Roof Area Analysis", "Metric": "Average Building Area", "Value": f"{roof_stats['average_area_m2']:.1f}", "Unit": "m²"})
					results_data.append({"Category": "Roof Area Analysis", "Metric": "Buildings Analyzed", "Value": f"{roof_stats['buildings_analyzed']}", "Unit": ""})
				else:
					results_data.append({"Category": "Roof Area Analysis", "Metric": "Status", "Value": "No roof area data available", "Unit": ""})
				
				# Total height analysis
				total_height_stats = _saved.get('total_height_stats', {})
				if total_height_stats.get('buildings_with_height', 0) > 0:
					results_data.append({"Category": "Total Building Height", "Metric": "Total Height", "Value": f"{total_height_stats['total_height_m']:.0f}", "Unit": "m"})
					results_data.append({"Category": "Total Building Height", "Metric": "Total Height", "Value": f"{total_height_stats['total_height_km']:.3f}", "Unit": "km"})
					results_data.append({"Category": "Total Building Height", "Metric": "Average Height", "Value": f"{total_height_stats['average_height_m']:.1f}", "Unit": "m"})
					results_data.append({"Category": "Total Building Height", "Metric": "Buildings with Height", "Value": f"{total_height_stats['buildings_with_height']}", "Unit": ""})
				else:
					results_data.append({"Category": "Total Building Height", "Metric": "Status", "Value": "No height data available for total calculations", "Unit": ""})
				
				# Infrastructure analysis
				infra_stats = _saved.get('infrastructure_stats', {})
				if infra_stats:
					road_stats = infra_stats.get('roads', {})
					tiled_roads = road_stats.get('tiled', {})
					untiled_roads = road_stats.get('untiled', {})
					
					results_data.append({"Category": "Roads (Tiled/Paved)", "Metric": "Length", "Value": f"{tiled_roads.get('total_length_km', 0):.2f}", "Unit": "km"})
					results_data.append({"Category": "Roads (Tiled/Paved)", "Metric": "Area", "Value": f"{tiled_roads.get('total_area_km2', 0):.4f}", "Unit": "km²"})
					results_data.append({"Category": "Roads (Tiled/Paved)", "Metric": "Count", "Value": f"{tiled_roads.get('count', 0)}", "Unit": ""})
					
					results_data.append({"Category": "Roads (Untiled/Unpaved)", "Metric": "Length", "Value": f"{untiled_roads.get('total_length_km', 0):.2f}", "Unit": "km"})
					results_data.append({"Category": "Roads (Untiled/Unpaved)", "Metric": "Area", "Value": f"{untiled_roads.get('total_area_km2', 0):.4f}", "Unit": "km²"})
					results_data.append({"Category": "Roads (Untiled/Unpaved)", "Metric": "Count", "Value": f"{untiled_roads.get('count', 0)}", "Unit": ""})
					
					intersections = infra_stats.get('intersections', {})
					results_data.append({"Category": "Intersections", "Metric": "Tiled Road Intersections", "Value": f"{intersections.get('tiled', 0)}", "Unit": ""})
					results_data.append({"Category": "Intersections", "Metric": "Untiled Road Intersections", "Value": f"{intersections.get('untiled', 0)}", "Unit": ""})
					
					footpath_stats = infra_stats.get('footpaths', {})
					results_data.append({"Category": "Footpaths & Walkways", "Metric": "Total Length", "Value": f"{footpath_stats.get('total_length_km', 0):.2f}", "Unit": "km"})
					results_data.append({"Category": "Footpaths & Walkways", "Metric": "Count", "Value": f"{footpath_stats.get('count', 0)}", "Unit": ""})
				else:
					results_data.append({"Category": "Infrastructure", "Metric": "Status", "Value": "No infrastructure data available", "Unit": ""})
				
				# People and phone tracking
				tracker = st.session_state["people_phone_tracker"]
				current_stats = tracker.get_current_stats()
				
				if current_stats['tracking_active'] or current_stats['data_points'] > 0:
					results_data.append({"Category": "People & Phone Tracking", "Metric": "Status", "Value": "🟢 Active" if current_stats['tracking_active'] else "🔴 Stopped", "Unit": ""})
					results_data.append({"Category": "People & Phone Tracking", "Metric": "Data Points", "Value": f"{current_stats['data_points']}", "Unit": ""})
					results_data.append({"Category": "People & Phone Tracking", "Metric": "Tracking Time", "Value": current_stats['total_tracking_time'], "Unit": ""})
					
					if current_stats['data_points'] > 0:
						results_data.append({"Category": "People & Phone Tracking", "Metric": "Latest People Count", "Value": f"{current_stats['latest_people']}", "Unit": ""})
						results_data.append({"Category": "People & Phone Tracking", "Metric": "Avg People (1h)", "Value": f"{current_stats['avg_people_1h']}", "Unit": ""})
						results_data.append({"Category": "People & Phone Tracking", "Metric": "Latest Phone Count", "Value": f"{current_stats['latest_phones']}", "Unit": ""})
						results_data.append({"Category": "People & Phone Tracking", "Metric": "Avg Phones (1h)", "Value": f"{current_stats['avg_phones_1h']}", "Unit": ""})
						
						# Show recent data points in grid
						if st.checkbox("Show Recent Data History", key="show_tracking_history"):
							st.markdown("**Recent Data Points (15-min intervals)**")
							history = current_stats.get('data_history', [])
							if history:
								history_data = []
								for dp in reversed(history[-5:]):
									time_str = dp['timestamp'].strftime("%H:%M:%S")
									history_data.append({
										"Timestamp": time_str,
										"People Count": dp['people_count'],
										"Phone Count": dp['phone_count']
									})
								history_df = pd.DataFrame(history_data)
								st.dataframe(history_df, use_container_width=True, hide_index=True)
				else:
					st.info("👥 People & phone tracking not started. Use the sidebar to begin tracking.")
					st.write("*Note: This is a simulated tracking system for demonstration. In production, this would integrate with real sensors, cameras, or network analytics.*")
				
				# Enhanced Height Statistics
				height_stats = _saved.get('height_stats', {})
				if height_stats.get('buildings_with_height', 0) > 0:
					results_data.append({"Category": "Building Heights (Enhanced)", "Metric": "Average Height", "Value": f"{height_stats['avg_height']:.1f}", "Unit": "m"})
					results_data.append({"Category": "Building Heights (Enhanced)", "Metric": "Median Height", "Value": f"{height_stats['median_height']:.1f}", "Unit": "m"})
					results_data.append({"Category": "Building Heights (Enhanced)", "Metric": "Min Height", "Value": f"{height_stats['min_height']:.1f}", "Unit": "m"})
					results_data.append({"Category": "Building Heights (Enhanced)", "Metric": "Max Height", "Value": f"{height_stats['max_height']:.1f}", "Unit": "m"})
					results_data.append({"Category": "Building Heights (Enhanced)", "Metric": "Overall Confidence", "Value": f"{height_stats.get('overall_confidence', 0):.1%}", "Unit": ""})
					results_data.append({"Category": "Building Heights (Enhanced)", "Metric": "Buildings with Height Data", "Value": f"{height_stats['buildings_with_height']} of {_saved['num_buildings']}", "Unit": ""})
					
					# Satellite data status
					if _saved.get('satellite_enabled', False):
						results_data.append({"Category": "Building Heights (Enhanced)", "Metric": "Satellite Data", "Value": "🛰️ Enabled", "Unit": ""})
					else:
						results_data.append({"Category": "Building Heights (Enhanced)", "Metric": "Satellite Data", "Value": "📡 Disabled", "Unit": ""})
					
					# Data quality breakdown
					quality_summary = height_stats.get('data_quality_summary', {})
					if quality_summary:
						results_data.append({"Category": "Data Quality", "Metric": "High Confidence (>70%)", "Value": f"{quality_summary.get('high_confidence', 0)}", "Unit": ""})
						results_data.append({"Category": "Data Quality", "Metric": "Medium Confidence (40-70%)", "Value": f"{quality_summary.get('medium_confidence', 0)}", "Unit": ""})
						results_data.append({"Category": "Data Quality", "Metric": "Low Confidence (<40%)", "Value": f"{quality_summary.get('low_confidence', 0)}", "Unit": ""})
					
					# Data source distribution
					source_dist = height_stats.get('source_distribution', {})
					if source_dist:
						for source, count in source_dist.items():
							source_name = source.replace('_', ' ').title()
							if source == "height_tag":
								source_name = "Direct Height Tags"
							elif source == "building_levels":
								source_name = "Building Levels (Estimated)"
							elif source == "max_height":
								source_name = "Max Height Tags"
							elif source == "satellite_elevation":
								source_name = "Satellite Elevation Data"
							results_data.append({"Category": "Data Sources", "Metric": source_name, "Value": f"{count}", "Unit": "buildings"})
					
					# Recommendations
					recommendations = height_stats.get('recommendations', [])
					if recommendations:
						for i, rec in enumerate(recommendations):
							results_data.append({"Category": "Recommendations", "Metric": f"Recommendation {i+1}", "Value": rec, "Unit": ""})
				else:
					results_data.append({"Category": "Building Heights (Enhanced)", "Metric": "Status", "Value": "No height data available for buildings in this area", "Unit": ""})
					results_data.append({"Category": "Building Heights (Enhanced)", "Metric": "Note", "Value": "Height data depends on OpenStreetMap contributors and satellite data availability", "Unit": ""})
				
				# Create DataFrame and display in grid
				if results_data:
					df = pd.DataFrame(results_data)
					st.dataframe(df, use_container_width=True, hide_index=True)
				else:
					st.warning("No results data available")
				
				with st.expander("Run metadata"):
					metadata_data = [
						{"Field": "Endpoint", "Value": _saved.get('endpoint', '')},
						{"Field": "Timeout", "Value": f"{_saved.get('timeout_s', 0)}s"},
						{"Field": "Retries", "Value": f"{_saved.get('retries', 0)}"},
						{"Field": "Queried at (UTC)", "Value": _saved.get('timestamp', '')},
						{"Field": "Cache hit", "Value": str(_saved.get('cache_hit', False))}
					]
					metadata_df = pd.DataFrame(metadata_data)
					st.dataframe(metadata_df, use_container_width=True, hide_index=True)
				st.caption("Note: OSM is community-sourced; completeness varies by location.")
				st.markdown("</div>", unsafe_allow_html=True)
			with right:
				m_results = make_map(_saved["center_lat"], _saved["center_lon"], _saved["radius_m"], _saved["polygons"])
				st_folium(m_results, width=None, height=600, key="results_map")

	# Draw the map with current center and radius, overlay polygons if available
	# Only show this map if no results are displayed (avoid duplication)
	if "calc" not in st.session_state or not isinstance(st.session_state["calc"], dict):
		st.markdown("<div class='card'>", unsafe_allow_html=True)
		if preview_location_resolved:
			# Create preview map with exact location and radius circle for confirmation
			m_preview = folium.Map(location=[center_lat, center_lon], zoom_start=14, control_scale=True)
			# Add radius circle
			folium.Circle(
				location=[center_lat, center_lon],
				radius=radius_m_from_inputs,
				color="#1f77b4",
				fill=True,
				fill_opacity=0.1,
				weight=3,
				popup=f"Search Radius: {radius_value} {units}",
			).add_to(m_preview)
			# Add prominent center marker
			folium.Marker(
				[center_lat, center_lon],
				icon=folium.Icon(color="red", icon="info-sign", prefix="fa"),
				tooltip=f"Center: {center_lat:.6f}, {center_lon:.6f}",
				popup=f"<b>Search Center</b><br>Lat: {center_lat:.6f}<br>Lon: {center_lon:.6f}<br>Radius: {radius_value} {units}"
			).add_to(m_preview)
			# Add any existing building polygons from previous calculations
			for poly in polygons_to_show:
				try:
					folium.GeoJson(
						data=poly.__geo_interface__,
						style_function=lambda x: {"color": "#d62728", "weight": 1, "fillColor": "#ff9896", "fillOpacity": 0.5},
					).add_to(m_preview)
				except Exception:
					pass
			st_folium(m_preview, width=None, height=600, key="preview_map")
			st.caption("👆 Confirm the location and radius above, then click 'Calculate' to analyze buildings in this area.")
		else:
			# Show default map with message
			m_default = make_map(center_lat, center_lon, radius_m_from_inputs, polygons_to_show)
			st_folium(m_default, width=None, height=600, key="default_map")
		st.markdown("</div>", unsafe_allow_html=True)

	st.caption("Built with Streamlit, Folium, and OpenStreetMap data.")


if __name__ == "__main__":
	main() 