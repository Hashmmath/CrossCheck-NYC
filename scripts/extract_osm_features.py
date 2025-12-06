#!/usr/bin/env python3
"""
CrossCheck NYC - OSM Feature Extraction for Failure Analysis
=============================================================
Extracts OSM features that may impact tile2net crosswalk detection:

1. Shadow Casters: building:levels, building:height
2. Road Context: highway type (primary, residential, etc.)
3. Visual Pattern: crossing:markings (zebra, lines, surface)
4. Surface Type: surface (asphalt, concrete, paving_stones)
5. Canopy Occlusion: natural=tree nodes near crosswalks

Output: GeoJSON files with features for each location
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("./data")
REFERENCE_DIR = DATA_DIR / "reference"
FEATURES_DIR = DATA_DIR / "features"
OUTPUT_DIR = DATA_DIR / "outputs"

# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Locations (NO WYANDANCH)
LOCATIONS = {
    "financial_district": {
        "bbox": (40.7025, -74.0125, 40.7075, -74.0065),  # (south, west, north, east)
        "name": "Financial District, Manhattan"
    },
    "east_village": {
        "bbox": (40.7235, -73.9900, 40.7295, -73.9830),
        "name": "East Village, Manhattan"
    },
    "bay_ridge": {
        "bbox": (40.6290, -74.0300, 40.6350, -74.0230),
        "name": "Bay Ridge, Brooklyn"
    },
    "downtown_brooklyn": {
        "bbox": (40.6880, -73.9900, 40.6950, -73.9800),
        "name": "Downtown Brooklyn"
    },
    "kew_gardens": {
        "bbox": (40.7050, -73.8350, 40.7150, -73.8200),
        "name": "Kew Gardens, Queens"
    }
}

# OSM Features to extract (based on your analysis framework)
OSM_FEATURES = {
    "shadow_casters": {
        "description": "Buildings that cast shadows on crosswalks",
        "tags": ["building"],
        "attributes": ["building:levels", "building:height", "height"]
    },
    "road_context": {
        "description": "Road type affects marking visibility and wear",
        "tags": ["highway"],
        "values": ["primary", "secondary", "tertiary", "residential", "unclassified"]
    },
    "crossing_markings": {
        "description": "Visual pattern of crosswalk markings",
        "tags": ["highway=crossing"],
        "attributes": ["crossing:markings", "crossing", "tactile_paving"]
    },
    "surface_type": {
        "description": "Road surface affects contrast with markings",
        "tags": ["highway"],
        "attributes": ["surface"]
    },
    "tree_canopy": {
        "description": "Trees that may occlude crosswalks from aerial view",
        "tags": ["natural=tree"],
        "attributes": ["height", "diameter_crown"]
    }
}


# =============================================================================
# OVERPASS QUERY BUILDERS
# =============================================================================

def build_overpass_query(bbox: Tuple[float, float, float, float], feature_type: str) -> str:
    """Build Overpass QL query for specific feature type."""
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"
    
    if feature_type == "shadow_casters":
        # Buildings with height/levels info
        query = f"""
        [out:json][timeout:60];
        (
          way["building"]({bbox_str});
          relation["building"]({bbox_str});
        );
        out body;
        >;
        out skel qt;
        """
    
    elif feature_type == "road_context":
        # Roads/highways
        query = f"""
        [out:json][timeout:60];
        (
          way["highway"~"primary|secondary|tertiary|residential|unclassified|footway|pedestrian"]({bbox_str});
        );
        out body;
        >;
        out skel qt;
        """
    
    elif feature_type == "crossing_markings":
        # Crossings with marking info
        query = f"""
        [out:json][timeout:60];
        (
          node["highway"="crossing"]({bbox_str});
          way["highway"="crossing"]({bbox_str});
          node["crossing"]({bbox_str});
        );
        out body;
        >;
        out skel qt;
        """
    
    elif feature_type == "surface_type":
        # Roads with surface info
        query = f"""
        [out:json][timeout:60];
        (
          way["highway"]["surface"]({bbox_str});
        );
        out body;
        >;
        out skel qt;
        """
    
    elif feature_type == "tree_canopy":
        # Trees
        query = f"""
        [out:json][timeout:60];
        (
          node["natural"="tree"]({bbox_str});
          way["natural"="tree_row"]({bbox_str});
          way["landuse"="forest"]({bbox_str});
          relation["landuse"="forest"]({bbox_str});
        );
        out body;
        >;
        out skel qt;
        """
    
    else:
        query = ""
    
    return query


def query_overpass(query: str, retries: int = 3) -> Optional[Dict]:
    """Execute Overpass API query with retries."""
    for attempt in range(retries):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=120
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"Overpass returned {response.status_code}")
        except Exception as e:
            logger.warning(f"Overpass query failed: {e}")
            time.sleep(10)
    
    return None


# =============================================================================
# DATA PROCESSING
# =============================================================================

def parse_osm_elements(data: Dict, feature_type: str) -> List[Dict]:
    """Parse OSM elements into feature records."""
    if not data or "elements" not in data:
        return []
    
    features = []
    nodes = {}
    
    # First pass: collect all nodes
    for elem in data["elements"]:
        if elem["type"] == "node":
            nodes[elem["id"]] = (elem.get("lon"), elem.get("lat"))
    
    # Second pass: process elements
    for elem in data["elements"]:
        feature = {
            "osm_id": elem["id"],
            "osm_type": elem["type"],
            "feature_category": feature_type
        }
        
        # Get tags
        tags = elem.get("tags", {})
        feature["tags"] = tags
        
        # Extract specific attributes based on feature type
        if feature_type == "shadow_casters":
            feature["building_type"] = tags.get("building", "yes")
            feature["levels"] = tags.get("building:levels")
            feature["height"] = tags.get("height") or tags.get("building:height")
            # Estimate height from levels if not provided
            if feature["levels"] and not feature["height"]:
                try:
                    feature["estimated_height_m"] = int(feature["levels"]) * 3.5
                except:
                    pass
        
        elif feature_type == "road_context":
            feature["highway_type"] = tags.get("highway")
            feature["name"] = tags.get("name")
            feature["lanes"] = tags.get("lanes")
            feature["surface"] = tags.get("surface")
        
        elif feature_type == "crossing_markings":
            feature["crossing_type"] = tags.get("crossing")
            feature["markings"] = tags.get("crossing:markings")
            feature["tactile_paving"] = tags.get("tactile_paving")
            feature["traffic_signals"] = tags.get("traffic_signals")
        
        elif feature_type == "surface_type":
            feature["surface"] = tags.get("surface")
            feature["highway_type"] = tags.get("highway")
            # Classify contrast
            surface = tags.get("surface", "").lower()
            if surface in ["asphalt"]:
                feature["contrast_level"] = "high"
            elif surface in ["concrete", "paving_stones", "sett"]:
                feature["contrast_level"] = "medium"
            else:
                feature["contrast_level"] = "unknown"
        
        elif feature_type == "tree_canopy":
            feature["tree_type"] = tags.get("leaf_type", "unknown")
            feature["height"] = tags.get("height")
            feature["crown_diameter"] = tags.get("diameter_crown")
        
        # Get geometry
        if elem["type"] == "node":
            if "lon" in elem and "lat" in elem:
                feature["geometry"] = Point(elem["lon"], elem["lat"])
                features.append(feature)
        
        elif elem["type"] == "way":
            if "nodes" in elem:
                coords = [nodes.get(n) for n in elem["nodes"] if n in nodes]
                coords = [c for c in coords if c is not None]
                if len(coords) >= 2:
                    from shapely.geometry import LineString, Polygon
                    try:
                        if coords[0] == coords[-1] and len(coords) >= 4:
                            feature["geometry"] = Polygon(coords)
                        else:
                            feature["geometry"] = LineString(coords)
                        features.append(feature)
                    except:
                        pass
    
    return features


def features_to_geodataframe(features: List[Dict]) -> Optional[gpd.GeoDataFrame]:
    """Convert feature list to GeoDataFrame."""
    if not features:
        return None
    
    # Filter features with valid geometry
    valid_features = [f for f in features if "geometry" in f and f["geometry"] is not None]
    
    if not valid_features:
        return None
    
    gdf = gpd.GeoDataFrame(valid_features, crs="EPSG:4326")
    
    # Remove tags column (it's a dict and causes issues)
    if "tags" in gdf.columns:
        gdf = gdf.drop(columns=["tags"])
    
    return gdf


# =============================================================================
# IMPACT ANALYSIS
# =============================================================================

def analyze_feature_impact_on_crosswalks(
    crosswalks_gdf: gpd.GeoDataFrame,
    features_gdf: gpd.GeoDataFrame,
    feature_type: str,
    buffer_meters: float = 15
) -> pd.DataFrame:
    """
    Analyze how features might impact crosswalk detection.
    Returns a DataFrame with crosswalk IDs and nearby feature metrics.
    """
    if crosswalks_gdf is None or len(crosswalks_gdf) == 0:
        return pd.DataFrame()
    
    if features_gdf is None or len(features_gdf) == 0:
        return pd.DataFrame()
    
    # Project to UTM for accurate distance
    try:
        cw_proj = crosswalks_gdf.to_crs(epsg=32618)
        feat_proj = features_gdf.to_crs(epsg=32618)
    except:
        cw_proj = crosswalks_gdf
        feat_proj = features_gdf
        buffer_meters = buffer_meters / 111000  # Approximate degrees
    
    results = []
    
    for idx, cw in cw_proj.iterrows():
        cw_centroid = cw.geometry.centroid
        buffer = cw_centroid.buffer(buffer_meters)
        
        # Find nearby features
        nearby = feat_proj[feat_proj.geometry.intersects(buffer)]
        
        record = {
            "crosswalk_idx": idx,
            "feature_type": feature_type,
            "nearby_count": len(nearby)
        }
        
        if feature_type == "shadow_casters" and len(nearby) > 0:
            heights = []
            for _, feat in nearby.iterrows():
                h = feat.get("estimated_height_m") or feat.get("height")
                if h:
                    try:
                        heights.append(float(str(h).replace("m", "").strip()))
                    except:
                        pass
            record["max_building_height"] = max(heights) if heights else 0
            record["avg_building_height"] = sum(heights) / len(heights) if heights else 0
            record["shadow_risk"] = "high" if record["max_building_height"] > 30 else "medium" if record["max_building_height"] > 15 else "low"
        
        elif feature_type == "tree_canopy" and len(nearby) > 0:
            record["tree_count"] = len(nearby)
            record["occlusion_risk"] = "high" if len(nearby) > 3 else "medium" if len(nearby) > 0 else "low"
        
        elif feature_type == "surface_type" and len(nearby) > 0:
            surfaces = nearby["surface"].dropna().tolist() if "surface" in nearby.columns else []
            record["surfaces"] = list(set(surfaces))
            # Check contrast
            low_contrast = ["concrete", "paving_stones", "sett", "cobblestone"]
            record["low_contrast_surface"] = any(s in low_contrast for s in surfaces)
        
        elif feature_type == "road_context" and len(nearby) > 0:
            types = nearby["highway_type"].dropna().tolist() if "highway_type" in nearby.columns else []
            record["road_types"] = list(set(types))
            record["is_residential"] = "residential" in types
        
        results.append(record)
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_features_for_location(location_id: str) -> Dict[str, gpd.GeoDataFrame]:
    """Extract all OSM features for a single location."""
    
    if location_id not in LOCATIONS:
        logger.error(f"Unknown location: {location_id}")
        return {}
    
    loc_info = LOCATIONS[location_id]
    bbox = loc_info["bbox"]
    
    logger.info(f"Extracting features for {loc_info['name']}")
    
    all_features = {}
    
    for feature_type in OSM_FEATURES.keys():
        logger.info(f"  Fetching {feature_type}...")
        
        query = build_overpass_query(bbox, feature_type)
        data = query_overpass(query)
        
        if data:
            features = parse_osm_elements(data, feature_type)
            gdf = features_to_geodataframe(features)
            
            if gdf is not None and len(gdf) > 0:
                all_features[feature_type] = gdf
                logger.info(f"    Found {len(gdf)} {feature_type} features")
            else:
                logger.info(f"    No {feature_type} features found")
        else:
            logger.warning(f"    Failed to fetch {feature_type}")
        
        # Rate limiting
        time.sleep(2)
    
    return all_features


def save_features(location_id: str, features: Dict[str, gpd.GeoDataFrame]):
    """Save extracted features to files."""
    
    loc_dir = FEATURES_DIR / location_id
    loc_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for feature_type, gdf in features.items():
        if gdf is not None and len(gdf) > 0:
            filepath = loc_dir / f"{feature_type}.geojson"
            gdf.to_file(filepath, driver="GeoJSON")
            saved_files.append(filepath)
            logger.info(f"  Saved {filepath}")
    
    # Also save combined summary
    summary = {
        "location_id": location_id,
        "location_name": LOCATIONS[location_id]["name"],
        "bbox": LOCATIONS[location_id]["bbox"],
        "feature_counts": {ft: len(gdf) for ft, gdf in features.items()}
    }
    
    with open(loc_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return saved_files


def load_features(location_id: str) -> Dict[str, gpd.GeoDataFrame]:
    """Load previously saved features."""
    
    loc_dir = FEATURES_DIR / location_id
    features = {}
    
    if not loc_dir.exists():
        return features
    
    for feature_type in OSM_FEATURES.keys():
        filepath = loc_dir / f"{feature_type}.geojson"
        if filepath.exists():
            try:
                gdf = gpd.read_file(filepath)
                features[feature_type] = gdf
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
    
    return features


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_feature_extraction(locations: List[str] = None):
    """Run feature extraction for all or specified locations."""
    
    print("\n" + "=" * 70)
    print("   CROSSCHECK NYC - OSM FEATURE EXTRACTION")
    print("   For Failure Analysis Visualization")
    print("=" * 70 + "\n")
    
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if locations is None:
        locations = list(LOCATIONS.keys())
    
    all_summaries = {}
    
    for loc_id in locations:
        print(f"\n{'='*50}")
        print(f"Location: {loc_id}")
        print('='*50)
        
        # Extract features
        features = extract_features_for_location(loc_id)
        
        if features:
            # Save features
            saved = save_features(loc_id, features)
            print(f"  Saved {len(saved)} feature files")
            
            # Summary
            all_summaries[loc_id] = {
                "features": {ft: len(gdf) for ft, gdf in features.items()},
                "files": [str(f) for f in saved]
            }
        else:
            print(f"  No features extracted")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Location':<25} {'Buildings':>10} {'Roads':>10} {'Crossings':>10} {'Trees':>10}")
    print("-" * 70)
    
    for loc_id, summary in all_summaries.items():
        f = summary.get("features", {})
        print(f"{loc_id:<25} "
              f"{f.get('shadow_casters', 0):>10} "
              f"{f.get('road_context', 0):>10} "
              f"{f.get('crossing_markings', 0):>10} "
              f"{f.get('tree_canopy', 0):>10}")
    
    # Save master summary
    with open(FEATURES_DIR / "extraction_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\n✓ All features saved to: {FEATURES_DIR}/")
    
    return all_summaries


if __name__ == "__main__":
    run_feature_extraction()