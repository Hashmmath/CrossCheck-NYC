#!/usr/bin/env python3
"""
CrossCheck NYC - Feature Impact Analysis
=========================================
Calculates how each OSM feature impacts tile2net detection metrics.

For each ground truth crosswalk:
1. Check if detected (TP) or missed (FN)
2. Find nearby OSM features (buildings, trees, surface, etc.)
3. Group crosswalks by feature conditions
4. Calculate Precision/Recall/F1 per group

Output: JSON with per-feature metrics for visualization
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path("./data")
OUTPUT_DIR = DATA_DIR / "outputs"
REFERENCE_DIR = DATA_DIR / "reference"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = DATA_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"

LOCATIONS = ["financial_district", "east_village", "bay_ridge", "downtown_brooklyn", "kew_gardens"]

# Buffer distances (meters)
BUILDING_BUFFER = 20  # Check buildings within 20m
TREE_BUFFER = 8       # Check trees within 8m
ROAD_BUFFER = 10      # Check road surface within 10m

# Feature thresholds for grouping
BUILDING_HEIGHT_THRESHOLDS = [15, 30]  # Low: <15m, Medium: 15-30m, High: >30m
TREE_COUNT_THRESHOLDS = [1, 3]         # None: 0, Few: 1-2, Many: ≥3


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ground_truth() -> Optional[gpd.GeoDataFrame]:
    """Load OSM ground truth crossings."""
    for f in [REFERENCE_DIR / "combined_reference.geojson", 
              REFERENCE_DIR / "osm_crossings_nyc.geojson"]:
        if f.exists():
            try:
                return gpd.read_file(f)
            except:
                pass
    return None


def load_predictions(location_id: str) -> Optional[gpd.GeoDataFrame]:
    """Load tile2net polygon predictions."""
    poly_dir = OUTPUT_DIR / location_id / "polygons"
    if not poly_dir.exists():
        return None
    
    for shp in poly_dir.rglob("*.shp"):
        try:
            gdf = gpd.read_file(shp)
            # Filter to crosswalks
            for col in ['f_type', 'class', 'Class', 'type']:
                if col in gdf.columns:
                    mask = gdf[col].astype(str).str.lower().str.contains('crosswalk|crossing', na=False)
                    return gdf[mask].copy()
            return gdf
        except:
            pass
    return None


def load_osm_features(location_id: str) -> Dict[str, gpd.GeoDataFrame]:
    """Load extracted OSM features."""
    loc_dir = FEATURES_DIR / location_id
    features = {}
    
    feature_types = ["shadow_casters", "tree_canopy", "surface_type", "road_context", "crossing_markings"]
    
    for ft in feature_types:
        filepath = loc_dir / f"{ft}.geojson"
        if filepath.exists():
            try:
                features[ft] = gpd.read_file(filepath)
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
    
    return features


# =============================================================================
# DETECTION MATCHING
# =============================================================================

def match_detections(
    ground_truth: gpd.GeoDataFrame,
    predictions: gpd.GeoDataFrame,
    location_id: str,
    buffer_meters: float = 15
) -> gpd.GeoDataFrame:
    """
    Match ground truth crosswalks with predictions.
    Returns ground truth with 'detected' column (True=TP, False=FN).
    """
    if ground_truth is None or len(ground_truth) == 0:
        return gpd.GeoDataFrame()
    
    # Get location bbox
    if predictions is not None and len(predictions) > 0:
        bounds = predictions.total_bounds
    else:
        # Use location-specific bounds
        loc_bounds = {
            "financial_district": (40.7025, -74.0125, 40.7075, -74.0065),
            "east_village": (40.7235, -73.9900, 40.7295, -73.9830),
            "bay_ridge": (40.6290, -74.0300, 40.6350, -74.0230),
            "downtown_brooklyn": (40.6880, -73.9900, 40.6950, -73.9800),
            "kew_gardens": (40.7050, -73.8350, 40.7150, -73.8200)
        }
        b = loc_bounds.get(location_id, (40.7, -74.0, 40.8, -73.9))
        bounds = [b[1], b[0], b[3], b[2]]  # west, south, east, north
    
    # Filter ground truth to location
    buffer_deg = 0.01
    bbox = box(bounds[0] - buffer_deg, bounds[1] - buffer_deg,
               bounds[2] + buffer_deg, bounds[3] + buffer_deg)
    
    try:
        gt_crs = ground_truth.to_crs(predictions.crs) if predictions is not None else ground_truth
    except:
        gt_crs = ground_truth
    
    gt_local = gt_crs[gt_crs.geometry.intersects(bbox)].copy()
    
    if len(gt_local) == 0:
        return gpd.GeoDataFrame()
    
    # Match with predictions
    gt_local['detected'] = False
    gt_local['location_id'] = location_id
    
    if predictions is not None and len(predictions) > 0:
        buffer_deg = buffer_meters / 111000
        pred_centroids = predictions.geometry.centroid
        pred_buffered = unary_union(pred_centroids.buffer(buffer_deg))
        
        gt_centroids = gt_local.geometry.centroid
        gt_local['detected'] = [c.within(pred_buffered) for c in gt_centroids]
    
    return gt_local


# =============================================================================
# FEATURE EXTRACTION PER CROSSWALK
# =============================================================================

def extract_nearby_features(
    crosswalks: gpd.GeoDataFrame,
    features: Dict[str, gpd.GeoDataFrame]
) -> pd.DataFrame:
    """
    For each crosswalk, extract nearby feature characteristics.
    """
    if len(crosswalks) == 0:
        return pd.DataFrame()
    
    # Project for accurate distances
    try:
        cw_proj = crosswalks.to_crs(epsg=32618)
        features_proj = {k: v.to_crs(epsg=32618) for k, v in features.items()}
    except:
        cw_proj = crosswalks
        features_proj = features
        # Adjust buffers to degrees
        global BUILDING_BUFFER, TREE_BUFFER, ROAD_BUFFER
        BUILDING_BUFFER = BUILDING_BUFFER / 111000
        TREE_BUFFER = TREE_BUFFER / 111000
        ROAD_BUFFER = ROAD_BUFFER / 111000
    
    results = []
    
    for idx, cw in cw_proj.iterrows():
        record = {
            'crosswalk_idx': idx,
            'detected': cw.get('detected', False),
            'location_id': cw.get('location_id', 'unknown')
        }
        
        cw_point = cw.geometry.centroid if hasattr(cw.geometry, 'centroid') else cw.geometry
        
        # 1. Building/Shadow analysis
        if 'shadow_casters' in features_proj:
            buildings = features_proj['shadow_casters']
            buffer = cw_point.buffer(BUILDING_BUFFER)
            nearby = buildings[buildings.geometry.intersects(buffer)]
            
            record['building_count'] = len(nearby)
            
            heights = []
            for _, b in nearby.iterrows():
                h = b.get('estimated_height_m') or b.get('height')
                if h:
                    try:
                        heights.append(float(str(h).replace('m', '').strip()))
                    except:
                        pass
                elif b.get('levels'):
                    try:
                        heights.append(int(b['levels']) * 3.5)
                    except:
                        pass
            
            record['max_building_height'] = max(heights) if heights else 0
            record['avg_building_height'] = np.mean(heights) if heights else 0
            
            # Categorize
            max_h = record['max_building_height']
            if max_h > BUILDING_HEIGHT_THRESHOLDS[1]:
                record['building_height_category'] = 'high'
            elif max_h > BUILDING_HEIGHT_THRESHOLDS[0]:
                record['building_height_category'] = 'medium'
            elif max_h > 0:
                record['building_height_category'] = 'low'
            else:
                record['building_height_category'] = 'none'
        
        # 2. Tree canopy analysis
        if 'tree_canopy' in features_proj:
            trees = features_proj['tree_canopy']
            buffer = cw_point.buffer(TREE_BUFFER)
            nearby = trees[trees.geometry.intersects(buffer)]
            
            record['tree_count'] = len(nearby)
            
            # Categorize
            tc = record['tree_count']
            if tc >= TREE_COUNT_THRESHOLDS[1]:
                record['tree_category'] = 'many'
            elif tc >= TREE_COUNT_THRESHOLDS[0]:
                record['tree_category'] = 'few'
            else:
                record['tree_category'] = 'none'
        
        # 3. Surface type analysis
        if 'surface_type' in features_proj:
            surfaces = features_proj['surface_type']
            buffer = cw_point.buffer(ROAD_BUFFER)
            nearby = surfaces[surfaces.geometry.intersects(buffer)]
            
            if len(nearby) > 0 and 'surface' in nearby.columns:
                surface_vals = nearby['surface'].dropna().tolist()
                record['surface_types'] = list(set(surface_vals))
                
                # Determine primary surface
                if surface_vals:
                    record['primary_surface'] = max(set(surface_vals), key=surface_vals.count)
                else:
                    record['primary_surface'] = 'unknown'
                
                # Contrast level
                low_contrast = ['concrete', 'paving_stones', 'sett', 'cobblestone']
                high_contrast = ['asphalt']
                
                if record['primary_surface'] in high_contrast:
                    record['contrast_category'] = 'high'
                elif record['primary_surface'] in low_contrast:
                    record['contrast_category'] = 'low'
                else:
                    record['contrast_category'] = 'unknown'
            else:
                record['primary_surface'] = 'unknown'
                record['contrast_category'] = 'unknown'
        
        # 4. Road context
        if 'road_context' in features_proj:
            roads = features_proj['road_context']
            buffer = cw_point.buffer(ROAD_BUFFER)
            nearby = roads[roads.geometry.intersects(buffer)]
            
            if len(nearby) > 0 and 'highway_type' in nearby.columns:
                road_types = nearby['highway_type'].dropna().tolist()
                record['road_types'] = list(set(road_types))
                
                if 'residential' in road_types:
                    record['road_category'] = 'residential'
                elif any(r in ['primary', 'secondary'] for r in road_types):
                    record['road_category'] = 'major'
                elif any(r in ['tertiary', 'unclassified'] for r in road_types):
                    record['road_category'] = 'minor'
                else:
                    record['road_category'] = 'other'
            else:
                record['road_category'] = 'unknown'
        
        # 5. Crossing markings
        if 'crossing_markings' in features_proj:
            crossings = features_proj['crossing_markings']
            buffer = cw_point.buffer(5)  # Small buffer for exact match
            nearby = crossings[crossings.geometry.intersects(buffer)]
            
            if len(nearby) > 0:
                if 'markings' in nearby.columns:
                    markings = nearby['markings'].dropna().tolist()
                    record['marking_type'] = markings[0] if markings else 'unknown'
                elif 'crossing_type' in nearby.columns:
                    ct = nearby['crossing_type'].dropna().tolist()
                    record['marking_type'] = ct[0] if ct else 'unknown'
                else:
                    record['marking_type'] = 'unknown'
            else:
                record['marking_type'] = 'unknown'
        
        results.append(record)
    
    return pd.DataFrame(results)


# =============================================================================
# METRICS CALCULATION PER FEATURE GROUP
# =============================================================================

def calculate_metrics_by_feature(df: pd.DataFrame, feature_col: str) -> Dict:
    """
    Calculate detection metrics grouped by a feature column.
    """
    if len(df) == 0 or feature_col not in df.columns:
        return {}
    
    results = {}
    
    for group_val in df[feature_col].unique():
        if pd.isna(group_val):
            continue
        
        group_df = df[df[feature_col] == group_val]
        
        total = len(group_df)
        detected = group_df['detected'].sum()
        missed = total - detected
        
        # For crosswalks, we're looking at recall (did we detect ground truth?)
        # TP = detected, FN = missed
        # We don't have FP here (that requires looking at predictions not matching GT)
        
        recall = detected / total if total > 0 else 0
        
        results[str(group_val)] = {
            'total_crosswalks': int(total),
            'detected': int(detected),
            'missed': int(missed),
            'recall': round(recall, 4),
            'miss_rate': round(1 - recall, 4)
        }
    
    return results


def calculate_all_feature_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate metrics for all feature categories.
    """
    metrics = {}
    
    feature_columns = [
        ('building_height_category', 'Building Height'),
        ('tree_category', 'Tree Proximity'),
        ('contrast_category', 'Surface Contrast'),
        ('road_category', 'Road Type'),
        ('marking_type', 'Crossing Markings')
    ]
    
    for col, display_name in feature_columns:
        if col in df.columns:
            metrics[display_name] = calculate_metrics_by_feature(df, col)
    
    return metrics


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_feature_impact_analysis():
    """Run complete feature impact analysis."""
    
    print("\n" + "=" * 70)
    print("   CROSSCHECK NYC - FEATURE IMPACT ANALYSIS")
    print("   Calculating per-feature detection metrics")
    print("=" * 70 + "\n")
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth
    gt = load_ground_truth()
    if gt is None:
        print("ERROR: No ground truth data found")
        print("Run: python scripts/download_reference.py --all")
        return
    
    print(f"Loaded {len(gt)} ground truth crossings\n")
    
    all_crosswalks = []
    all_metrics = {}
    
    for loc_id in LOCATIONS:
        print(f"\n{'='*50}")
        print(f"Location: {loc_id}")
        print('='*50)
        
        # Load predictions
        preds = load_predictions(loc_id)
        if preds is not None:
            print(f"  Predictions: {len(preds)} crosswalks")
        else:
            print(f"  Predictions: None found")
        
        # Load OSM features
        features = load_osm_features(loc_id)
        print(f"  OSM Features: {list(features.keys())}")
        
        if not features:
            print(f"  No features found. Run: python scripts/extract_osm_features.py")
            continue
        
        # Match detections
        matched = match_detections(gt, preds, loc_id)
        print(f"  Ground truth in area: {len(matched)}")
        
        if len(matched) == 0:
            continue
        
        detected_count = matched['detected'].sum()
        print(f"  Detected (TP): {detected_count}")
        print(f"  Missed (FN): {len(matched) - detected_count}")
        
        # Extract features per crosswalk
        crosswalk_features = extract_nearby_features(matched, features)
        print(f"  Features extracted: {len(crosswalk_features)} rows")
        
        if len(crosswalk_features) > 0:
            all_crosswalks.append(crosswalk_features)
            
            # Calculate metrics for this location
            loc_metrics = calculate_all_feature_metrics(crosswalk_features)
            all_metrics[loc_id] = loc_metrics
    
    # Combine all crosswalks
    if all_crosswalks:
        combined_df = pd.concat(all_crosswalks, ignore_index=True)
        print(f"\n\nTotal crosswalks analyzed: {len(combined_df)}")
        
        # Calculate overall metrics
        overall_metrics = calculate_all_feature_metrics(combined_df)
        all_metrics['overall'] = overall_metrics
        
        # Save raw data
        combined_df.to_csv(METRICS_DIR / "crosswalk_features.csv", index=False)
        
        # Save metrics
        with open(METRICS_DIR / "feature_impact_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 70)
        print("FEATURE IMPACT SUMMARY (Overall)")
        print("=" * 70)
        
        for feature_name, groups in overall_metrics.items():
            print(f"\n{feature_name}:")
            print(f"  {'Category':<15} {'Total':>8} {'Detected':>10} {'Missed':>8} {'Recall':>10}")
            print(f"  {'-'*55}")
            
            for cat, data in sorted(groups.items()):
                print(f"  {cat:<15} {data['total_crosswalks']:>8} {data['detected']:>10} "
                      f"{data['missed']:>8} {data['recall']:>10.1%}")
        
        print(f"\n✓ Raw data saved to: {METRICS_DIR}/crosswalk_features.csv")
        print(f"✓ Metrics saved to: {METRICS_DIR}/feature_impact_metrics.json")
        
        return all_metrics
    
    else:
        print("\nNo crosswalks could be analyzed.")
        return None


if __name__ == "__main__":
    run_feature_impact_analysis()