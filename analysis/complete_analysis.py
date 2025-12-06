#!/usr/bin/env python3
"""
CrossCheck NYC - Complete Analysis Pipeline
============================================
This single script runs all analysis stages.
Robust handling of different folder structures and image formats.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

import geopandas as gpd
import pandas as pd
from PIL import Image
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS IF NEEDED
# =============================================================================
OUTPUT_DIR = Path("./outputs")
REFERENCE_DIR = Path("./reference")
RESULTS_DIR = Path("./results")

# Class definitions from tile2net
CLASS_LABELS = {0: "background", 1: "road", 2: "crosswalk", 3: "sidewalk"}
CROSSWALK_CLASS = 2

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_all_locations(output_dir: Path) -> List[str]:
    """Auto-detect all location folders."""
    locations = []
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                locations.append(item.name)
    logger.info(f"Found locations: {locations}")
    return locations


def find_segmentation_images(project_dir: Path) -> List[Path]:
    """Find all segmentation PNG files."""
    seg_images = []
    
    # Try multiple possible paths
    possible_paths = [
        project_dir / "segmentation",
        project_dir / "seg_results",
        project_dir,
    ]
    
    for base_path in possible_paths:
        if base_path.exists():
            # Search recursively for seg_results folder or PNG files
            for seg_dir in base_path.rglob("seg_results"):
                seg_images.extend(seg_dir.glob("*.png"))
            
            # Also check for PNGs directly in segmentation folder
            if "segmentation" in str(base_path):
                for png in base_path.rglob("*.png"):
                    if png not in seg_images:
                        seg_images.append(png)
    
    return sorted(seg_images)


def find_polygon_files(project_dir: Path) -> List[Path]:
    """Find all polygon shapefiles or geojsons."""
    polygons_dir = project_dir / "polygons"
    
    if not polygons_dir.exists():
        return []
    
    files = []
    
    # Find shapefiles
    for shp in polygons_dir.rglob("*.shp"):
        files.append(shp)
    
    # Find geojsons
    for geojson in polygons_dir.rglob("*.geojson"):
        files.append(geojson)
    
    return files


def load_segmentation_image(img_path: Path) -> np.ndarray:
    """Load and normalize segmentation image."""
    img = np.array(Image.open(img_path))
    
    # Handle different image formats
    if len(img.shape) == 3:
        # Multi-channel - take first channel or convert
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, 0]
        elif img.shape[2] == 3:  # RGB - might be colored segmentation
            # Check if it's a colored segmentation map
            # Convert to class labels based on colors
            img = img[:, :, 0]  # Simplified - take first channel
    
    return img


def analyze_class_distribution(img: np.ndarray) -> Dict:
    """Analyze pixel class distribution."""
    unique, counts = np.unique(img, return_counts=True)
    
    total_pixels = img.size
    distribution = {}
    
    for cls, count in zip(unique, counts):
        cls_name = CLASS_LABELS.get(int(cls), f"class_{cls}")
        distribution[cls_name] = {
            "class_id": int(cls),
            "pixels": int(count),
            "percentage": round(count / total_pixels * 100, 4)
        }
    
    return distribution


# =============================================================================
# STAGE A: PIXEL-LEVEL ANALYSIS
# =============================================================================

def run_stage_a(location_id: str, project_dir: Path) -> Dict:
    """Run pixel-level analysis for a location."""
    logger.info(f"Stage A: {location_id}")
    
    results = {
        "location_id": location_id,
        "status": "success",
        "images_analyzed": 0,
        "total_pixels": 0,
        "class_totals": defaultdict(int),
        "crosswalk_stats": {},
        "images": []
    }
    
    seg_images = find_segmentation_images(project_dir)
    
    if not seg_images:
        results["status"] = "no_images"
        results["error"] = "No segmentation images found"
        logger.warning(f"  No segmentation images found for {location_id}")
        return results
    
    logger.info(f"  Found {len(seg_images)} segmentation images")
    
    all_crosswalk_pixels = 0
    all_total_pixels = 0
    region_counts = []
    
    for img_path in seg_images:
        try:
            img = load_segmentation_image(img_path)
            
            # Analyze distribution
            dist = analyze_class_distribution(img)
            
            # Count crosswalk pixels
            crosswalk_pixels = 0
            for cls_name, cls_data in dist.items():
                if "crosswalk" in cls_name.lower() or cls_data["class_id"] == CROSSWALK_CLASS:
                    crosswalk_pixels = cls_data["pixels"]
                results["class_totals"][cls_name] += cls_data["pixels"]
            
            all_crosswalk_pixels += crosswalk_pixels
            all_total_pixels += img.size
            
            # Count connected regions
            crosswalk_mask = (img == CROSSWALK_CLASS).astype(np.uint8)
            from scipy import ndimage
            _, num_regions = ndimage.label(crosswalk_mask)
            region_counts.append(num_regions)
            
            results["images"].append({
                "filename": img_path.name,
                "pixels": int(img.size),
                "crosswalk_pixels": int(crosswalk_pixels),
                "crosswalk_pct": round(crosswalk_pixels / img.size * 100, 4),
                "regions": int(num_regions)
            })
            
        except Exception as e:
            logger.warning(f"  Error processing {img_path.name}: {e}")
    
    results["images_analyzed"] = len(results["images"])
    results["total_pixels"] = all_total_pixels
    results["crosswalk_stats"] = {
        "total_crosswalk_pixels": int(all_crosswalk_pixels),
        "overall_percentage": round(all_crosswalk_pixels / all_total_pixels * 100, 4) if all_total_pixels > 0 else 0,
        "total_regions": int(sum(region_counts)),
        "avg_regions_per_image": round(np.mean(region_counts), 2) if region_counts else 0
    }
    
    # Convert defaultdict to regular dict
    results["class_totals"] = dict(results["class_totals"])
    
    return results


# =============================================================================
# STAGE B: NETWORK/POLYGON ANALYSIS
# =============================================================================

def run_stage_b(location_id: str, project_dir: Path) -> Dict:
    """Run network/polygon analysis for a location."""
    logger.info(f"Stage B: {location_id}")
    
    results = {
        "location_id": location_id,
        "status": "success",
        "polygon_count": 0,
        "area_stats": {},
        "class_breakdown": {}
    }
    
    polygon_files = find_polygon_files(project_dir)
    
    if not polygon_files:
        results["status"] = "no_polygons"
        results["error"] = "No polygon files found"
        logger.warning(f"  No polygon files found for {location_id}")
        return results
    
    logger.info(f"  Found {len(polygon_files)} polygon files")
    
    all_polygons = []
    
    for poly_file in polygon_files:
        try:
            gdf = gpd.read_file(poly_file)
            gdf["source_file"] = poly_file.name
            all_polygons.append(gdf)
            logger.info(f"    Loaded {len(gdf)} polygons from {poly_file.name}")
        except Exception as e:
            logger.warning(f"    Error loading {poly_file.name}: {e}")
    
    if not all_polygons:
        results["status"] = "load_error"
        results["error"] = "Could not load any polygon files"
        return results
    
    # Combine all polygons
    combined = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True))
    results["polygon_count"] = len(combined)
    
    # Area statistics (convert to approx square meters)
    areas = combined.geometry.area * 111000 * 111000  # Rough conversion
    
    results["area_stats"] = {
        "total_sqm": float(areas.sum()),
        "mean_sqm": float(areas.mean()),
        "median_sqm": float(areas.median()),
        "min_sqm": float(areas.min()),
        "max_sqm": float(areas.max()),
        "std_sqm": float(areas.std()) if len(areas) > 1 else 0
    }
    
    # Class breakdown if available
    class_col = None
    for col in ['class', 'Class', 'CLASS', 'f_type', 'type', 'Type']:
        if col in combined.columns:
            class_col = col
            break
    
    if class_col:
        class_counts = combined[class_col].value_counts().to_dict()
        results["class_breakdown"] = {str(k): int(v) for k, v in class_counts.items()}
    
    # Identify crosswalks
    crosswalk_count = 0
    if class_col:
        crosswalk_mask = combined[class_col].astype(str).str.lower().str.contains('crosswalk|crossing', na=False)
        crosswalk_count = crosswalk_mask.sum()
    
    results["crosswalk_polygons"] = int(crosswalk_count)
    
    return results


# =============================================================================
# REFERENCE COMPARISON
# =============================================================================

def run_reference_comparison(location_id: str, project_dir: Path, reference_dir: Path) -> Dict:
    """Compare predictions with reference data."""
    logger.info(f"Reference comparison: {location_id}")
    
    results = {
        "location_id": location_id,
        "status": "success",
        "comparisons": {}
    }
    
    # Load predictions
    polygon_files = find_polygon_files(project_dir)
    if not polygon_files:
        results["status"] = "no_predictions"
        return results
    
    try:
        pred_gdf = gpd.read_file(polygon_files[0])
    except Exception as e:
        results["status"] = "load_error"
        results["error"] = str(e)
        return results
    
    # Get prediction bounds for filtering reference
    pred_bounds = pred_gdf.total_bounds
    
    # Load and compare with each reference file
    if not reference_dir.exists():
        results["status"] = "no_reference"
        results["error"] = "Reference directory not found"
        return results
    
    for ref_file in reference_dir.glob("*.geojson"):
        if "raw" in ref_file.name:
            continue
        
        try:
            ref_gdf = gpd.read_file(ref_file)
            
            # Ensure same CRS
            if pred_gdf.crs and ref_gdf.crs and pred_gdf.crs != ref_gdf.crs:
                ref_gdf = ref_gdf.to_crs(pred_gdf.crs)
            
            # Filter reference to prediction area
            buffer = 0.01  # ~1km buffer
            bbox = (pred_bounds[0] - buffer, pred_bounds[1] - buffer,
                    pred_bounds[2] + buffer, pred_bounds[3] + buffer)
            
            ref_local = ref_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            
            if len(ref_local) == 0:
                results["comparisons"][ref_file.stem] = {
                    "status": "no_overlap",
                    "reference_count": 0
                }
                continue
            
            # Simple spatial comparison using buffer matching
            buffer_dist = 0.0001  # ~11m
            
            pred_centroids = pred_gdf.geometry.centroid
            ref_points = ref_local.geometry.centroid if ref_local.geometry.geom_type.iloc[0] != 'Point' else ref_local.geometry
            
            ref_union = unary_union(ref_points.buffer(buffer_dist))
            pred_union = unary_union(pred_centroids.buffer(buffer_dist))
            
            matched_pred = sum(1 for c in pred_centroids if c.within(ref_union))
            matched_ref = sum(1 for p in ref_points if p.within(pred_union))
            
            n_pred = len(pred_gdf)
            n_ref = len(ref_local)
            
            precision = matched_pred / n_pred if n_pred > 0 else 0
            recall = matched_ref / n_ref if n_ref > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results["comparisons"][ref_file.stem] = {
                "predictions": n_pred,
                "references": n_ref,
                "matched_predictions": matched_pred,
                "matched_references": matched_ref,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
            
        except Exception as e:
            results["comparisons"][ref_file.stem] = {"error": str(e)}
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_summary_visualization(all_results: Dict, output_path: Path):
    """Create summary visualization of all results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        locations = []
        crosswalk_pcts = []
        polygon_counts = []
        
        for loc_id, data in all_results["stage_a"].items():
            if data.get("status") == "success":
                locations.append(loc_id)
                crosswalk_pcts.append(data["crosswalk_stats"].get("overall_percentage", 0))
        
        for loc_id, data in all_results["stage_b"].items():
            if data.get("status") == "success":
                if loc_id not in locations:
                    locations.append(loc_id)
                polygon_counts.append(data.get("polygon_count", 0))
        
        # Plot 1: Crosswalk percentage by location
        if crosswalk_pcts:
            ax = axes[0, 0]
            colors = plt.cm.Set2(np.linspace(0, 1, len(locations)))
            ax.bar(locations[:len(crosswalk_pcts)], crosswalk_pcts, color=colors)
            ax.set_ylabel('Crosswalk %')
            ax.set_title('Crosswalk Pixel Percentage by Location')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Polygon counts
        if polygon_counts:
            ax = axes[0, 1]
            ax.bar(locations[:len(polygon_counts)], polygon_counts, color=colors)
            ax.set_ylabel('Count')
            ax.set_title('Polygon Count by Location')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Class distribution (first location with data)
        ax = axes[1, 0]
        for loc_id, data in all_results["stage_a"].items():
            if data.get("status") == "success" and data.get("class_totals"):
                class_totals = data["class_totals"]
                ax.pie(class_totals.values(), labels=class_totals.keys(), autopct='%1.1f%%')
                ax.set_title(f'Class Distribution ({loc_id})')
                break
        
        # Plot 4: Reference comparison (if available)
        ax = axes[1, 1]
        ref_data = all_results.get("reference_comparison", {})
        f1_scores = []
        ref_locations = []
        
        for loc_id, data in ref_data.items():
            if data.get("comparisons"):
                for ref_name, comp in data["comparisons"].items():
                    if isinstance(comp, dict) and "f1_score" in comp:
                        ref_locations.append(f"{loc_id[:10]}")
                        f1_scores.append(comp["f1_score"])
                        break  # Just first reference
        
        if f1_scores:
            ax.bar(ref_locations, f1_scores, color='green', alpha=0.7)
            ax.set_ylabel('F1 Score')
            ax.set_title('Reference Comparison (F1 Score)')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, 'No reference data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Reference Comparison')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization to {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline():
    """Run the complete analysis pipeline."""
    
    print("\n" + "=" * 70)
    print("   CROSSCHECK NYC - COMPLETE ANALYSIS PIPELINE")
    print("=" * 70 + "\n")
    
    # Create results directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "stage_a").mkdir(exist_ok=True)
    (RESULTS_DIR / "stage_b").mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    
    # Find all locations
    locations = find_all_locations(OUTPUT_DIR)
    
    if not locations:
        print("ERROR: No locations found in ./outputs/")
        print("Run tile2net inference first!")
        return
    
    all_results = {
        "stage_a": {},
        "stage_b": {},
        "reference_comparison": {}
    }
    
    # =================================
    # STAGE A: Pixel-Level Analysis
    # =================================
    print("\n" + "=" * 50)
    print("STAGE A: PIXEL-LEVEL ANALYSIS")
    print("=" * 50)
    
    for loc_id in locations:
        project_dir = OUTPUT_DIR / loc_id
        results = run_stage_a(loc_id, project_dir)
        all_results["stage_a"][loc_id] = results
        
        # Save individual results
        output_path = RESULTS_DIR / "stage_a" / f"{loc_id}_pixel_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print Stage A Summary
    print("\n" + "-" * 70)
    print("STAGE A SUMMARY")
    print("-" * 70)
    print(f"{'Location':<25} {'Images':>10} {'Crosswalk %':>15} {'Regions':>12}")
    print("-" * 70)
    
    for loc_id, data in all_results["stage_a"].items():
        if data["status"] == "success":
            stats = data["crosswalk_stats"]
            print(f"{loc_id:<25} {data['images_analyzed']:>10} "
                  f"{stats['overall_percentage']:>14.2f}% "
                  f"{stats['total_regions']:>12}")
        else:
            print(f"{loc_id:<25} {'ERROR: ' + data.get('error', 'Unknown'):<40}")
    
    # =================================
    # STAGE B: Network/Polygon Analysis
    # =================================
    print("\n" + "=" * 50)
    print("STAGE B: POLYGON/NETWORK ANALYSIS")
    print("=" * 50)
    
    for loc_id in locations:
        project_dir = OUTPUT_DIR / loc_id
        results = run_stage_b(loc_id, project_dir)
        all_results["stage_b"][loc_id] = results
        
        # Save individual results
        output_path = RESULTS_DIR / "stage_b" / f"{loc_id}_network_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print Stage B Summary
    print("\n" + "-" * 70)
    print("STAGE B SUMMARY")
    print("-" * 70)
    print(f"{'Location':<25} {'Polygons':>12} {'Total Area (m²)':>18} {'Crosswalks':>12}")
    print("-" * 70)
    
    for loc_id, data in all_results["stage_b"].items():
        if data["status"] == "success":
            area = data["area_stats"].get("total_sqm", 0)
            print(f"{loc_id:<25} {data['polygon_count']:>12} "
                  f"{area:>17.1f} "
                  f"{data.get('crosswalk_polygons', 'N/A'):>12}")
        else:
            print(f"{loc_id:<25} {'ERROR: ' + data.get('error', 'Unknown'):<40}")
    
    # =================================
    # REFERENCE COMPARISON
    # =================================
    print("\n" + "=" * 50)
    print("REFERENCE COMPARISON")
    print("=" * 50)
    
    if REFERENCE_DIR.exists() and list(REFERENCE_DIR.glob("*.geojson")):
        for loc_id in locations:
            project_dir = OUTPUT_DIR / loc_id
            results = run_reference_comparison(loc_id, project_dir, REFERENCE_DIR)
            all_results["reference_comparison"][loc_id] = results
        
        # Print Reference Summary
        print("\n" + "-" * 70)
        print(f"{'Location':<25} {'Reference':>20} {'Precision':>12} {'Recall':>12} {'F1':>10}")
        print("-" * 70)
        
        for loc_id, data in all_results["reference_comparison"].items():
            if data.get("comparisons"):
                for ref_name, comp in data["comparisons"].items():
                    if isinstance(comp, dict) and "precision" in comp:
                        print(f"{loc_id:<25} {ref_name[:20]:>20} "
                              f"{comp['precision']:>12.3f} "
                              f"{comp['recall']:>12.3f} "
                              f"{comp['f1_score']:>10.3f}")
            else:
                print(f"{loc_id:<25} {'No comparison data':<50}")
    else:
        print("No reference data found. Skipping comparison.")
        print("Run: python scripts/download_reference.py --all")
    
    # =================================
    # SAVE COMBINED RESULTS
    # =================================
    combined_path = RESULTS_DIR / "complete_analysis.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"  - stage_a/           (pixel-level analysis)")
    print(f"  - stage_b/           (polygon/network analysis)")
    print(f"  - complete_analysis.json (combined results)")
    
    # Create visualization
    viz_path = RESULTS_DIR / "figures" / "analysis_summary.png"
    create_summary_visualization(all_results, viz_path)
    
    return all_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_full_pipeline()
