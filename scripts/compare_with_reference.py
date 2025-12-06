#!/usr/bin/env python3
"""
CrossCheck NYC - Compare Predictions with Reference
====================================================
Compare tile2net crosswalk predictions with reference data.

Usage:
    python compare_with_reference.py --all
    python compare_with_reference.py --location financial_district
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import logging

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "stage_b"


def load_predictions(location_id: str) -> gpd.GeoDataFrame:
    """Load crosswalk predictions for a location."""
    path = OUTPUT_DIR / location_id / "polygons" / "crosswalk.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    return gpd.read_file(path)


def load_reference(reference_type: str = "combined") -> gpd.GeoDataFrame:
    """Load reference dataset."""
    if reference_type == "combined":
        path = REFERENCE_DIR / "combined_reference.geojson"
    else:
        path = REFERENCE_DIR / f"{reference_type}.geojson"
    
    if not path.exists():
        raise FileNotFoundError(f"Reference not found: {path}")
    return gpd.read_file(path)


def spatial_comparison(predictions: gpd.GeoDataFrame, 
                       reference: gpd.GeoDataFrame,
                       buffer_distance: float = 0.0001) -> Dict:
    """
    Compare predictions with reference using buffered spatial matching.
    
    Args:
        predictions: Predicted crosswalks
        reference: Reference crosswalks
        buffer_distance: Buffer in degrees (~11m at this latitude)
    
    Returns:
        Dict with comparison metrics
    """
    # Ensure same CRS
    if predictions.crs != reference.crs:
        reference = reference.to_crs(predictions.crs)
    
    # Get centroids
    pred_centroids = predictions.geometry.centroid
    
    # Handle different geometry types
    if reference.geometry.geom_type.iloc[0] in ['Point', 'MultiPoint']:
        ref_points = reference.geometry
    else:
        ref_points = reference.geometry.centroid
    
    # Buffer and union for matching
    ref_buffered = unary_union(ref_points.buffer(buffer_distance))
    pred_buffered = unary_union(pred_centroids.buffer(buffer_distance))
    
    # Count matches
    matched_pred = sum(1 for c in pred_centroids if c.within(ref_buffered))
    matched_ref = sum(1 for p in ref_points if p.within(pred_buffered))
    
    # Metrics
    n_pred = len(predictions)
    n_ref = len(reference)
    
    precision = matched_pred / n_pred if n_pred > 0 else 0
    recall = matched_ref / n_ref if n_ref > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "num_predictions": n_pred,
        "num_references": n_ref,
        "buffer_meters": buffer_distance * 111000,
        "true_positives": matched_pred,
        "matched_references": matched_ref,
        "false_positives": n_pred - matched_pred,
        "false_negatives": n_ref - matched_ref,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }


def compare_location(location_id: str) -> Dict:
    """Run comparison for a single location."""
    logger.info(f"Comparing: {location_id}")
    
    results = {"location_id": location_id, "comparisons": {}}
    
    try:
        predictions = load_predictions(location_id)
        logger.info(f"  Loaded {len(predictions)} predictions")
    except FileNotFoundError as e:
        results["error"] = str(e)
        return results
    
    # Get bounding box for filtering reference
    bounds = predictions.total_bounds
    pred_bbox = gpd.GeoDataFrame(
        geometry=[predictions.unary_union.convex_hull.buffer(0.001)],
        crs=predictions.crs
    )
    
    # Compare with each reference dataset
    for ref_file in REFERENCE_DIR.glob("*.geojson"):
        if "raw" in ref_file.name:
            continue
            
        ref_name = ref_file.stem
        logger.info(f"  vs {ref_name}...")
        
        try:
            reference = gpd.read_file(ref_file)
            
            # Filter to prediction area
            reference = reference.to_crs(predictions.crs)
            reference_local = reference[reference.geometry.intersects(pred_bbox.geometry.iloc[0])]
            
            if len(reference_local) == 0:
                results["comparisons"][ref_name] = {"note": "No reference data in area"}
                continue
            
            comparison = spatial_comparison(predictions, reference_local)
            results["comparisons"][ref_name] = comparison
            
            logger.info(f"    P: {comparison['precision']:.3f}, R: {comparison['recall']:.3f}, F1: {comparison['f1_score']:.3f}")
            
        except Exception as e:
            results["comparisons"][ref_name] = {"error": str(e)}
    
    return results


def compare_all_locations() -> List[Dict]:
    """Compare all locations with reference data."""
    locations = ["financial_district", "central_park_south", "bay_ridge", "downtown_brooklyn"]
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    logger.info("\n" + "=" * 60)
    logger.info("CROSSCHECK NYC - REFERENCE COMPARISON")
    logger.info("=" * 60 + "\n")
    
    for loc_id in locations:
        results = compare_location(loc_id)
        all_results.append(results)
        print()
    
    # Save results
    output_path = RESULTS_DIR / "reference_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (vs combined_reference)")
    print("=" * 80)
    print(f"{'Location':<25} {'Predictions':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 80)
    
    for r in all_results:
        loc = r['location_id']
        if 'error' in r:
            print(f"{loc:<25} {'ERROR':>12}")
        else:
            comp = r['comparisons'].get('combined_reference', {})
            if 'error' in comp or 'note' in comp:
                print(f"{loc:<25} {'N/A':>12}")
            else:
                print(f"{loc:<25} {comp.get('num_predictions', 0):>12} "
                      f"{comp.get('precision', 0):>12.3f} "
                      f"{comp.get('recall', 0):>12.3f} "
                      f"{comp.get('f1_score', 0):>12.3f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare with reference data")
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--location', '-l', type=str)
    
    args = parser.parse_args()
    
    if args.all:
        compare_all_locations()
    elif args.location:
        results = compare_location(args.location)
        print(json.dumps(results, indent=2))
    else:
        print("Usage: python compare_with_reference.py --all")


if __name__ == "__main__":
    main()
