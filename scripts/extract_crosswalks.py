#!/usr/bin/env python3
"""
CrossCheck NYC - Crosswalk Extraction
=====================================
Extract and analyze crosswalk-specific data from tile2net outputs.

Usage:
    python extract_crosswalks.py --location financial_district
    python extract_crosswalks.py --all
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

import geopandas as gpd
from PIL import Image
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "locations.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

# Tile2net class labels
CLASS_LABELS = {0: "background", 1: "sidewalk", 2: "crosswalk", 3: "road"}
CROSSWALK_CLASS = 2


def load_locations() -> Dict:
    """Load location configurations."""
    with open(CONFIG_PATH) as f:
        return {loc['id']: loc for loc in json.load(f)['locations']}


class CrosswalkExtractor:
    """Extract crosswalk data from a tile2net project."""
    
    def __init__(self, location_id: str, output_dir: Path):
        self.location_id = location_id
        self.project_dir = output_dir / location_id
        self.polygons_dir = self.project_dir / "polygons"
        self.seg_dir = self.project_dir / "segmentation" / "seg_results"
        
    def extract_polygons(self) -> Optional[gpd.GeoDataFrame]:
        """Extract crosswalk polygons."""
        crosswalk_path = self.polygons_dir / "crosswalk.geojson"
        
        if not crosswalk_path.exists():
            logger.warning(f"Crosswalk file not found: {crosswalk_path}")
            return None
        
        gdf = gpd.read_file(crosswalk_path)
        logger.info(f"Loaded {len(gdf)} crosswalk polygons from {self.location_id}")
        
        # Add derived columns
        if len(gdf) > 0:
            gdf['area_sqm'] = gdf.geometry.area * 111000**2  # Approx conversion
            gdf['centroid_lon'] = gdf.geometry.centroid.x
            gdf['centroid_lat'] = gdf.geometry.centroid.y
            gdf['location_id'] = self.location_id
            
        return gdf
    
    def extract_pixel_stats(self) -> Dict:
        """Extract pixel statistics from segmentation images."""
        if not self.seg_dir.exists():
            logger.warning(f"Segmentation dir not found: {self.seg_dir}")
            return {"error": "No segmentation images (run with --dump_percent 100)"}
        
        stats = {
            "location_id": self.location_id,
            "images": [],
            "total_pixels": 0,
            "crosswalk_pixels": 0
        }
        
        for img_path in sorted(self.seg_dir.glob("*.png")):
            img = np.array(Image.open(img_path))
            unique, counts = np.unique(img, return_counts=True)
            class_counts = dict(zip(unique, counts))
            
            img_stats = {
                "filename": img_path.name,
                "total_pixels": int(img.size),
                "crosswalk_pixels": int(class_counts.get(CROSSWALK_CLASS, 0)),
                "class_distribution": {
                    CLASS_LABELS.get(k, f"class_{k}"): int(v)
                    for k, v in class_counts.items()
                }
            }
            
            stats["images"].append(img_stats)
            stats["total_pixels"] += img.size
            stats["crosswalk_pixels"] += class_counts.get(CROSSWALK_CLASS, 0)
        
        if stats["total_pixels"] > 0:
            stats["crosswalk_percentage"] = stats["crosswalk_pixels"] / stats["total_pixels"] * 100
            
        logger.info(f"Processed {len(stats['images'])} segmentation images")
        return stats
    
    def compute_statistics(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Compute statistics about crosswalk detections."""
        if gdf is None or len(gdf) == 0:
            return {"count": 0}
        
        areas = gdf['area_sqm']
        
        return {
            "count": len(gdf),
            "total_area_sqm": float(areas.sum()),
            "mean_area_sqm": float(areas.mean()),
            "median_area_sqm": float(areas.median()),
            "std_area_sqm": float(areas.std()),
            "min_area_sqm": float(areas.min()),
            "max_area_sqm": float(areas.max()),
            "bounds": {
                "minx": float(gdf.total_bounds[0]),
                "miny": float(gdf.total_bounds[1]),
                "maxx": float(gdf.total_bounds[2]),
                "maxy": float(gdf.total_bounds[3])
            }
        }
    
    def extract_all(self) -> Dict:
        """Run full extraction."""
        results = {
            "location_id": self.location_id,
            "project_dir": str(self.project_dir),
            "exists": self.project_dir.exists()
        }
        
        if not results["exists"]:
            results["error"] = "Project directory not found"
            return results
        
        # Extract polygons
        gdf = self.extract_polygons()
        results["polygon_stats"] = self.compute_statistics(gdf)
        
        # Extract pixel stats
        results["pixel_stats"] = self.extract_pixel_stats()
        
        return results


def extract_all_locations():
    """Extract crosswalk data from all locations."""
    locations = load_locations()
    all_results = []
    all_crosswalks = []
    
    for loc_id in locations:
        logger.info(f"\n{'='*50}")
        logger.info(f"Extracting: {loc_id}")
        logger.info('='*50)
        
        extractor = CrosswalkExtractor(loc_id, OUTPUT_DIR)
        results = extractor.extract_all()
        all_results.append(results)
        
        # Collect polygons
        gdf = extractor.extract_polygons()
        if gdf is not None and len(gdf) > 0:
            all_crosswalks.append(gdf)
    
    # Save combined results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    results_path = RESULTS_DIR / "crosswalk_extraction.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Combined GeoJSON
    if all_crosswalks:
        import pandas as pd
        combined = gpd.GeoDataFrame(pd.concat(all_crosswalks, ignore_index=True))
        combined_path = RESULTS_DIR / "all_crosswalks_combined.geojson"
        combined.to_file(combined_path, driver='GeoJSON')
        logger.info(f"Combined GeoJSON saved to: {combined_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    for r in all_results:
        count = r.get('polygon_stats', {}).get('count', 0)
        print(f"  {r['location_id']}: {count} crosswalks")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Extract crosswalk data")
    parser.add_argument('--location', '-l', type=str, help='Specific location')
    parser.add_argument('--all', '-a', action='store_true', help='All locations')
    
    args = parser.parse_args()
    
    if args.all:
        extract_all_locations()
    elif args.location:
        extractor = CrosswalkExtractor(args.location, OUTPUT_DIR)
        results = extractor.extract_all()
        print(json.dumps(results, indent=2, default=str))
    else:
        print("Usage: python extract_crosswalks.py --all")
        print("       python extract_crosswalks.py --location financial_district")


if __name__ == "__main__":
    main()
