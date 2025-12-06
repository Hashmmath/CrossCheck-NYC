#!/usr/bin/env python3
"""
CrossCheck NYC - Stage A: Pixel-Level Analysis
===============================================
Analyze segmentation outputs at the pixel level.

Research Questions Addressed:
- RQ1: When does the model detect crosswalks correctly?
- RQ2: How do threshold changes affect predictions?
- RQ5: What visual cues help users trust predictions?

Usage:
    python stage_a_pixel_analysis.py --all
    python stage_a_pixel_analysis.py --location financial_district
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

from PIL import Image
import cv2
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "stage_a"
FIGURES_DIR = PROJECT_ROOT / "data" / "results" / "figures" / "segmentation_samples"

# Class definitions
CLASS_LABELS = {0: "Background", 1: "Sidewalk", 2: "Crosswalk", 3: "Road"}
CLASS_COLORS = {
    0: (0, 0, 0),       # Black
    1: (0, 0, 255),     # Blue
    2: (255, 0, 0),     # Red
    3: (0, 255, 0)      # Green
}
CROSSWALK_CLASS = 2


class PixelAnalyzer:
    """Pixel-level analysis of segmentation outputs."""
    
    def __init__(self, location_id: str):
        self.location_id = location_id
        self.project_dir = OUTPUT_DIR / location_id
        self.seg_dir = self.project_dir / "segmentation" / "seg_results"
        self.output_dir = RESULTS_DIR / location_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_image(self, img_path: Path) -> Dict:
        """Analyze a single segmentation image."""
        img = np.array(Image.open(img_path))
        
        # Class distribution
        unique, counts = np.unique(img, return_counts=True)
        class_dist = {
            CLASS_LABELS.get(k, f"class_{k}"): int(v)
            for k, v in zip(unique, counts)
        }
        
        # Crosswalk-specific analysis
        crosswalk_mask = (img == CROSSWALK_CLASS).astype(np.uint8)
        
        # Connected components
        labeled, num_regions = ndimage.label(crosswalk_mask)
        
        regions = []
        if num_regions > 0:
            props = measure.regionprops(labeled)
            for prop in props:
                regions.append({
                    "area": int(prop.area),
                    "bbox": prop.bbox,
                    "centroid": prop.centroid,
                    "eccentricity": float(prop.eccentricity) if hasattr(prop, 'eccentricity') else None,
                    "solidity": float(prop.solidity) if hasattr(prop, 'solidity') else None
                })
        
        return {
            "filename": img_path.name,
            "shape": img.shape,
            "total_pixels": int(img.size),
            "class_distribution": class_dist,
            "crosswalk_pixels": int(crosswalk_mask.sum()),
            "crosswalk_percentage": float(crosswalk_mask.sum() / img.size * 100),
            "num_crosswalk_regions": num_regions,
            "regions": regions
        }
    
    def analyze_morphology_effects(self, img_path: Path) -> Dict:
        """Analyze effect of morphological operations."""
        img = np.array(Image.open(img_path))
        crosswalk_mask = (img == CROSSWALK_CLASS).astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8)
        
        operations = {
            "original": crosswalk_mask,
            "erode_1": cv2.erode(crosswalk_mask, kernel, iterations=1),
            "erode_2": cv2.erode(crosswalk_mask, kernel, iterations=2),
            "dilate_1": cv2.dilate(crosswalk_mask, kernel, iterations=1),
            "dilate_2": cv2.dilate(crosswalk_mask, kernel, iterations=2),
            "open": cv2.morphologyEx(crosswalk_mask, cv2.MORPH_OPEN, kernel),
            "close": cv2.morphologyEx(crosswalk_mask, cv2.MORPH_CLOSE, kernel)
        }
        
        results = {}
        for name, result in operations.items():
            _, num_regions = ndimage.label(result)
            results[name] = {
                "pixels": int(result.sum()),
                "regions": num_regions
            }
        
        return results
    
    def create_visualization(self, img_path: Path, save_path: Path):
        """Create visualization of segmentation."""
        img = np.array(Image.open(img_path))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Colored segmentation
        colored = np.zeros((*img.shape, 3), dtype=np.uint8)
        for cls, color in CLASS_COLORS.items():
            colored[img == cls] = color
        axes[0].imshow(colored)
        axes[0].set_title('Segmentation')
        axes[0].axis('off')
        
        # Crosswalk only
        crosswalk_mask = (img == CROSSWALK_CLASS).astype(np.uint8) * 255
        axes[1].imshow(crosswalk_mask, cmap='Reds')
        axes[1].set_title('Crosswalks Only')
        axes[1].axis('off')
        
        # Class distribution
        unique, counts = np.unique(img, return_counts=True)
        names = [CLASS_LABELS.get(u, f'cls_{u}') for u in unique]
        colors = [np.array(CLASS_COLORS.get(u, (128, 128, 128))) / 255 for u in unique]
        axes[2].bar(names, counts, color=colors)
        axes[2].set_title('Pixel Distribution')
        axes[2].set_ylabel('Count')
        
        # Legend
        patches = [mpatches.Patch(color=np.array(c)/255, label=CLASS_LABELS[k]) 
                   for k, c in CLASS_COLORS.items()]
        fig.legend(handles=patches, loc='lower center', ncol=4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self) -> Dict:
        """Run full Stage A analysis."""
        if not self.seg_dir.exists():
            return {"location_id": self.location_id, "error": "No segmentation images"}
        
        results = {
            "location_id": self.location_id,
            "images": [],
            "summary": {}
        }
        
        total_crosswalk_pixels = 0
        total_pixels = 0
        total_regions = 0
        all_region_sizes = []
        
        # Create figures directory
        fig_dir = FIGURES_DIR / self.location_id
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        seg_files = sorted(self.seg_dir.glob("*.png"))
        logger.info(f"Processing {len(seg_files)} images for {self.location_id}")
        
        for i, seg_path in enumerate(seg_files):
            # Basic analysis
            img_analysis = self.analyze_image(seg_path)
            
            # Morphology effects (sample)
            if i < 5:
                img_analysis["morphology"] = self.analyze_morphology_effects(seg_path)
                
                # Create visualization
                vis_path = fig_dir / f"vis_{seg_path.stem}.png"
                self.create_visualization(seg_path, vis_path)
            
            results["images"].append(img_analysis)
            
            # Accumulate stats
            total_crosswalk_pixels += img_analysis["crosswalk_pixels"]
            total_pixels += img_analysis["total_pixels"]
            total_regions += img_analysis["num_crosswalk_regions"]
            
            for r in img_analysis["regions"]:
                all_region_sizes.append(r["area"])
        
        # Summary statistics
        results["summary"] = {
            "total_images": len(results["images"]),
            "total_pixels": total_pixels,
            "total_crosswalk_pixels": total_crosswalk_pixels,
            "overall_crosswalk_percentage": total_crosswalk_pixels / total_pixels * 100 if total_pixels > 0 else 0,
            "total_crosswalk_regions": total_regions,
            "avg_regions_per_image": total_regions / len(results["images"]) if results["images"] else 0
        }
        
        if all_region_sizes:
            results["summary"]["region_size_stats"] = {
                "mean": float(np.mean(all_region_sizes)),
                "median": float(np.median(all_region_sizes)),
                "std": float(np.std(all_region_sizes)),
                "min": int(min(all_region_sizes)),
                "max": int(max(all_region_sizes))
            }
        
        # Save results
        output_path = self.output_dir / "pixel_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved to: {output_path}")
        
        return results


def analyze_all_locations():
    """Run Stage A analysis on all locations."""
    locations = ["financial_district", "central_park_south", "bay_ridge", "downtown_brooklyn"]
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    logger.info("\n" + "=" * 60)
    logger.info("STAGE A: PIXEL-LEVEL ANALYSIS")
    logger.info("=" * 60 + "\n")
    
    for loc_id in locations:
        logger.info(f"\n{'='*50}")
        logger.info(f"Location: {loc_id}")
        logger.info('='*50)
        
        analyzer = PixelAnalyzer(loc_id)
        results = analyzer.run_analysis()
        all_results.append(results)
    
    # Save combined results
    combined_path = RESULTS_DIR / "all_locations_pixel_analysis.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 80)
    print("STAGE A SUMMARY")
    print("=" * 80)
    print(f"{'Location':<25} {'Images':>10} {'Crosswalk %':>15} {'Regions':>12}")
    print("-" * 80)
    
    for r in all_results:
        if 'error' in r:
            print(f"{r['location_id']:<25} ERROR: {r['error']}")
        else:
            s = r['summary']
            print(f"{r['location_id']:<25} {s['total_images']:>10} "
                  f"{s['overall_crosswalk_percentage']:>14.2f}% "
                  f"{s['total_crosswalk_regions']:>12}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Stage A Pixel Analysis")
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--location', '-l', type=str)
    
    args = parser.parse_args()
    
    if args.all:
        analyze_all_locations()
    elif args.location:
        analyzer = PixelAnalyzer(args.location)
        results = analyzer.run_analysis()
        print(json.dumps(results["summary"], indent=2))
    else:
        print("Usage: python stage_a_pixel_analysis.py --all")


if __name__ == "__main__":
    main()
