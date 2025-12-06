#!/usr/bin/env python3
"""
CrossCheck NYC - Complete Analysis Pipeline
Locations: financial_district, east_village, bay_ridge, downtown_brooklyn, kew_gardens
NO WYANDANCH
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
from PIL import Image
from shapely.geometry import box
from shapely.ops import unary_union
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("./data")
OUTPUT_DIR = DATA_DIR / "outputs"
REFERENCE_DIR = DATA_DIR / "reference"
RESULTS_DIR = DATA_DIR / "results"
VIZ_DIR = RESULTS_DIR / "visualizations"
METRICS_DIR = RESULTS_DIR / "metrics"

LOCATIONS = ["financial_district", "east_village", "bay_ridge", "downtown_brooklyn", "kew_gardens"]

SEG_COLORS = {
    'crosswalk': (255, 0, 0),
    'sidewalk': (0, 0, 255),
    'road': (0, 255, 0),
    'background': (0, 0, 0)
}
COLOR_TOLERANCE = 40
MIN_CROSSWALK_PIXELS = 50
MIN_ROAD_PIXELS = 500


def find_locations():
    if not OUTPUT_DIR.exists():
        return []
    return sorted([d.name for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name in LOCATIONS])


def find_sidebside_images(location_id):
    base = OUTPUT_DIR / location_id / "segmentation"
    if not base.exists():
        return []
    return sorted(list(base.rglob("sidebside_*.png")))


def split_sidebside(img_path):
    img = np.array(Image.open(img_path))
    h, w = img.shape[:2]
    return img[:, :w//2], img[:, w//2:]


def detect_class_mask(seg, class_name):
    if len(seg.shape) != 3:
        return np.zeros(seg.shape[:2], dtype=bool)
    target = np.array(SEG_COLORS.get(class_name, (0, 0, 0)))
    diff = np.abs(seg.astype(float) - target)
    return np.all(diff < COLOR_TOLERANCE, axis=2)


def is_valid_road_tile(seg):
    road = detect_class_mask(seg, 'road')
    sidewalk = detect_class_mask(seg, 'sidewalk')
    return (road.sum() + sidewalk.sum()) >= MIN_ROAD_PIXELS


def has_crosswalks(seg):
    return detect_class_mask(seg, 'crosswalk').sum() >= MIN_CROSSWALK_PIXELS


def load_osm_crossings():
    for f in [REFERENCE_DIR / "osm_crossings_nyc.geojson", REFERENCE_DIR / "combined_reference.geojson"]:
        if f.exists():
            try:
                return gpd.read_file(f)
            except:
                pass
    return None


def load_polygon_predictions(location_id):
    poly_dir = OUTPUT_DIR / location_id / "polygons"
    if not poly_dir.exists():
        return None
    for shp in poly_dir.rglob("*.shp"):
        try:
            return gpd.read_file(shp)
        except:
            pass
    return None


class StageA_PixelAnalysis:
    def __init__(self, location_id):
        self.location_id = location_id
        self.images = find_sidebside_images(location_id)
        self.results = {'location_id': location_id, 'stage': 'A', 'metrics': {}, 'rq_answers': {}}
        self.valid_tiles = []
        self.tiles_with_crosswalks = []
        self.tiles_without_crosswalks = []
    
    def analyze_all_images(self):
        logger.info(f"Stage A: Analyzing {len(self.images)} images for {self.location_id}")
        if not self.images:
            self.results['error'] = 'No segmentation images'
            return self.results
        
        class_totals = {'crosswalk': 0, 'sidewalk': 0, 'road': 0, 'background': 0}
        region_sizes = []
        
        for img_path in self.images:
            try:
                orig, seg = split_sidebside(img_path)
            except:
                continue
            if not is_valid_road_tile(seg):
                continue
            
            tile = {'path': str(img_path), 'filename': img_path.name}
            for cn in SEG_COLORS:
                mask = detect_class_mask(seg, cn)
                px = int(mask.sum())
                class_totals[cn] += px
                tile[f'{cn}_pixels'] = px
            
            cw_mask = detect_class_mask(seg, 'crosswalk')
            labeled, num = ndimage.label(cw_mask)
            tile['crosswalk_regions'] = num
            for i in range(1, num + 1):
                region_sizes.append(int((labeled == i).sum()))
            
            self.valid_tiles.append(tile)
            if has_crosswalks(seg):
                self.tiles_with_crosswalks.append(tile)
            else:
                self.tiles_without_crosswalks.append(tile)
        
        total = sum(class_totals.values())
        self.results['metrics'] = {
            'total_valid_tiles': len(self.valid_tiles),
            'tiles_with_crosswalks': len(self.tiles_with_crosswalks),
            'tiles_without_crosswalks': len(self.tiles_without_crosswalks),
            'crosswalk_detection_rate': len(self.tiles_with_crosswalks) / len(self.valid_tiles) if self.valid_tiles else 0,
            'class_distribution': {k: {'pixels': v, 'pct': round(v/total*100, 2) if total else 0} for k, v in class_totals.items()},
            'crosswalk_region_stats': {
                'total': len(region_sizes),
                'mean': float(np.mean(region_sizes)) if region_sizes else 0,
                'median': float(np.median(region_sizes)) if region_sizes else 0,
                'min': int(min(region_sizes)) if region_sizes else 0,
                'max': int(max(region_sizes)) if region_sizes else 0
            }
        }
        return self.results
    
    def analyze_threshold_effects(self, n=10):
        logger.info("Analyzing threshold effects (RQ2)")
        if not self.tiles_with_crosswalks:
            self.results['rq_answers']['RQ2'] = {'error': 'No tiles'}
            return
        
        sample = self.tiles_with_crosswalks[:n]
        avg = {}
        
        for tile in sample:
            try:
                _, seg = split_sidebside(Path(tile['path']))
                mask = detect_class_mask(seg, 'crosswalk').astype(np.uint8)
                orig = int(mask.sum())
                if orig == 0:
                    continue
                
                ops = {
                    'erode1': int(binary_erosion(mask, iterations=1).sum()),
                    'erode2': int(binary_erosion(mask, iterations=2).sum()),
                    'dilate1': int(binary_dilation(mask, iterations=1).sum()),
                    'dilate2': int(binary_dilation(mask, iterations=2).sum()),
                    'opening': int(binary_opening(mask).sum()),
                    'closing': int(binary_closing(mask).sum())
                }
                
                for k, v in ops.items():
                    pct = (v - orig) / orig * 100
                    if k not in avg:
                        avg[k] = []
                    avg[k].append(pct)
            except:
                pass
        
        self.results['rq_answers']['RQ2'] = {
            'desc': 'Morphology effects on crosswalk pixels',
            'avg_pct_change': {k: round(float(np.mean(v)), 2) for k, v in avg.items()}
        }
    
    def create_visualizations(self):
        out_dir = VIZ_DIR / self.location_id / "stage_a"
        out_dir.mkdir(parents=True, exist_ok=True)
        files = []
        
        # Mixed gallery
        with_cw = self.tiles_with_crosswalks[:12]
        without_cw = self.tiles_without_crosswalks[:4]
        all_t = with_cw + without_cw
        if all_t:
            n = len(all_t)
            cols = 4
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
            axes = axes.flatten() if n > 1 else [axes]
            for i, t in enumerate(all_t):
                try:
                    img = np.array(Image.open(t['path']))
                    axes[i].imshow(img)
                    has = t.get('crosswalk_pixels', 0) >= MIN_CROSSWALK_PIXELS
                    axes[i].set_title(f"{t['filename']}\n{'✓' if has else '✗'}", fontsize=8, color='green' if has else 'red')
                except:
                    pass
                axes[i].axis('off')
            for i in range(n, len(axes)):
                axes[i].axis('off')
            plt.suptitle(f"{self.location_id} - Mixed Gallery", fontsize=14)
            plt.tight_layout()
            p = out_dir / "mixed_gallery.png"
            plt.savefig(p, dpi=150)
            plt.close()
            files.append(p)
        
        # Segmentation breakdown
        samples = self.tiles_with_crosswalks[:5]
        if samples:
            fig, axes = plt.subplots(len(samples), 5, figsize=(25, 5*len(samples)))
            if len(samples) == 1:
                axes = axes.reshape(1, -1)
            for i, t in enumerate(samples):
                try:
                    orig, seg = split_sidebside(Path(t['path']))
                    axes[i,0].imshow(orig); axes[i,0].set_title("Original"); axes[i,0].axis('off')
                    axes[i,1].imshow(seg); axes[i,1].set_title("Segmentation"); axes[i,1].axis('off')
                    cw = detect_class_mask(seg, 'crosswalk')
                    axes[i,2].imshow(cw, cmap='Reds'); axes[i,2].set_title(f"Crosswalks ({cw.sum()})"); axes[i,2].axis('off')
                    ctx = np.zeros((*cw.shape, 3), dtype=np.uint8)
                    ctx[detect_class_mask(seg, 'road')] = [0,255,0]
                    ctx[detect_class_mask(seg, 'sidewalk')] = [0,0,255]
                    ctx[cw] = [255,0,0]
                    axes[i,3].imshow(ctx); axes[i,3].set_title("Context"); axes[i,3].axis('off')
                    ov = orig.copy(); ov[cw] = [255,0,0]
                    axes[i,4].imshow(ov); axes[i,4].set_title("Overlay"); axes[i,4].axis('off')
                except:
                    for j in range(5):
                        axes[i,j].axis('off')
            plt.suptitle(f"{self.location_id} - Breakdown (RQ1, RQ5)", fontsize=14)
            plt.tight_layout()
            p = out_dir / "breakdown.png"
            plt.savefig(p, dpi=150)
            plt.close()
            files.append(p)
        
        return files


class StageB_NetworkAnalysis:
    def __init__(self, location_id):
        self.location_id = location_id
        self.predictions = load_polygon_predictions(location_id)
        self.ground_truth = load_osm_crossings()
        self.crosswalks = None
        self.gt_local = None
        self.results = {'location_id': location_id, 'stage': 'B', 'metrics': {}, 'rq_answers': {}}
    
    def analyze_predictions(self):
        logger.info(f"Stage B: {self.location_id}")
        if self.predictions is None or len(self.predictions) == 0:
            self.results['error'] = 'No polygons'
            return self.results
        
        class_col = None
        for c in ['f_type', 'class', 'Class', 'type']:
            if c in self.predictions.columns:
                class_col = c
                break
        
        if class_col:
            mask = self.predictions[class_col].astype(str).str.lower().str.contains('crosswalk|crossing', na=False)
            self.crosswalks = self.predictions[mask].copy()
        else:
            self.crosswalks = self.predictions.copy()
        
        if len(self.crosswalks) == 0:
            self.results['metrics']['crosswalk_stats'] = {'count': 0}
            return self.results
        
        try:
            proj = self.crosswalks.to_crs(epsg=32618)
            areas = proj.geometry.area
        except:
            areas = self.crosswalks.geometry.area * 111000 * 111000
        
        self.results['metrics']['crosswalk_stats'] = {
            'count': len(self.crosswalks),
            'total_sqm': float(areas.sum()),
            'mean_sqm': float(areas.mean()),
            'median_sqm': float(areas.median())
        }
        return self.results
    
    def analyze_ground_truth(self, buffer_m=15):
        logger.info("Ground truth comparison (RQ4)")
        if self.ground_truth is None:
            self.results['rq_answers']['RQ4'] = {'error': 'No ground truth'}
            return
        if self.crosswalks is None or len(self.crosswalks) == 0:
            self.results['rq_answers']['RQ4'] = {'error': 'No predictions'}
            return
        
        bounds = self.crosswalks.total_bounds
        try:
            gt = self.ground_truth.to_crs(self.crosswalks.crs)
        except:
            gt = self.ground_truth
        
        buf = 0.01
        bbox = box(bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf)
        self.gt_local = gt[gt.geometry.intersects(bbox)]
        
        if len(self.gt_local) == 0:
            self.results['rq_answers']['RQ4'] = {'status': 'no_overlap', 'preds': len(self.crosswalks), 'gt': 0}
            return
        
        buf_deg = buffer_m / 111000
        pred_c = self.crosswalks.geometry.centroid
        gt_pts = self.gt_local.geometry.centroid if self.gt_local.geometry.geom_type.iloc[0] != 'Point' else self.gt_local.geometry
        
        gt_buf = unary_union(gt_pts.buffer(buf_deg))
        pred_buf = unary_union(pred_c.buffer(buf_deg))
        
        tp = sum(1 for c in pred_c if c.within(gt_buf))
        matched = sum(1 for p in gt_pts if p.within(pred_buf))
        
        n_pred, n_gt = len(self.crosswalks), len(self.gt_local)
        prec = tp / n_pred if n_pred else 0
        rec = matched / n_gt if n_gt else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        
        self.results['rq_answers']['RQ4'] = {
            'predictions': n_pred, 'ground_truth': n_gt,
            'tp': tp, 'fp': n_pred - tp, 'fn': n_gt - matched,
            'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4)
        }
    
    def analyze_placement(self):
        logger.info("Placement analysis (RQ3)")
        if self.crosswalks is None or len(self.crosswalks) == 0:
            self.results['rq_answers']['RQ3'] = {'error': 'No predictions'}
            return
        if self.predictions is None:
            return
        
        class_col = None
        for c in ['f_type', 'class', 'Class', 'type']:
            if c in self.predictions.columns:
                class_col = c
                break
        if not class_col:
            return
        
        sw = self.predictions[self.predictions[class_col].astype(str).str.lower().str.contains('sidewalk', na=False)]
        rd = self.predictions[self.predictions[class_col].astype(str).str.lower().str.contains('road', na=False)]
        
        cw_c = self.crosswalks.geometry.centroid
        buf = 0.0001
        
        near_sw = sum(1 for c in cw_c if len(sw) and c.within(unary_union(sw.geometry.buffer(buf)))) if len(sw) else 0
        near_rd = sum(1 for c in cw_c if len(rd) and c.within(unary_union(rd.geometry.buffer(buf)))) if len(rd) else 0
        
        n = len(self.crosswalks)
        self.results['rq_answers']['RQ3'] = {
            'total': n,
            'near_sidewalk': near_sw, 'near_sidewalk_pct': round(near_sw/n*100, 2) if n else 0,
            'near_road': near_rd, 'near_road_pct': round(near_rd/n*100, 2) if n else 0
        }
    
    def identify_edge_cases(self):
        logger.info("Edge cases (RQ7)")
        if self.crosswalks is None or len(self.crosswalks) == 0:
            self.results['rq_answers']['RQ7'] = {'error': 'No predictions'}
            return
        
        try:
            proj = self.crosswalks.to_crs(epsg=32618)
            areas = proj.geometry.area
        except:
            areas = self.crosswalks.geometry.area * 111000 * 111000
        
        mean, std = areas.mean(), areas.std() if len(areas) > 1 else 0
        small = (areas < max(0, mean - 2*std)).sum()
        large = (areas > mean + 2*std).sum()
        
        self.results['rq_answers']['RQ7'] = {
            'total': len(self.crosswalks),
            'very_small': int(small),
            'very_large': int(large),
            'typical': int(len(self.crosswalks) - small - large)
        }
    
    def create_visualizations(self):
        out_dir = VIZ_DIR / self.location_id / "stage_b"
        out_dir.mkdir(parents=True, exist_ok=True)
        files = []
        
        if self.crosswalks is None or len(self.crosswalks) == 0:
            return files
        
        # Ground truth map
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        self.crosswalks.plot(ax=axes[0], color='red', alpha=0.6)
        axes[0].set_title(f"Predictions ({len(self.crosswalks)})")
        
        if self.gt_local is not None and len(self.gt_local) > 0:
            self.gt_local.plot(ax=axes[1], color='blue', alpha=0.6, markersize=20)
            axes[1].set_title(f"Ground Truth ({len(self.gt_local)})")
        else:
            axes[1].text(0.5, 0.5, 'No GT', ha='center', transform=axes[1].transAxes)
        
        self.crosswalks.plot(ax=axes[2], color='red', alpha=0.5, label='Pred')
        if self.gt_local is not None and len(self.gt_local) > 0:
            self.gt_local.plot(ax=axes[2], color='blue', alpha=0.7, markersize=30, label='GT')
        axes[2].legend()
        axes[2].set_title("Overlay (RQ4)")
        
        plt.suptitle(f"{self.location_id} - Ground Truth Comparison", fontsize=14)
        plt.tight_layout()
        p = out_dir / "gt_comparison.png"
        plt.savefig(p, dpi=150)
        plt.close()
        files.append(p)
        
        # Error analysis
        rq4 = self.results.get('rq_answers', {}).get('RQ4', {})
        if 'precision' in rq4:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].bar(['TP', 'FP', 'FN'], [rq4['tp'], rq4['fp'], rq4['fn']], color=['green', 'orange', 'red'])
            axes[0].set_title('Error Breakdown')
            axes[1].bar(['Precision', 'Recall', 'F1'], [rq4['precision'], rq4['recall'], rq4['f1']], color=['steelblue', 'darkorange', 'green'])
            axes[1].set_ylim(0, 1)
            axes[1].set_title('Metrics (RQ4)')
            plt.suptitle(f"{self.location_id} - Error Analysis", fontsize=14)
            plt.tight_layout()
            p = out_dir / "error_analysis.png"
            plt.savefig(p, dpi=150)
            plt.close()
            files.append(p)
        
        return files


def run_complete_analysis():
    print("\n" + "="*70)
    print("   CROSSCHECK NYC - COMPLETE ANALYSIS")
    print("   Locations:", LOCATIONS)
    print("="*70 + "\n")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    locs = find_locations()
    if not locs:
        print("ERROR: No locations found")
        return
    
    print(f"Found: {locs}\n")
    
    all_results = {'stage_a': {}, 'stage_b': {}}
    
    for loc in locs:
        print(f"\n{'='*50}\n{loc}\n{'='*50}")
        
        # Stage A
        print("\n--- STAGE A ---")
        a = StageA_PixelAnalysis(loc)
        a.analyze_all_images()
        a.analyze_threshold_effects()
        a_viz = a.create_visualizations()
        all_results['stage_a'][loc] = a.results
        m = a.results.get('metrics', {})
        print(f"  Tiles: {m.get('total_valid_tiles', 0)}, With CW: {m.get('tiles_with_crosswalks', 0)}")
        
        # Stage B
        print("\n--- STAGE B ---")
        b = StageB_NetworkAnalysis(loc)
        b.analyze_predictions()
        b.analyze_ground_truth()
        b.analyze_placement()
        b.identify_edge_cases()
        b_viz = b.create_visualizations()
        all_results['stage_b'][loc] = b.results
        cw = b.results.get('metrics', {}).get('crosswalk_stats', {})
        print(f"  Polygons: {cw.get('count', 0)}")
        rq4 = b.results.get('rq_answers', {}).get('RQ4', {})
        if 'f1' in rq4:
            print(f"  F1: {rq4['f1']}")
    
    # Summary
    print("\n" + "="*70)
    print("RQ4 SUMMARY")
    print("-"*60)
    print(f"{'Location':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*60)
    for loc in locs:
        rq4 = all_results['stage_b'].get(loc, {}).get('rq_answers', {}).get('RQ4', {})
        if 'precision' in rq4:
            print(f"{loc:<25} {rq4['precision']:>10.3f} {rq4['recall']:>10.3f} {rq4['f1']:>10.3f}")
        else:
            print(f"{loc:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    # Save
    with open(METRICS_DIR / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Saved to {METRICS_DIR}/")
    print(f"✓ Visualizations in {VIZ_DIR}/")
    return all_results


if __name__ == "__main__":
    run_complete_analysis()
