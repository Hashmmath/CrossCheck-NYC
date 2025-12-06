#!/usr/bin/env python3
"""
CrossCheck NYC - Stage B: Network & Spatial Analysis
=====================================================
Analyze crosswalk predictions at network/spatial level.

Research Questions Addressed:
- RQ3: Are crosswalks in logical locations?
- RQ4: How well do results match reference data?
- RQ7: What confuses the model?

Usage:
    python stage_b_network_analysis.py --all
    python stage_b_network_analysis.py --location financial_district
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import logging

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "stage_b"
FIGURES_DIR = PROJECT_ROOT / "data" / "results" / "figures" / "comparison_maps"


class NetworkAnalyzer:
    """Network-level analysis of crosswalk predictions."""
    
    def __init__(self, location_id: str):
        self.location_id = location_id
        self.project_dir = OUTPUT_DIR / location_id
        self.output_dir = RESULTS_DIR / location_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.crosswalks = self._load_polygons("crosswalk")
        self.sidewalks = self._load_polygons("sidewalk")
        self.roads = self._load_polygons("road")
        
    def _load_polygons(self, poly_type: str) -> gpd.GeoDataFrame:
        """Load polygon file."""
        path = self.project_dir / "polygons" / f"{poly_type}.geojson"
        if path.exists():
            return gpd.read_file(path)
        return None
    
    def analyze_topology(self) -> Dict:
        """Analyze topology of crosswalk detections."""
        if self.crosswalks is None or len(self.crosswalks) == 0:
            return {"error": "No crosswalk data"}
        
        results = {
            "total_crosswalks": len(self.crosswalks),
            "geometry_types": self.crosswalks.geometry.geom_type.value_counts().to_dict()
        }
        
        # Area statistics (convert to approx sqm)
        areas = self.crosswalks.geometry.area * 111000**2
        results["area_stats"] = {
            "mean_sqm": float(areas.mean()),
            "median_sqm": float(areas.median()),
            "std_sqm": float(areas.std()),
            "min_sqm": float(areas.min()),
            "max_sqm": float(areas.max()),
            "total_sqm": float(areas.sum())
        }
        
        # Connectivity via proximity graph
        threshold = 0.0001  # ~11m
        G = nx.Graph()
        
        for idx in self.crosswalks.index:
            G.add_node(idx)
        
        for i, row_i in self.crosswalks.iterrows():
            for j, row_j in self.crosswalks.iterrows():
                if i < j:
                    dist = row_i.geometry.distance(row_j.geometry)
                    if dist < threshold:
                        G.add_edge(i, j)
        
        results["connectivity"] = {
            "num_components": nx.number_connected_components(G),
            "isolated_crosswalks": sum(1 for n in G.nodes() if G.degree(n) == 0),
            "largest_component": len(max(nx.connected_components(G), key=len)) if G.nodes() else 0
        }
        
        return results
    
    def analyze_spatial_context(self) -> Dict:
        """Check if crosswalks are in logical locations."""
        if self.crosswalks is None:
            return {"error": "No crosswalk data"}
        
        results = {
            "near_roads": 0,
            "near_sidewalks": 0,
            "connecting_both": 0,
            "isolated": 0
        }
        
        road_union = unary_union(self.roads.geometry) if self.roads is not None and len(self.roads) > 0 else None
        sidewalk_union = unary_union(self.sidewalks.geometry) if self.sidewalks is not None and len(self.sidewalks) > 0 else None
        
        for idx, row in self.crosswalks.iterrows():
            near_road = False
            near_sidewalk = False
            
            if road_union is not None:
                if row.geometry.distance(road_union) < 0.00005:  # ~5m
                    near_road = True
                    results["near_roads"] += 1
            
            if sidewalk_union is not None:
                if row.geometry.distance(sidewalk_union) < 0.00005:
                    near_sidewalk = True
                    results["near_sidewalks"] += 1
            
            if near_road and near_sidewalk:
                results["connecting_both"] += 1
            elif not near_road and not near_sidewalk:
                results["isolated"] += 1
        
        # Percentages
        total = len(self.crosswalks)
        if total > 0:
            results["pct_near_roads"] = results["near_roads"] / total * 100
            results["pct_near_sidewalks"] = results["near_sidewalks"] / total * 100
            results["pct_connecting_both"] = results["connecting_both"] / total * 100
            results["pct_isolated"] = results["isolated"] / total * 100
        
        return results
    
    def identify_edge_cases(self) -> Dict:
        """Identify unusual detections."""
        if self.crosswalks is None:
            return {"error": "No crosswalk data"}
        
        areas = self.crosswalks.geometry.area * 111000**2
        mean, std = areas.mean(), areas.std()
        
        edge_cases = {
            "very_small": {
                "count": int((areas < mean - 2*std).sum()),
                "threshold_sqm": float(mean - 2*std),
                "indices": self.crosswalks[areas < mean - 2*std].index.tolist()
            },
            "very_large": {
                "count": int((areas > mean + 2*std).sum()),
                "threshold_sqm": float(mean + 2*std),
                "indices": self.crosswalks[areas > mean + 2*std].index.tolist()
            }
        }
        
        # Check for unusual shapes
        unusual_shapes = []
        for idx, row in self.crosswalks.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Polygon':
                hull_area = geom.convex_hull.area
                if hull_area > 0:
                    solidity = geom.area / hull_area
                    if solidity < 0.5:
                        unusual_shapes.append(idx)
        
        edge_cases["unusual_shape"] = {
            "count": len(unusual_shapes),
            "indices": unusual_shapes
        }
        
        return edge_cases
    
    def create_map(self, save_path: Path):
        """Create comparison map."""
        if self.crosswalks is None:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Roads
        if self.roads is not None and len(self.roads) > 0:
            self.roads.plot(ax=ax, color='lightgray', alpha=0.5, label='Roads')
        
        # Sidewalks
        if self.sidewalks is not None and len(self.sidewalks) > 0:
            self.sidewalks.plot(ax=ax, color='lightblue', alpha=0.5, label='Sidewalks')
        
        # Crosswalks
        self.crosswalks.plot(ax=ax, color='red', alpha=0.7, edgecolor='darkred', label='Crosswalks')
        
        ax.set_title(f"CrossCheck NYC - {self.location_id}")
        ax.legend()
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self) -> Dict:
        """Run full Stage B analysis."""
        results = {
            "location_id": self.location_id,
            "topology": self.analyze_topology(),
            "spatial_context": self.analyze_spatial_context(),
            "edge_cases": self.identify_edge_cases()
        }
        
        # Create visualization
        fig_dir = FIGURES_DIR
        fig_dir.mkdir(parents=True, exist_ok=True)
        map_path = fig_dir / f"{self.location_id}_map.png"
        self.create_map(map_path)
        results["map_path"] = str(map_path)
        
        # Save results
        output_path = self.output_dir / "network_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results


def cross_location_comparison(results: List[Dict]) -> Dict:
    """Compare results across all locations."""
    comparison = {
        "locations": {},
        "rankings": {}
    }
    
    for r in results:
        loc_id = r["location_id"]
        if "error" in r.get("topology", {}):
            continue
            
        comparison["locations"][loc_id] = {
            "crosswalk_count": r["topology"].get("total_crosswalks", 0),
            "mean_area": r["topology"].get("area_stats", {}).get("mean_sqm", 0),
            "pct_near_roads": r["spatial_context"].get("pct_near_roads", 0),
            "pct_isolated": r["spatial_context"].get("pct_isolated", 0),
            "edge_cases": r["edge_cases"].get("very_small", {}).get("count", 0) + 
                          r["edge_cases"].get("very_large", {}).get("count", 0)
        }
    
    # Rankings
    if comparison["locations"]:
        df = pd.DataFrame(comparison["locations"]).T
        comparison["rankings"] = {
            "most_crosswalks": df["crosswalk_count"].idxmax(),
            "highest_road_proximity": df["pct_near_roads"].idxmax(),
            "most_edge_cases": df["edge_cases"].idxmax()
        }
    
    return comparison


def analyze_all_locations():
    """Run Stage B analysis on all locations."""
    locations = ["financial_district", "central_park_south", "bay_ridge", "downtown_brooklyn"]
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    logger.info("\n" + "=" * 60)
    logger.info("STAGE B: NETWORK & SPATIAL ANALYSIS")
    logger.info("=" * 60 + "\n")
    
    for loc_id in locations:
        logger.info(f"\nAnalyzing: {loc_id}")
        logger.info("-" * 40)
        
        analyzer = NetworkAnalyzer(loc_id)
        results = analyzer.run_analysis()
        all_results.append(results)
    
    # Cross-location comparison
    comparison = cross_location_comparison(all_results)
    
    # Save all results
    combined_path = RESULTS_DIR / "all_locations_network_analysis.json"
    with open(combined_path, 'w') as f:
        json.dump({"locations": all_results, "comparison": comparison}, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 90)
    print("STAGE B SUMMARY")
    print("=" * 90)
    print(f"{'Location':<25} {'Count':>10} {'Avg Area':>12} {'Near Roads':>12} {'Isolated':>12}")
    print("-" * 90)
    
    for r in all_results:
        loc = r["location_id"]
        if "error" in r.get("topology", {}):
            print(f"{loc:<25} {'NO DATA':>10}")
        else:
            t = r["topology"]
            s = r["spatial_context"]
            print(f"{loc:<25} {t.get('total_crosswalks', 0):>10} "
                  f"{t.get('area_stats', {}).get('mean_sqm', 0):>11.1f}m² "
                  f"{s.get('pct_near_roads', 0):>11.1f}% "
                  f"{s.get('pct_isolated', 0):>11.1f}%")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Stage B Network Analysis")
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--location', '-l', type=str)
    
    args = parser.parse_args()
    
    if args.all:
        analyze_all_locations()
    elif args.location:
        analyzer = NetworkAnalyzer(args.location)
        results = analyzer.run_analysis()
        print(json.dumps(results, indent=2, default=str))
    else:
        print("Usage: python stage_b_network_analysis.py --all")


if __name__ == "__main__":
    main()
