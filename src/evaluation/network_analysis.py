"""
Stage B: Network/Vector-Level Evaluation
=========================================

Converts pixel-level predictions to vector networks and evaluates:
1. Topology quality (connectivity, dead-ends, stubs)
2. Geometry quality (curvature, zigzags)
3. Agreement with authoritative data (NYC GIS / OSM)

This module implements the A→B linkage showing how pixel-level
calibration choices affect network-level quality.

Usage:
    python network_analysis.py --input data/processed/predictions --output outputs/network
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString, box
from shapely.ops import linemerge, unary_union, split
from scipy import ndimage
from skimage.morphology import skeletonize, thin
from skimage.graph import route_through_array
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Part 1: Vectorization (Mask → Skeleton → Graph)
# =============================================================================

class MaskVectorizer:
    """
    Convert binary segmentation mask to vector network.
    
    Pipeline:
    1. Skeletonization (morphological thinning)
    2. Centerline extraction
    3. Simplification (Douglas-Peucker)
    4. Split at intersections
    5. Build graph with nodes and edges
    """
    
    def __init__(
        self,
        simplify_tolerance: float = 0.5,  # meters
        min_edge_length: float = 3.0,     # meters
        pixel_size: float = 0.3           # meters per pixel at zoom 19
    ):
        """
        Initialize vectorizer.
        
        Args:
            simplify_tolerance: Douglas-Peucker simplification tolerance (meters)
            min_edge_length: Minimum edge length to keep (meters)
            pixel_size: Ground sample distance (meters per pixel)
        """
        self.simplify_tolerance = simplify_tolerance
        self.min_edge_length = min_edge_length
        self.pixel_size = pixel_size
    
    def vectorize(
        self,
        mask: np.ndarray,
        transform: Optional[Affine] = None,
        crs: Optional[str] = None
    ) -> Tuple[gpd.GeoDataFrame, nx.Graph]:
        """
        Convert binary mask to vector network.
        
        Args:
            mask: Binary mask (H, W) with 1=feature, 0=background
            transform: Affine transform for georeferencing
            crs: Coordinate reference system
            
        Returns:
            edges_gdf: GeoDataFrame with edge geometries and attributes
            graph: NetworkX graph representation
        """
        # Step 1: Skeletonization
        logger.info("Step 1: Skeletonizing mask...")
        skeleton = self._skeletonize(mask)
        
        # Step 2: Extract centerlines as polylines
        logger.info("Step 2: Extracting centerlines...")
        lines = self._extract_lines(skeleton, transform)
        
        if len(lines) == 0:
            logger.warning("No lines extracted from skeleton")
            return gpd.GeoDataFrame(), nx.Graph()
        
        # Step 3: Simplify
        logger.info("Step 3: Simplifying lines...")
        lines = self._simplify_lines(lines)
        
        # Step 4: Build graph
        logger.info("Step 4: Building network graph...")
        graph, edges_gdf = self._build_graph(lines, crs)
        
        # Step 5: Remove short stubs
        logger.info("Step 5: Removing short stubs...")
        graph, edges_gdf = self._remove_stubs(graph, edges_gdf)
        
        logger.info(f"Vectorization complete: {len(edges_gdf)} edges, {graph.number_of_nodes()} nodes")
        
        return edges_gdf, graph
    
    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological skeletonization."""
        # Ensure binary
        binary = (mask > 0).astype(np.uint8)
        
        # Skeletonize
        skeleton = skeletonize(binary).astype(np.uint8)
        
        return skeleton
    
    def _extract_lines(
        self,
        skeleton: np.ndarray,
        transform: Optional[Affine] = None
    ) -> List[LineString]:
        """
        Extract line geometries from skeleton.
        
        Uses connected component analysis and contour tracing.
        """
        lines = []
        
        # Label connected components
        labeled, num_features = ndimage.label(skeleton)
        
        for i in range(1, num_features + 1):
            component = (labeled == i).astype(np.uint8)
            
            # Get coordinates of skeleton pixels
            coords = np.argwhere(component > 0)
            
            if len(coords) < 2:
                continue
            
            # Order points along the skeleton
            ordered_coords = self._order_skeleton_points(coords)
            
            if len(ordered_coords) < 2:
                continue
            
            # Convert pixel coords to geographic coords
            if transform is not None:
                geo_coords = [
                    transform * (c[1], c[0])  # col, row -> x, y
                    for c in ordered_coords
                ]
            else:
                geo_coords = [(c[1] * self.pixel_size, c[0] * self.pixel_size) 
                             for c in ordered_coords]
            
            line = LineString(geo_coords)
            if line.is_valid and line.length > 0:
                lines.append(line)
        
        return lines
    
    def _order_skeleton_points(self, coords: np.ndarray) -> List[Tuple[int, int]]:
        """
        Order skeleton points into a traversable sequence.
        
        Uses nearest neighbor traversal from an endpoint.
        """
        if len(coords) < 2:
            return []
        
        # Find endpoints (pixels with only 1 neighbor)
        # or start from any point if no clear endpoints
        point_set = set(map(tuple, coords))
        
        def count_neighbors(p):
            count = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (p[0] + dr, p[1] + dc)
                    if neighbor in point_set:
                        count += 1
            return count
        
        # Find endpoints (degree 1) or use first point
        endpoints = [p for p in point_set if count_neighbors(p) == 1]
        start = endpoints[0] if endpoints else coords[0]
        
        # Traverse using nearest neighbor
        ordered = [tuple(start)]
        remaining = point_set - {tuple(start)}
        
        while remaining:
            current = ordered[-1]
            
            # Find nearest unvisited neighbor
            best_dist = float('inf')
            best_neighbor = None
            
            for p in remaining:
                dist = (p[0] - current[0])**2 + (p[1] - current[1])**2
                if dist < best_dist:
                    best_dist = dist
                    best_neighbor = p
            
            if best_neighbor is None or best_dist > 8:  # max 2*sqrt(2) for 8-connected
                break
            
            ordered.append(best_neighbor)
            remaining.remove(best_neighbor)
        
        return ordered
    
    def _simplify_lines(self, lines: List[LineString]) -> List[LineString]:
        """Apply Douglas-Peucker simplification."""
        simplified = []
        
        for line in lines:
            simp_line = line.simplify(self.simplify_tolerance, preserve_topology=True)
            if simp_line.is_valid and simp_line.length > 0:
                simplified.append(simp_line)
        
        return simplified
    
    def _build_graph(
        self,
        lines: List[LineString],
        crs: Optional[str] = None
    ) -> Tuple[nx.Graph, gpd.GeoDataFrame]:
        """
        Build network graph from line geometries.
        
        Nodes at endpoints and intersections.
        Edges as line geometries with length attributes.
        """
        G = nx.Graph()
        edges_data = []
        
        # Merge intersecting lines
        merged = linemerge(lines)
        
        if isinstance(merged, LineString):
            lines = [merged]
        elif isinstance(merged, MultiLineString):
            lines = list(merged.geoms)
        else:
            lines = [merged]
        
        # Build graph
        node_id = 0
        edge_id = 0
        point_to_node = {}
        
        def get_or_create_node(point):
            nonlocal node_id
            # Round coordinates for matching
            key = (round(point.x, 6), round(point.y, 6))
            if key not in point_to_node:
                point_to_node[key] = node_id
                G.add_node(node_id, x=point.x, y=point.y, geometry=point)
                node_id += 1
            return point_to_node[key]
        
        for line in lines:
            if not isinstance(line, LineString) or len(line.coords) < 2:
                continue
            
            start_point = Point(line.coords[0])
            end_point = Point(line.coords[-1])
            
            start_node = get_or_create_node(start_point)
            end_node = get_or_create_node(end_point)
            
            if start_node != end_node:
                G.add_edge(
                    start_node, 
                    end_node,
                    edge_id=edge_id,
                    geometry=line,
                    length=line.length,
                    curvature=self._compute_curvature(line)
                )
                
                edges_data.append({
                    'edge_id': edge_id,
                    'start_node': start_node,
                    'end_node': end_node,
                    'length': line.length,
                    'curvature': self._compute_curvature(line),
                    'geometry': line
                })
                
                edge_id += 1
        
        edges_gdf = gpd.GeoDataFrame(edges_data, crs=crs)
        
        return G, edges_gdf
    
    def _compute_curvature(self, line: LineString) -> float:
        """
        Compute average curvature of line.
        
        Returns ratio of straight-line distance to actual length.
        """
        if line.length == 0:
            return 0.0
        
        straight_dist = Point(line.coords[0]).distance(Point(line.coords[-1]))
        
        if line.length == 0:
            return 1.0
        
        return straight_dist / line.length  # 1.0 = perfectly straight
    
    def _remove_stubs(
        self,
        graph: nx.Graph,
        edges_gdf: gpd.GeoDataFrame
    ) -> Tuple[nx.Graph, gpd.GeoDataFrame]:
        """Remove edges shorter than min_edge_length that are dead-ends."""
        
        edges_to_remove = []
        
        for u, v, data in graph.edges(data=True):
            if data['length'] < self.min_edge_length:
                # Check if either endpoint is a dead-end (degree 1)
                if graph.degree(u) == 1 or graph.degree(v) == 1:
                    edges_to_remove.append((u, v, data['edge_id']))
        
        # Remove from graph
        for u, v, edge_id in edges_to_remove:
            graph.remove_edge(u, v)
            
            # Remove isolated nodes
            if graph.degree(u) == 0:
                graph.remove_node(u)
            if v in graph and graph.degree(v) == 0:
                graph.remove_node(v)
        
        # Remove from GeoDataFrame
        removed_ids = {e[2] for e in edges_to_remove}
        edges_gdf = edges_gdf[~edges_gdf['edge_id'].isin(removed_ids)].copy()
        
        logger.info(f"Removed {len(edges_to_remove)} short stubs")
        
        return graph, edges_gdf


# =============================================================================
# Part 2: Topology & Geometry Quality Analysis
# =============================================================================

class TopologyAnalyzer:
    """
    Analyze network topology and geometry quality.
    
    Metrics:
    - Connected components (count, size distribution)
    - Dead-ends (degree-1 nodes)
    - Short stubs (short edges at dead-ends)
    - Curvature/zigzags (high local angle changes)
    """
    
    def __init__(
        self,
        min_edge_length: float = 3.0,      # meters
        curvature_threshold: float = 0.7,   # ratio
        angle_threshold: float = 45.0       # degrees
    ):
        self.min_edge_length = min_edge_length
        self.curvature_threshold = curvature_threshold
        self.angle_threshold = angle_threshold
    
    def analyze(
        self,
        graph: nx.Graph,
        edges_gdf: gpd.GeoDataFrame
    ) -> Dict:
        """
        Run full topology analysis.
        
        Returns:
            Dict with all topology metrics
        """
        results = {}
        
        # Basic counts
        results['num_nodes'] = graph.number_of_nodes()
        results['num_edges'] = graph.number_of_edges()
        
        # Connected components
        results.update(self._analyze_components(graph))
        
        # Dead-ends
        results.update(self._analyze_dead_ends(graph))
        
        # Edge quality
        results.update(self._analyze_edges(graph, edges_gdf))
        
        # Compute quality score
        results['topology_score'] = self._compute_quality_score(results)
        
        return results
    
    def _analyze_components(self, graph: nx.Graph) -> Dict:
        """Analyze connected components."""
        components = list(nx.connected_components(graph))
        
        component_sizes = [len(c) for c in components]
        
        return {
            'num_components': len(components),
            'largest_component_size': max(component_sizes) if component_sizes else 0,
            'isolated_nodes': sum(1 for c in components if len(c) == 1),
            'component_sizes': component_sizes
        }
    
    def _analyze_dead_ends(self, graph: nx.Graph) -> Dict:
        """Analyze dead-end nodes (degree 1)."""
        degrees = dict(graph.degree())
        
        dead_ends = [n for n, d in degrees.items() if d == 1]
        junctions = [n for n, d in degrees.items() if d >= 3]
        
        return {
            'num_dead_ends': len(dead_ends),
            'num_junctions': len(junctions),
            'dead_end_ratio': len(dead_ends) / max(1, len(degrees))
        }
    
    def _analyze_edges(
        self,
        graph: nx.Graph,
        edges_gdf: gpd.GeoDataFrame
    ) -> Dict:
        """Analyze edge quality."""
        
        lengths = edges_gdf['length'].tolist() if len(edges_gdf) > 0 else []
        curvatures = edges_gdf['curvature'].tolist() if len(edges_gdf) > 0 else []
        
        # Short edges
        short_edges = [l for l in lengths if l < self.min_edge_length]
        
        # Curved edges (low straightness ratio)
        curved_edges = [c for c in curvatures if c < self.curvature_threshold]
        
        return {
            'total_length': sum(lengths),
            'mean_edge_length': np.mean(lengths) if lengths else 0,
            'num_short_edges': len(short_edges),
            'num_curved_edges': len(curved_edges),
            'mean_curvature': np.mean(curvatures) if curvatures else 1.0
        }
    
    def _compute_quality_score(self, results: Dict) -> float:
        """
        Compute overall topology quality score [0, 1].
        
        Higher = better quality network.
        """
        score = 1.0
        
        # Penalize many small components
        if results['num_components'] > 1:
            score -= 0.1 * min(results['num_components'] - 1, 5) / 5
        
        # Penalize high dead-end ratio
        score -= 0.3 * min(results['dead_end_ratio'], 1.0)
        
        # Penalize short edges
        if results['num_edges'] > 0:
            short_ratio = results['num_short_edges'] / results['num_edges']
            score -= 0.2 * min(short_ratio, 1.0)
        
        # Penalize curved edges
        score -= 0.2 * (1 - results['mean_curvature'])
        
        return max(0, min(1, score))


# =============================================================================
# Part 3: Baseline Agreement Analysis
# =============================================================================

class BaselineComparator:
    """
    Compare detected network against authoritative baselines.
    
    Baselines:
    - NYC GIS (LION centerlines) - primary
    - OpenStreetMap - secondary
    
    Metrics:
    - Buffered overlap ratio
    - Symmetric difference
    - FP/FN spatial analysis
    """
    
    def __init__(
        self,
        buffer_distance: float = 1.5,  # meters
        match_tolerance: float = 2.0    # meters
    ):
        self.buffer_distance = buffer_distance
        self.match_tolerance = match_tolerance
    
    def compare(
        self,
        detected_gdf: gpd.GeoDataFrame,
        baseline_gdf: gpd.GeoDataFrame,
        baseline_name: str = "baseline"
    ) -> Dict:
        """
        Compare detected network against baseline.
        
        Args:
            detected_gdf: Detected network edges
            baseline_gdf: Baseline network edges
            baseline_name: Name for logging
            
        Returns:
            Dict with comparison metrics
        """
        if len(detected_gdf) == 0 or len(baseline_gdf) == 0:
            logger.warning("Empty GeoDataFrame provided for comparison")
            return self._empty_results(baseline_name)
        
        # Ensure same CRS
        if detected_gdf.crs != baseline_gdf.crs:
            baseline_gdf = baseline_gdf.to_crs(detected_gdf.crs)
        
        results = {
            'baseline_name': baseline_name,
            'detected_length': detected_gdf.geometry.length.sum(),
            'baseline_length': baseline_gdf.geometry.length.sum()
        }
        
        # Buffer-based comparison
        results.update(self._buffered_comparison(detected_gdf, baseline_gdf))
        
        # Symmetric difference
        results.update(self._symmetric_difference(detected_gdf, baseline_gdf))
        
        # Spatial mismatch analysis
        results.update(self._mismatch_analysis(detected_gdf, baseline_gdf))
        
        return results
    
    def _buffered_comparison(
        self,
        detected_gdf: gpd.GeoDataFrame,
        baseline_gdf: gpd.GeoDataFrame
    ) -> Dict:
        """Compute buffered overlap metrics."""
        
        # Create buffers
        detected_union = unary_union(detected_gdf.geometry)
        baseline_union = unary_union(baseline_gdf.geometry)
        
        detected_buffer = detected_union.buffer(self.buffer_distance)
        baseline_buffer = baseline_union.buffer(self.buffer_distance)
        
        # Intersection
        intersection = detected_buffer.intersection(baseline_buffer)
        
        # Coverage metrics
        detected_covered = detected_union.intersection(baseline_buffer).length
        baseline_covered = baseline_union.intersection(detected_buffer).length
        
        detected_overlap_ratio = detected_covered / max(detected_union.length, 1e-10)
        baseline_overlap_ratio = baseline_covered / max(baseline_union.length, 1e-10)
        
        return {
            'detected_overlap_ratio': detected_overlap_ratio,
            'baseline_overlap_ratio': baseline_overlap_ratio,
            'mutual_overlap': (detected_overlap_ratio + baseline_overlap_ratio) / 2
        }
    
    def _symmetric_difference(
        self,
        detected_gdf: gpd.GeoDataFrame,
        baseline_gdf: gpd.GeoDataFrame
    ) -> Dict:
        """Compute symmetric difference (areas of disagreement)."""
        
        detected_union = unary_union(detected_gdf.geometry.buffer(self.buffer_distance))
        baseline_union = unary_union(baseline_gdf.geometry.buffer(self.buffer_distance))
        
        # Symmetric difference
        sym_diff = detected_union.symmetric_difference(baseline_union)
        
        # Only in detected (potential FP)
        detected_only = detected_union.difference(baseline_union)
        
        # Only in baseline (potential FN)
        baseline_only = baseline_union.difference(detected_union)
        
        return {
            'symmetric_difference_area': sym_diff.area,
            'detected_only_area': detected_only.area,  # potential FP
            'baseline_only_area': baseline_only.area   # potential FN
        }
    
    def _mismatch_analysis(
        self,
        detected_gdf: gpd.GeoDataFrame,
        baseline_gdf: gpd.GeoDataFrame
    ) -> Dict:
        """Analyze spatial distribution of mismatches."""
        
        baseline_buffer = unary_union(baseline_gdf.geometry.buffer(self.match_tolerance))
        
        # Edges in detected that don't match baseline (potential FP)
        unmatched_detected = []
        matched_detected = []
        
        for idx, row in detected_gdf.iterrows():
            if row.geometry.intersection(baseline_buffer).length / max(row.geometry.length, 1e-10) < 0.5:
                unmatched_detected.append(row)
            else:
                matched_detected.append(row)
        
        return {
            'num_matched_edges': len(matched_detected),
            'num_unmatched_edges': len(unmatched_detected),
            'match_ratio': len(matched_detected) / max(len(detected_gdf), 1)
        }
    
    def _empty_results(self, baseline_name: str) -> Dict:
        """Return empty results structure."""
        return {
            'baseline_name': baseline_name,
            'detected_length': 0,
            'baseline_length': 0,
            'detected_overlap_ratio': 0,
            'baseline_overlap_ratio': 0,
            'mutual_overlap': 0,
            'symmetric_difference_area': 0,
            'detected_only_area': 0,
            'baseline_only_area': 0,
            'num_matched_edges': 0,
            'num_unmatched_edges': 0,
            'match_ratio': 0
        }


# =============================================================================
# Part 4: A→B Linkage Experiments
# =============================================================================

class ABLinkageExperiment:
    """
    Demonstrate A→B linkage: how pixel-level choices affect network quality.
    
    Experiments:
    1. Operating-point transfer (uncalibrated vs calibrated)
    2. Sensitivity to tolerances
    """
    
    def __init__(
        self,
        vectorizer: MaskVectorizer,
        topology_analyzer: TopologyAnalyzer,
        baseline_comparator: BaselineComparator
    ):
        self.vectorizer = vectorizer
        self.topology_analyzer = topology_analyzer
        self.baseline_comparator = baseline_comparator
    
    def run_operating_point_experiment(
        self,
        probability_map: np.ndarray,
        baseline_gdf: gpd.GeoDataFrame,
        transform: Optional[Affine] = None,
        crs: Optional[str] = None,
        default_threshold: float = 0.5,
        calibrated_threshold: float = 0.4,
        morphology_kernel: int = 5
    ) -> Dict:
        """
        Experiment 1: Compare uncalibrated baseline vs calibrated+cleanup.
        
        Args:
            probability_map: Per-pixel probability array
            baseline_gdf: Authoritative baseline network
            transform: Georeferencing transform
            crs: Coordinate reference system
            default_threshold: Uncalibrated threshold (typically 0.5)
            calibrated_threshold: Calibrated threshold from Stage A
            morphology_kernel: Kernel size for morphological cleanup
            
        Returns:
            Dict with comparative metrics
        """
        results = {
            'experiment': 'operating_point_transfer',
            'default_threshold': default_threshold,
            'calibrated_threshold': calibrated_threshold
        }
        
        # Setting 1: Uncalibrated baseline
        logger.info("Running uncalibrated baseline...")
        mask_uncal = (probability_map >= default_threshold).astype(np.uint8)
        edges_uncal, graph_uncal = self.vectorizer.vectorize(mask_uncal, transform, crs)
        
        topo_uncal = self.topology_analyzer.analyze(graph_uncal, edges_uncal)
        compare_uncal = self.baseline_comparator.compare(edges_uncal, baseline_gdf, "uncalibrated")
        
        results['uncalibrated'] = {
            'topology': topo_uncal,
            'baseline_comparison': compare_uncal
        }
        
        # Setting 2: Calibrated + cleanup
        logger.info("Running calibrated + cleanup...")
        mask_cal = (probability_map >= calibrated_threshold).astype(np.uint8)
        
        # Apply morphological opening (erosion then dilation)
        from scipy.ndimage import binary_opening
        kernel = np.ones((morphology_kernel, morphology_kernel))
        mask_cal = binary_opening(mask_cal, structure=kernel).astype(np.uint8)
        
        edges_cal, graph_cal = self.vectorizer.vectorize(mask_cal, transform, crs)
        
        topo_cal = self.topology_analyzer.analyze(graph_cal, edges_cal)
        compare_cal = self.baseline_comparator.compare(edges_cal, baseline_gdf, "calibrated")
        
        results['calibrated'] = {
            'topology': topo_cal,
            'baseline_comparison': compare_cal
        }
        
        # Compute deltas
        results['deltas'] = {
            'dead_ends_change': topo_cal['num_dead_ends'] - topo_uncal['num_dead_ends'],
            'short_edges_change': topo_cal['num_short_edges'] - topo_uncal['num_short_edges'],
            'overlap_ratio_change': compare_cal['mutual_overlap'] - compare_uncal['mutual_overlap'],
            'components_change': topo_cal['num_components'] - topo_uncal['num_components'],
            'topology_score_change': topo_cal['topology_score'] - topo_uncal['topology_score']
        }
        
        # Log summary
        logger.info("A→B Experiment Results:")
        logger.info(f"  Dead-ends: {topo_uncal['num_dead_ends']} → {topo_cal['num_dead_ends']} ({results['deltas']['dead_ends_change']:+d})")
        logger.info(f"  Overlap ratio: {compare_uncal['mutual_overlap']:.3f} → {compare_cal['mutual_overlap']:.3f} ({results['deltas']['overlap_ratio_change']:+.3f})")
        logger.info(f"  Topology score: {topo_uncal['topology_score']:.3f} → {topo_cal['topology_score']:.3f} ({results['deltas']['topology_score_change']:+.3f})")
        
        return results
    
    def run_tolerance_sensitivity_experiment(
        self,
        detected_gdf: gpd.GeoDataFrame,
        baseline_gdf: gpd.GeoDataFrame,
        tolerances: List[float] = [1.0, 1.5, 2.0]
    ) -> Dict:
        """
        Experiment 2: Show sensitivity to buffer tolerances.
        
        Args:
            detected_gdf: Detected network
            baseline_gdf: Baseline network
            tolerances: List of buffer distances to test (meters)
            
        Returns:
            Dict with metrics at each tolerance
        """
        results = {
            'experiment': 'tolerance_sensitivity',
            'tolerances': tolerances,
            'metrics_by_tolerance': {}
        }
        
        for tol in tolerances:
            comparator = BaselineComparator(buffer_distance=tol)
            metrics = comparator.compare(detected_gdf, baseline_gdf, f"tol_{tol}")
            results['metrics_by_tolerance'][tol] = metrics
            
            logger.info(f"Tolerance {tol}m: overlap={metrics['mutual_overlap']:.3f}")
        
        return results


# =============================================================================
# Part 5: Complete Stage B Pipeline
# =============================================================================

class StageBPipeline:
    """
    Complete Stage B evaluation pipeline.
    
    Runs:
    1. Vectorization of Stage A's final mask
    2. Topology & geometry quality analysis
    3. Baseline agreement analysis
    4. A→B linkage experiments
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        simplify_tolerance: float = 0.5,
        min_edge_length: float = 3.0,
        buffer_distance: float = 1.5
    ):
        self.vectorizer = MaskVectorizer(
            simplify_tolerance=simplify_tolerance,
            min_edge_length=min_edge_length
        )
        self.topology_analyzer = TopologyAnalyzer(min_edge_length=min_edge_length)
        self.baseline_comparator = BaselineComparator(buffer_distance=buffer_distance)
        self.ab_experiment = ABLinkageExperiment(
            self.vectorizer, 
            self.topology_analyzer,
            self.baseline_comparator
        )
    
    def run(
        self,
        probability_path: str,
        baseline_path: str,
        output_dir: str = "outputs/network",
        default_threshold: float = 0.5,
        calibrated_threshold: float = 0.4,
        morphology_kernel: int = 5
    ) -> Dict:
        """
        Run complete Stage B pipeline.
        
        Args:
            probability_path: Path to probability map from Stage A
            baseline_path: Path to baseline network (LION or OSM)
            output_dir: Output directory
            default_threshold: Uncalibrated threshold
            calibrated_threshold: Calibrated threshold from Stage A
            morphology_kernel: Morphology kernel size
            
        Returns:
            Dict with all Stage B results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("STAGE B: NETWORK/VECTOR-LEVEL EVALUATION")
        logger.info("=" * 60)
        
        # Load data
        logger.info("Loading data...")
        with rasterio.open(probability_path) as src:
            probabilities = src.read(1)
            transform = src.transform
            crs = src.crs
        
        baseline_gdf = gpd.read_file(baseline_path)
        
        # Create binary mask at calibrated threshold
        mask = (probabilities >= calibrated_threshold).astype(np.uint8)
        
        # Apply morphology
        from scipy.ndimage import binary_opening
        kernel = np.ones((morphology_kernel, morphology_kernel))
        mask = binary_opening(mask, structure=kernel).astype(np.uint8)
        
        # 1. Vectorization
        logger.info("\n1. VECTORIZATION")
        edges_gdf, graph = self.vectorizer.vectorize(mask, transform, str(crs))
        
        # Save vectorized network
        if len(edges_gdf) > 0:
            edges_gdf.to_file(output_dir / "detected_network.geojson", driver="GeoJSON")
        
        # 2. Topology Analysis
        logger.info("\n2. TOPOLOGY ANALYSIS")
        topology_results = self.topology_analyzer.analyze(graph, edges_gdf)
        
        # 3. Baseline Comparison
        logger.info("\n3. BASELINE COMPARISON")
        comparison_results = self.baseline_comparator.compare(
            edges_gdf, baseline_gdf, "NYC_GIS"
        )
        
        # 4. A→B Linkage Experiment
        logger.info("\n4. A→B LINKAGE EXPERIMENT")
        ab_results = self.ab_experiment.run_operating_point_experiment(
            probability_map=probabilities,
            baseline_gdf=baseline_gdf,
            transform=transform,
            crs=str(crs),
            default_threshold=default_threshold,
            calibrated_threshold=calibrated_threshold,
            morphology_kernel=morphology_kernel
        )
        
        # Compile results
        results = {
            'stage': 'B',
            'vectorization': {
                'num_edges': len(edges_gdf),
                'num_nodes': graph.number_of_nodes(),
                'total_length': edges_gdf.geometry.length.sum() if len(edges_gdf) > 0 else 0
            },
            'topology': topology_results,
            'baseline_comparison': comparison_results,
            'ab_linkage': ab_results
        }
        
        # Save results
        results_path = output_dir / "stage_b_results.json"
        with open(results_path, 'w') as f:
            # Convert non-serializable items
            json.dump(self._make_serializable(results), f, indent=2)
        
        logger.info(f"\nResults saved to {results_path}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _print_summary(self, results: Dict):
        """Print results summary."""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE B SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"\nNetwork Statistics:")
        logger.info(f"  Edges: {results['vectorization']['num_edges']}")
        logger.info(f"  Nodes: {results['vectorization']['num_nodes']}")
        logger.info(f"  Total Length: {results['vectorization']['total_length']:.1f}m")
        
        logger.info(f"\nTopology Quality:")
        topo = results['topology']
        logger.info(f"  Components: {topo['num_components']}")
        logger.info(f"  Dead-ends: {topo['num_dead_ends']}")
        logger.info(f"  Junctions: {topo['num_junctions']}")
        logger.info(f"  Quality Score: {topo['topology_score']:.3f}")
        
        logger.info(f"\nBaseline Agreement:")
        comp = results['baseline_comparison']
        logger.info(f"  Overlap Ratio: {comp['mutual_overlap']:.3f}")
        logger.info(f"  Match Ratio: {comp['match_ratio']:.3f}")
        
        logger.info(f"\nA→B Improvement (Calibrated vs Uncalibrated):")
        deltas = results['ab_linkage']['deltas']
        logger.info(f"  Dead-ends: {deltas['dead_ends_change']:+d}")
        logger.info(f"  Overlap: {deltas['overlap_ratio_change']:+.3f}")
        logger.info(f"  Topology Score: {deltas['topology_score_change']:+.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage B: Network Analysis")
    parser.add_argument("--prob", required=True, help="Probability map path")
    parser.add_argument("--baseline", required=True, help="Baseline network path")
    parser.add_argument("--output", default="outputs/network", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.4, help="Calibrated threshold")
    
    args = parser.parse_args()
    
    pipeline = StageBPipeline()
    results = pipeline.run(
        probability_path=args.prob,
        baseline_path=args.baseline,
        output_dir=args.output,
        calibrated_threshold=args.threshold
    )