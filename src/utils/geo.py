"""
Geospatial Utilities for Brooklyn Crosswalk QA
=============================================

Common geospatial operations:
- Coordinate transformations
- Bounding box utilities
- GeoJSON I/O
- Intersection and buffer operations
"""

import math
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point, Polygon, MultiPolygon, mapping, shape
from shapely.ops import unary_union
import pyproj
from pyproj import Transformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeoUtils:
    """Geospatial utility functions."""
    
    # Common CRS definitions
    WGS84 = "EPSG:4326"
    WEB_MERCATOR = "EPSG:3857"
    NYC_STATE_PLANE = "EPSG:2263"  # NYC State Plane (feet)
    UTM_18N = "EPSG:32618"  # UTM zone 18N (meters)
    
    @staticmethod
    def create_bbox(
        west: float,
        south: float,
        east: float,
        north: float
    ) -> Polygon:
        """Create a bounding box polygon."""
        return box(west, south, east, north)
    
    @staticmethod
    def bbox_from_center(
        center_lon: float,
        center_lat: float,
        width_m: float,
        height_m: float
    ) -> Tuple[float, float, float, float]:
        """
        Create bounding box from center point and dimensions in meters.
        
        Args:
            center_lon: Center longitude
            center_lat: Center latitude
            width_m: Width in meters
            height_m: Height in meters
            
        Returns:
            (west, south, east, north) tuple
        """
        # Approximate degrees per meter at this latitude
        lat_rad = math.radians(center_lat)
        m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad)
        m_per_deg_lon = 111412.84 * math.cos(lat_rad)
        
        half_width = (width_m / 2) / m_per_deg_lon
        half_height = (height_m / 2) / m_per_deg_lat
        
        return (
            center_lon - half_width,
            center_lat - half_height,
            center_lon + half_width,
            center_lat + half_height
        )
    
    @staticmethod
    def transform_point(
        lon: float,
        lat: float,
        from_crs: str,
        to_crs: str
    ) -> Tuple[float, float]:
        """Transform a point between coordinate systems."""
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        return transformer.transform(lon, lat)
    
    @staticmethod
    def transform_bbox(
        bbox: Tuple[float, float, float, float],
        from_crs: str,
        to_crs: str
    ) -> Tuple[float, float, float, float]:
        """Transform bounding box between coordinate systems."""
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        
        west, south, east, north = bbox
        
        # Transform all four corners and find new extent
        corners = [
            (west, south),
            (west, north),
            (east, south),
            (east, north)
        ]
        
        transformed = [transformer.transform(x, y) for x, y in corners]
        
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    @staticmethod
    def tile_to_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
        """
        Convert tile indices to geographic bounds.
        
        Uses Web Mercator tiling scheme.
        
        Returns:
            (west, south, east, north) in EPSG:4326
        """
        n = 2 ** z
        
        west = x / n * 360.0 - 180.0
        east = (x + 1) / n * 360.0 - 180.0
        
        north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        
        return (west, south, east, north)
    
    @staticmethod
    def bounds_to_tile(lat: float, lon: float, z: int) -> Tuple[int, int]:
        """Convert lat/lon to tile indices at zoom level z."""
        n = 2 ** z
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y
    
    @staticmethod
    def get_tiles_for_bbox(
        bbox: Tuple[float, float, float, float],
        zoom: int
    ) -> List[Tuple[int, int, int]]:
        """
        Get all tile indices covering a bounding box.
        
        Args:
            bbox: (west, south, east, north)
            zoom: Zoom level
            
        Returns:
            List of (z, x, y) tuples
        """
        west, south, east, north = bbox
        
        x_min, y_max = GeoUtils.bounds_to_tile(south, west, zoom)
        x_max, y_min = GeoUtils.bounds_to_tile(north, east, zoom)
        
        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tiles.append((zoom, x, y))
        
        return tiles
    
    @staticmethod
    def buffer_meters(
        gdf: gpd.GeoDataFrame,
        distance_m: float,
        crs_meters: str = "EPSG:32618"
    ) -> gpd.GeoDataFrame:
        """
        Buffer geometries by distance in meters.
        
        Args:
            gdf: GeoDataFrame to buffer
            distance_m: Buffer distance in meters
            crs_meters: CRS with meter units for accurate buffering
            
        Returns:
            Buffered GeoDataFrame in original CRS
        """
        original_crs = gdf.crs
        
        # Project to meter-based CRS
        gdf_proj = gdf.to_crs(crs_meters)
        
        # Buffer
        gdf_proj['geometry'] = gdf_proj.geometry.buffer(distance_m)
        
        # Project back
        return gdf_proj.to_crs(original_crs)
    
    @staticmethod
    def spatial_join_within_distance(
        gdf1: gpd.GeoDataFrame,
        gdf2: gpd.GeoDataFrame,
        distance_m: float,
        crs_meters: str = "EPSG:32618"
    ) -> gpd.GeoDataFrame:
        """
        Spatial join where features are within distance of each other.
        
        Args:
            gdf1: First GeoDataFrame
            gdf2: Second GeoDataFrame
            distance_m: Maximum distance in meters
            crs_meters: CRS for distance calculations
            
        Returns:
            Joined GeoDataFrame
        """
        # Project to meters
        gdf1_proj = gdf1.to_crs(crs_meters)
        gdf2_proj = gdf2.to_crs(crs_meters)
        
        # Buffer gdf2
        gdf2_buffered = gdf2_proj.copy()
        gdf2_buffered['geometry'] = gdf2_buffered.geometry.buffer(distance_m)
        
        # Spatial join
        joined = gpd.sjoin(gdf1_proj, gdf2_buffered, how='inner', predicate='intersects')
        
        return joined
    
    @staticmethod
    def load_and_clip(
        path: Union[str, Path],
        clip_bbox: Tuple[float, float, float, float],
        target_crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Load a vector file and clip to bounding box.
        
        Args:
            path: Path to vector file
            clip_bbox: Bounding box (west, south, east, north)
            target_crs: Target CRS
            
        Returns:
            Clipped GeoDataFrame
        """
        gdf = gpd.read_file(path)
        
        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs)
        elif str(gdf.crs) != target_crs:
            gdf = gdf.to_crs(target_crs)
        
        # Create clip polygon
        clip_poly = box(*clip_bbox)
        
        # Clip
        gdf_clipped = gdf[gdf.intersects(clip_poly)].copy()
        gdf_clipped['geometry'] = gdf_clipped.intersection(clip_poly)
        
        # Remove empty geometries
        gdf_clipped = gdf_clipped[~gdf_clipped.is_empty]
        
        return gdf_clipped
    
    @staticmethod
    def save_geojson(
        gdf: gpd.GeoDataFrame,
        path: Union[str, Path],
        simplify_tolerance: Optional[float] = None
    ):
        """
        Save GeoDataFrame to GeoJSON.
        
        Args:
            gdf: GeoDataFrame to save
            path: Output path
            simplify_tolerance: Optional simplification tolerance
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if simplify_tolerance is not None:
            gdf = gdf.copy()
            gdf['geometry'] = gdf.geometry.simplify(simplify_tolerance)
        
        gdf.to_file(path, driver='GeoJSON')
        logger.info(f"Saved {len(gdf)} features to {path}")
    
    @staticmethod
    def compute_area_sq_meters(
        geometry,
        crs: str = "EPSG:4326"
    ) -> float:
        """Compute area in square meters."""
        if crs == "EPSG:4326":
            # Project to UTM for accurate area
            gdf = gpd.GeoDataFrame(geometry=[geometry], crs=crs)
            gdf_proj = gdf.to_crs("EPSG:32618")
            return gdf_proj.geometry.area.values[0]
        else:
            return geometry.area


def extract_osm_crossings(
    osm_path: Path,
    output_path: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> gpd.GeoDataFrame:
    """
    Extract crossing features from OSM data.
    
    Args:
        osm_path: Path to OSM data file
        output_path: Output path for extracted crossings
        bbox: Optional bounding box filter
        
    Returns:
        GeoDataFrame of crossings
    """
    try:
        import osmnx as ox
        
        if bbox:
            poly = box(*bbox)
            gdf = ox.features_from_polygon(poly, tags={'highway': 'crossing'})
        else:
            gdf = gpd.read_file(osm_path)
            gdf = gdf[gdf.get('highway') == 'crossing']
        
        # Save
        gdf.to_file(output_path, driver='GeoJSON')
        
        return gdf
        
    except ImportError:
        logger.warning("osmnx not available, using basic file loading")
        gdf = gpd.read_file(osm_path)
        
        if bbox:
            clip_poly = box(*bbox)
            gdf = gdf[gdf.intersects(clip_poly)]
        
        gdf.to_file(output_path, driver='GeoJSON')
        return gdf