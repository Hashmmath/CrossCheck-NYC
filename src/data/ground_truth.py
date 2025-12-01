"""
Ground Truth Generation for Brooklyn Crosswalk QA
=================================================

Creates rasterized ground truth masks from vector data sources:
1. OSM crossings (primary)
2. Vision Zero enhanced crossings (validation subset)
3. (Optional) Any custom annotated crossings

The ground truth masks are aligned to match model prediction grids.

Usage:
    python ground_truth.py --config config/config.yaml
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.affinity import scale
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundTruthGenerator:
    """
    Generate rasterized ground truth masks for crosswalk detection.
    
    Creates pixel-level masks from vector crossing data, aligned to
    match the model prediction grid for accurate evaluation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.paths = self.config['paths']
        self.target_crs = CRS.from_string(self.config['location']['crs']['target'])
        
        # Crosswalk buffer size (meters) - crosswalks are typically 2-4m wide
        self.crossing_buffer = 2.5  # meters
        
        # Default pixel resolution (matches typical ortho imagery)
        self.pixel_resolution = 0.15  # meters per pixel at zoom 19
    
    def load_osm_crossings(self, osm_path: Optional[Path] = None) -> gpd.GeoDataFrame:
        """
        Load OSM crossing data.
        
        Args:
            osm_path: Path to OSM crossings GeoJSON
            
        Returns:
            GeoDataFrame of crossings
        """
        osm_path = osm_path or Path(self.paths['raw']['osm']) / "osm_crossings_brooklyn.geojson"
        
        if not osm_path.exists():
            raise FileNotFoundError(f"OSM crossings not found at {osm_path}")
        
        gdf = gpd.read_file(osm_path)
        logger.info(f"Loaded {len(gdf)} OSM crossings")
        
        return gdf
    
    def load_vision_zero_crossings(
        self, 
        vz_path: Optional[Path] = None
    ) -> Optional[gpd.GeoDataFrame]:
        """
        Load Vision Zero enhanced crossings data.
        
        Args:
            vz_path: Path to Vision Zero GeoJSON
            
        Returns:
            GeoDataFrame of crossings, or None if not available
        """
        vz_path = vz_path or Path(self.paths['raw']['vision_zero']) / "vision_zero_crossings.geojson"
        
        if not vz_path.exists():
            logger.warning(f"Vision Zero crossings not found at {vz_path}")
            return None
        
        gdf = gpd.read_file(vz_path)
        logger.info(f"Loaded {len(gdf)} Vision Zero crossings")
        
        return gdf
    
    def buffer_crossings(
        self,
        gdf: gpd.GeoDataFrame,
        buffer_meters: float = 2.5
    ) -> gpd.GeoDataFrame:
        """
        Buffer point crossings to create polygon footprints.
        
        Most OSM crossings are points. We buffer them to approximate
        the actual crosswalk area for rasterization.
        
        Args:
            gdf: GeoDataFrame with crossing geometries
            buffer_meters: Buffer radius in meters
            
        Returns:
            GeoDataFrame with buffered geometries
        """
        # Ensure CRS is set
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        
        # Project to a meter-based CRS for accurate buffering
        # Using UTM zone 18N for NYC
        gdf_projected = gdf.to_crs('EPSG:32618')
        
        # Buffer
        gdf_projected['geometry'] = gdf_projected.geometry.buffer(buffer_meters)
        
        # Project back to WGS84
        gdf_buffered = gdf_projected.to_crs('EPSG:4326')
        
        logger.info(f"Buffered {len(gdf_buffered)} crossings with {buffer_meters}m radius")
        
        return gdf_buffered
    
    def create_crossing_polygons(
        self,
        gdf: gpd.GeoDataFrame,
        crossing_width: float = 3.0,
        crossing_length: float = 10.0
    ) -> gpd.GeoDataFrame:
        """
        Create more realistic crosswalk polygons based on typical dimensions.
        
        For point crossings, creates rectangular polygons oriented
        perpendicular to the nearest road (approximated).
        
        Args:
            gdf: GeoDataFrame with crossing points
            crossing_width: Crosswalk width in meters
            crossing_length: Crosswalk length in meters
            
        Returns:
            GeoDataFrame with rectangular crossing polygons
        """
        # For now, use circular buffer as a simpler approximation
        # TODO: Integrate with LION data to get road orientations
        return self.buffer_crossings(gdf, buffer_meters=crossing_width)
    
    def rasterize_to_tile(
        self,
        gdf: gpd.GeoDataFrame,
        bounds: Tuple[float, float, float, float],
        width: int,
        height: int,
        all_touched: bool = True
    ) -> np.ndarray:
        """
        Rasterize crossing polygons to match a tile's grid.
        
        Args:
            gdf: GeoDataFrame with crossing polygons
            bounds: Tile bounds (west, south, east, north) in EPSG:4326
            width: Output raster width in pixels
            height: Output raster height in pixels
            all_touched: Whether to include pixels that are touched by geometry edge
            
        Returns:
            Binary mask array (1 = crossing, 0 = background)
        """
        west, south, east, north = bounds
        
        # Create transform
        transform = from_bounds(west, south, east, north, width, height)
        
        # Clip geometries to bounds
        tile_box = box(west, south, east, north)
        
        # Filter to geometries that intersect the tile
        gdf_clipped = gdf[gdf.intersects(tile_box)].copy()
        
        if len(gdf_clipped) == 0:
            return np.zeros((height, width), dtype=np.uint8)
        
        # Clip geometries to tile bounds
        gdf_clipped['geometry'] = gdf_clipped.intersection(tile_box)
        
        # Remove empty geometries
        gdf_clipped = gdf_clipped[~gdf_clipped.is_empty]
        
        if len(gdf_clipped) == 0:
            return np.zeros((height, width), dtype=np.uint8)
        
        # Rasterize
        shapes = [(geom, 1) for geom in gdf_clipped.geometry]
        
        mask = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            all_touched=all_touched,
            dtype=np.uint8
        )
        
        return mask
    
    def generate_ground_truth_for_tiles(
        self,
        tile_manifest_path: Path,
        output_dir: Path,
        osm_gdf: Optional[gpd.GeoDataFrame] = None
    ) -> List[Path]:
        """
        Generate ground truth masks for all tiles in a manifest.
        
        Args:
            tile_manifest_path: Path to tile processing manifest JSON
            output_dir: Output directory for masks
            osm_gdf: Pre-loaded OSM crossings (optional)
            
        Returns:
            List of generated mask paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tile manifest
        with open(tile_manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Load crossings if not provided
        if osm_gdf is None:
            osm_gdf = self.load_osm_crossings()
        
        # Buffer crossings
        crossings_buffered = self.buffer_crossings(osm_gdf, self.crossing_buffer)
        
        generated_masks = []
        
        # Process each stitched tile
        for tile_path_str in tqdm(manifest.get('stitched', []), desc="Generating ground truth"):
            tile_path = Path(tile_path_str)
            
            if not tile_path.exists():
                logger.warning(f"Tile not found: {tile_path}")
                continue
            
            # Read tile to get bounds and dimensions
            with rasterio.open(tile_path) as src:
                bounds = src.bounds
                width = src.width
                height = src.height
            
            # Rasterize
            mask = self.rasterize_to_tile(
                crossings_buffered,
                (bounds.left, bounds.bottom, bounds.right, bounds.top),
                width,
                height
            )
            
            # Save mask
            mask_name = tile_path.stem + "_gt.tif"
            mask_path = output_dir / mask_name
            
            self._save_mask(mask, mask_path, bounds, width, height)
            
            generated_masks.append(mask_path)
        
        # Save ground truth manifest
        gt_manifest = {
            "source_manifest": str(tile_manifest_path),
            "crossing_buffer_m": self.crossing_buffer,
            "masks": [str(p) for p in generated_masks],
            "mask_count": len(generated_masks)
        }
        
        manifest_path = output_dir / "ground_truth_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(gt_manifest, f, indent=2)
        
        logger.info(f"Generated {len(generated_masks)} ground truth masks")
        
        return generated_masks
    
    def _save_mask(
        self,
        mask: np.ndarray,
        output_path: Path,
        bounds: rasterio.coords.BoundingBox,
        width: int,
        height: int
    ):
        """Save mask as GeoTIFF."""
        transform = from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top,
            width, height
        )
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=mask.dtype,
            crs=CRS.from_epsg(4326),
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(mask, 1)
    
    def align_ground_truth_to_prediction(
        self,
        gt_path: Path,
        pred_path: Path,
        output_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align ground truth mask to match prediction grid exactly.
        
        Critical for accurate pixel-level evaluation. Ensures both
        rasters have identical CRS, extent, and resolution.
        
        Args:
            gt_path: Path to ground truth GeoTIFF
            pred_path: Path to prediction GeoTIFF
            output_path: Optional path to save aligned ground truth
            
        Returns:
            Tuple of (aligned ground truth array, prediction array)
        """
        # Read prediction to get target grid
        with rasterio.open(pred_path) as pred_src:
            pred_array = pred_src.read(1)
            pred_transform = pred_src.transform
            pred_crs = pred_src.crs
            pred_width = pred_src.width
            pred_height = pred_src.height
        
        # Read and reproject ground truth
        with rasterio.open(gt_path) as gt_src:
            gt_array = gt_src.read(1)
            gt_transform = gt_src.transform
            gt_crs = gt_src.crs
            
            # Prepare output array
            gt_aligned = np.zeros((pred_height, pred_width), dtype=gt_array.dtype)
            
            # Reproject
            reproject(
                source=gt_array,
                destination=gt_aligned,
                src_transform=gt_transform,
                src_crs=gt_crs,
                dst_transform=pred_transform,
                dst_crs=pred_crs,
                resampling=Resampling.nearest
            )
        
        # Save aligned ground truth if requested
        if output_path:
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=pred_height,
                width=pred_width,
                count=1,
                dtype=gt_aligned.dtype,
                crs=pred_crs,
                transform=pred_transform,
                compress='lzw'
            ) as dst:
                dst.write(gt_aligned, 1)
        
        return gt_aligned, pred_array
    
    def verify_alignment(
        self,
        gt_path: Path,
        pred_path: Path
    ) -> Dict:
        """
        Verify that ground truth and prediction are properly aligned.
        
        Returns:
            Dict with alignment verification results
        """
        with rasterio.open(gt_path) as gt_src:
            gt_bounds = gt_src.bounds
            gt_crs = gt_src.crs
            gt_res = gt_src.res
            gt_shape = (gt_src.height, gt_src.width)
        
        with rasterio.open(pred_path) as pred_src:
            pred_bounds = pred_src.bounds
            pred_crs = pred_src.crs
            pred_res = pred_src.res
            pred_shape = (pred_src.height, pred_src.width)
        
        # Check alignment
        crs_match = gt_crs == pred_crs
        bounds_match = (
            abs(gt_bounds.left - pred_bounds.left) < 1e-6 and
            abs(gt_bounds.bottom - pred_bounds.bottom) < 1e-6 and
            abs(gt_bounds.right - pred_bounds.right) < 1e-6 and
            abs(gt_bounds.top - pred_bounds.top) < 1e-6
        )
        res_match = (
            abs(gt_res[0] - pred_res[0]) < 1e-6 and
            abs(gt_res[1] - pred_res[1]) < 1e-6
        )
        shape_match = gt_shape == pred_shape
        
        result = {
            "aligned": crs_match and bounds_match and res_match and shape_match,
            "crs_match": crs_match,
            "bounds_match": bounds_match,
            "resolution_match": res_match,
            "shape_match": shape_match,
            "gt_crs": str(gt_crs),
            "pred_crs": str(pred_crs),
            "gt_bounds": dict(gt_bounds._asdict()),
            "pred_bounds": dict(pred_bounds._asdict()),
            "gt_resolution": gt_res,
            "pred_resolution": pred_res,
            "gt_shape": gt_shape,
            "pred_shape": pred_shape
        }
        
        if result["aligned"]:
            logger.info("✓ Ground truth and prediction are properly aligned")
        else:
            logger.warning("✗ Alignment issues detected:")
            if not crs_match:
                logger.warning(f"  - CRS mismatch: {gt_crs} vs {pred_crs}")
            if not bounds_match:
                logger.warning(f"  - Bounds mismatch")
            if not res_match:
                logger.warning(f"  - Resolution mismatch: {gt_res} vs {pred_res}")
            if not shape_match:
                logger.warning(f"  - Shape mismatch: {gt_shape} vs {pred_shape}")
        
        return result
    
    def create_checkerboard_overlay(
        self,
        gt_path: Path,
        pred_path: Path,
        output_path: Path,
        block_size: int = 32
    ):
        """
        Create a checkerboard overlay visualization to verify alignment.
        
        Alternates blocks between ground truth and prediction edges
        to visually confirm pixel-level alignment.
        
        Args:
            gt_path: Path to ground truth GeoTIFF
            pred_path: Path to prediction GeoTIFF
            output_path: Path to save visualization
            block_size: Size of checkerboard blocks in pixels
        """
        from PIL import Image
        import cv2
        
        # Read images
        with rasterio.open(gt_path) as src:
            gt = src.read(1)
        with rasterio.open(pred_path) as src:
            pred = src.read(1)
        
        if gt.shape != pred.shape:
            logger.error("Cannot create checkerboard: shapes don't match")
            return
        
        height, width = gt.shape
        
        # Create edge images
        gt_edges = cv2.Canny((gt * 255).astype(np.uint8), 50, 150)
        pred_edges = cv2.Canny((pred * 255).astype(np.uint8), 50, 150)
        
        # Create checkerboard mask
        checkerboard = np.zeros((height, width), dtype=bool)
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    checkerboard[i:i+block_size, j:j+block_size] = True
        
        # Combine
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[checkerboard, 0] = gt_edges[checkerboard]  # Red for GT
        overlay[~checkerboard, 2] = pred_edges[~checkerboard]  # Blue for pred
        
        # Save
        Image.fromarray(overlay).save(output_path)
        logger.info(f"Saved checkerboard overlay to {output_path}")


def main():
    """CLI for ground truth generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ground truth masks")
    
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file')
    parser.add_argument('--tile-manifest', '-t', help='Tile processing manifest')
    parser.add_argument('--output', '-o', help='Output directory for masks')
    parser.add_argument('--verify', action='store_true', help='Verify alignment between GT and prediction')
    parser.add_argument('--gt-path', help='Ground truth path (for verification)')
    parser.add_argument('--pred-path', help='Prediction path (for verification)')
    
    args = parser.parse_args()
    
    generator = GroundTruthGenerator(args.config)
    
    if args.verify and args.gt_path and args.pred_path:
        result = generator.verify_alignment(
            Path(args.gt_path),
            Path(args.pred_path)
        )
        print(json.dumps(result, indent=2))
    
    elif args.tile_manifest:
        output_dir = Path(args.output or generator.paths['processed']['ground_truth'])
        masks = generator.generate_ground_truth_for_tiles(
            Path(args.tile_manifest),
            output_dir
        )
        print(f"Generated {len(masks)} ground truth masks")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()