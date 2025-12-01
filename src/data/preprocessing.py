"""
Tile Preprocessing for Brooklyn Crosswalk QA
============================================

Prepares image tiles for Tile2Net inference:
1. Stitch individual tiles into larger inference tiles
2. Georeference tiles
3. Normalize and prepare for model input

Usage:
    python preprocessing.py --input data/raw/ortho --output data/processed/tiles
"""

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Container for tile metadata."""
    path: Path
    z: int
    x: int
    y: int
    
    @property
    def name(self) -> str:
        return f"tile_{self.z}_{self.x}_{self.y}"
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get tile bounds in EPSG:4326 (lon/lat)."""
        return tile_to_bounds(self.z, self.x, self.y)


def tile_to_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """
    Convert tile indices to geographic bounds.
    
    Returns:
        (west, south, east, north) in EPSG:4326
    """
    n = 2 ** z
    
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    
    return (west, south, east, north)


def bounds_to_tile(lat: float, lon: float, z: int) -> Tuple[int, int]:
    """Convert lat/lon to tile indices at zoom level z."""
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


class TilePreprocessor:
    """
    Preprocessor for aerial imagery tiles.
    
    Handles:
    - Loading tiles from various formats
    - Stitching tiles into larger images
    - Georeferencing
    - Normalization for model input
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tile_size = self.config['tile2net']['inference']['tile_size']
        self.zoom_level = self.config['tile2net']['inference']['zoom_level']
    
    def load_tile_index(self, index_path: Path) -> List[TileInfo]:
        """Load tile index from JSON file."""
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        tiles = []
        for t in index['tiles']:
            tiles.append(TileInfo(
                path=Path(t['path']),
                z=t['z'],
                x=t['x'],
                y=t['y']
            ))
        
        return tiles
    
    def load_tiles_from_directory(self, tile_dir: Path) -> List[TileInfo]:
        """Load tile info from directory naming convention."""
        tiles = []
        
        for path in tile_dir.glob("tile_*_*_*.png"):
            parts = path.stem.split('_')
            if len(parts) == 4:
                tiles.append(TileInfo(
                    path=path,
                    z=int(parts[1]),
                    x=int(parts[2]),
                    y=int(parts[3])
                ))
        
        return tiles
    
    def stitch_tiles(
        self,
        tiles: List[TileInfo],
        output_path: Path,
        tile_step: int = 2,
        save_geotiff: bool = True
    ) -> List[Path]:
        """
        Stitch small tiles into larger inference tiles.
        
        Args:
            tiles: List of tile info
            output_path: Output directory
            tile_step: Number of tiles to stitch (tile_step x tile_step)
            save_geotiff: Whether to save georeferenced GeoTIFFs
            
        Returns:
            List of stitched tile paths
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Group tiles by position
        tile_dict = {(t.x, t.y): t for t in tiles}
        
        # Find tile grid bounds
        x_coords = [t.x for t in tiles]
        y_coords = [t.y for t in tiles]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        logger.info(f"Tile grid: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        
        stitched_paths = []
        
        # Iterate over grid with step size
        for x_start in tqdm(range(x_min, x_max + 1, tile_step), desc="Stitching X"):
            for y_start in range(y_min, y_max + 1, tile_step):
                
                # Collect tiles for this stitch
                stitch_tiles = []
                for dx in range(tile_step):
                    for dy in range(tile_step):
                        key = (x_start + dx, y_start + dy)
                        if key in tile_dict:
                            stitch_tiles.append((dx, dy, tile_dict[key]))
                
                if len(stitch_tiles) < tile_step * tile_step * 0.5:
                    # Skip if less than half the tiles available
                    continue
                
                # Load and stitch
                stitched = self._stitch_tile_group(stitch_tiles, tile_step)
                
                if stitched is not None:
                    # Get combined bounds
                    z = tiles[0].z
                    west1, south1, _, _ = tile_to_bounds(z, x_start, y_start + tile_step - 1)
                    _, _, east1, north1 = tile_to_bounds(z, x_start + tile_step - 1, y_start)
                    bounds = (west1, south1, east1, north1)
                    
                    # Save
                    out_name = f"stitched_{z}_{x_start}_{y_start}.tif"
                    out_path = output_path / out_name
                    
                    if save_geotiff:
                        self._save_geotiff(stitched, out_path, bounds)
                    else:
                        Image.fromarray(stitched).save(out_path.with_suffix('.png'))
                    
                    stitched_paths.append(out_path)
        
        logger.info(f"Created {len(stitched_paths)} stitched tiles")
        
        return stitched_paths
    
    def _stitch_tile_group(
        self,
        tiles: List[Tuple[int, int, TileInfo]],
        tile_step: int
    ) -> Optional[np.ndarray]:
        """Stitch a group of tiles into a single image."""
        
        # Determine individual tile size from first tile
        first_tile = tiles[0][2]
        try:
            img = Image.open(first_tile.path)
            single_size = img.size[0]  # Assume square
        except Exception as e:
            logger.error(f"Failed to open tile {first_tile.path}: {e}")
            return None
        
        # Create output array
        total_size = single_size * tile_step
        stitched = np.zeros((total_size, total_size, 3), dtype=np.uint8)
        
        for dx, dy, tile_info in tiles:
            try:
                img = Image.open(tile_info.path).convert('RGB')
                arr = np.array(img)
                
                # Calculate position (y is inverted in image coordinates)
                x_pos = dx * single_size
                y_pos = dy * single_size
                
                stitched[y_pos:y_pos + single_size, x_pos:x_pos + single_size] = arr
                
            except Exception as e:
                logger.warning(f"Failed to load tile {tile_info.path}: {e}")
        
        return stitched
    
    def _save_geotiff(
        self,
        array: np.ndarray,
        output_path: Path,
        bounds: Tuple[float, float, float, float]
    ):
        """Save array as georeferenced GeoTIFF."""
        
        height, width = array.shape[:2]
        west, south, east, north = bounds
        
        transform = from_bounds(west, south, east, north, width, height)
        
        # Handle RGB vs grayscale
        if len(array.shape) == 3:
            count = array.shape[2]
            # Transpose to (bands, height, width)
            array = np.transpose(array, (2, 0, 1))
        else:
            count = 1
            array = array[np.newaxis, :, :]
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=array.dtype,
            crs=CRS.from_epsg(4326),
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(array)
    
    def prepare_for_inference(
        self,
        tile_path: Path,
        target_size: int = 512
    ) -> Tuple[np.ndarray, Dict]:
        """
        Prepare a tile for Tile2Net inference.
        
        Args:
            tile_path: Path to tile image
            target_size: Target size for model input
            
        Returns:
            Tuple of (normalized array, metadata dict)
        """
        # Load image
        if tile_path.suffix in ['.tif', '.tiff']:
            with rasterio.open(tile_path) as src:
                img = src.read()
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                bounds = src.bounds
                crs = src.crs
        else:
            img = np.array(Image.open(tile_path).convert('RGB'))
            bounds = None
            crs = None
        
        # Resize if needed
        if img.shape[0] != target_size or img.shape[1] != target_size:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
            img = np.array(pil_img)
        
        # Normalize to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0
        
        metadata = {
            'original_path': str(tile_path),
            'original_size': img.shape[:2],
            'target_size': target_size,
            'bounds': bounds,
            'crs': str(crs) if crs else None
        }
        
        return img_normalized, metadata
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        stitch: bool = True,
        tile_step: int = 2
    ) -> Dict:
        """
        Process all tiles in a directory.
        
        Args:
            input_dir: Input directory with raw tiles
            output_dir: Output directory for processed tiles
            stitch: Whether to stitch tiles
            tile_step: Stitching step size
            
        Returns:
            Processing summary dict
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tiles
        index_path = input_dir / "tile_index.json"
        if index_path.exists():
            tiles = self.load_tile_index(index_path)
        else:
            tiles = self.load_tiles_from_directory(input_dir)
        
        logger.info(f"Found {len(tiles)} tiles in {input_dir}")
        
        if len(tiles) == 0:
            logger.warning("No tiles found!")
            return {"status": "error", "message": "No tiles found"}
        
        results = {
            "input_tiles": len(tiles),
            "output_dir": str(output_dir),
            "stitched": []
        }
        
        if stitch:
            stitched_dir = output_dir / "stitched"
            stitched_paths = self.stitch_tiles(tiles, stitched_dir, tile_step)
            results["stitched"] = [str(p) for p in stitched_paths]
            results["stitched_count"] = len(stitched_paths)
        
        # Save processing manifest
        manifest_path = output_dir / "processing_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    """CLI for tile preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess tiles for inference")
    
    parser.add_argument('--input', '-i', required=True, help='Input tile directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file')
    parser.add_argument('--no-stitch', action='store_true', help='Skip stitching')
    parser.add_argument('--tile-step', type=int, default=2, help='Tiles to stitch together')
    
    args = parser.parse_args()
    
    preprocessor = TilePreprocessor(args.config)
    
    results = preprocessor.process_directory(
        Path(args.input),
        Path(args.output),
        stitch=not args.no_stitch,
        tile_step=args.tile_step
    )
    
    print(f"\nProcessing complete:")
    print(f"  Input tiles: {results.get('input_tiles', 0)}")
    print(f"  Stitched tiles: {results.get('stitched_count', 0)}")
    print(f"  Output: {results.get('output_dir')}")


if __name__ == "__main__":
    main()