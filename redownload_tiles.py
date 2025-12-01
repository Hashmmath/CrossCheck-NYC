"""
Redownload and Verify NYC Orthoimagery Tiles
=============================================

This script:
1. Checks existing tiles for validity
2. Redownloads corrupted/empty tiles
3. Tries multiple tile servers as fallback

Usage:
    python redownload_tiles.py
"""

import os
import sys
import math
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io
import logging
from tqdm import tqdm
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Brooklyn test area bounding box (downtown Brooklyn)
TEST_BBOX = {
    'west': -73.9900,
    'south': 40.6870,
    'east': -73.9750,
    'north': 40.6970
}

# Tile servers to try (in order)
TILE_SERVERS = [
    {
        "name": "NYC ArcGIS 2022",
        "url": "https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_-_2022/MapServer/tile/{z}/{y}/{x}",
        "format": "png"
    },
    {
        "name": "NYC ArcGIS 2020",
        "url": "https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_-_2020/MapServer/tile/{z}/{y}/{x}",
        "format": "png"
    },
    {
        "name": "NYC Map Tiles (Aerial)",
        "url": "https://maps.nyc.gov/xyz/1.0.0/photo/{z}/{x}/{y}.png8",
        "format": "png"
    },
    {
        "name": "ESRI World Imagery",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "format": "jpg"
    }
]


def lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to tile coordinates."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def get_tile_indices(bbox, zoom):
    """Get all tile indices for a bounding box."""
    x_min, y_max = lat_lon_to_tile(bbox['south'], bbox['west'], zoom)
    x_max, y_min = lat_lon_to_tile(bbox['north'], bbox['east'], zoom)
    
    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append((zoom, x, y))
    
    return tiles


def is_valid_image(file_path):
    """Check if a file is a valid image."""
    try:
        if not file_path.exists():
            return False
        
        # Check file size (should be at least a few KB for a real image)
        if file_path.stat().st_size < 1000:
            return False
        
        # Try to open as image
        with Image.open(file_path) as img:
            img.verify()
        
        return True
    except:
        return False


def download_tile(z, x, y, output_path, servers=TILE_SERVERS):
    """Download a single tile, trying multiple servers."""
    
    for server in servers:
        url = server["url"].format(z=z, x=x, y=y)
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                content = response.content
                
                # Verify it's actually an image
                try:
                    img = Image.open(io.BytesIO(content))
                    img.verify()
                    
                    # Save the valid image
                    output_path.write_bytes(content)
                    return True, server["name"]
                except:
                    continue
            
        except requests.exceptions.RequestException as e:
            continue
    
    return False, None


def check_and_download_tiles(output_dir, bbox, zoom=19):
    """Check existing tiles and redownload as needed."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get required tiles
    tiles = get_tile_indices(bbox, zoom)
    logger.info(f"Need {len(tiles)} tiles for zoom {zoom}")
    
    # Check existing tiles
    valid_tiles = []
    invalid_tiles = []
    missing_tiles = []
    
    for z, x, y in tiles:
        tile_path = output_dir / f"tile_{z}_{x}_{y}.png"
        
        if tile_path.exists():
            if is_valid_image(tile_path):
                valid_tiles.append((z, x, y))
            else:
                invalid_tiles.append((z, x, y))
                # Remove invalid file
                tile_path.unlink()
        else:
            missing_tiles.append((z, x, y))
    
    logger.info(f"Tile status:")
    logger.info(f"  Valid: {len(valid_tiles)}")
    logger.info(f"  Invalid/Corrupted: {len(invalid_tiles)}")
    logger.info(f"  Missing: {len(missing_tiles)}")
    
    # Tiles to download
    to_download = invalid_tiles + missing_tiles
    
    if len(to_download) == 0:
        logger.info("All tiles are valid!")
        return len(valid_tiles)
    
    logger.info(f"Downloading {len(to_download)} tiles...")
    
    # Test which server works
    logger.info("Testing tile servers...")
    working_server = None
    test_z, test_x, test_y = tiles[0]
    
    for server in TILE_SERVERS:
        url = server["url"].format(z=test_z, x=test_x, y=test_y)
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                try:
                    img = Image.open(io.BytesIO(response.content))
                    img.verify()
                    working_server = server
                    logger.info(f"Using server: {server['name']}")
                    break
                except:
                    pass
        except:
            pass
    
    if working_server is None:
        logger.error("No working tile server found!")
        logger.error("")
        logger.error("Please try manually downloading tiles from:")
        logger.error("  https://maps.nyc.gov/tiles/")
        logger.error("  Or use QGIS to export tiles from NYC ortho WMS")
        return len(valid_tiles)
    
    # Download tiles
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        
        for z, x, y in to_download:
            tile_path = output_dir / f"tile_{z}_{x}_{y}.png"
            future = executor.submit(download_tile, z, x, y, tile_path, [working_server])
            futures[future] = (z, x, y, tile_path)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            z, x, y, tile_path = futures[future]
            try:
                success, server_used = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                logger.error(f"Failed tile ({z}, {x}, {y}): {e}")
            
            # Rate limiting
            time.sleep(0.1)
    
    logger.info(f"Download complete:")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {fail_count}")
    logger.info(f"  Total valid: {len(valid_tiles) + success_count}")
    
    return len(valid_tiles) + success_count


def main():
    """Main function."""
    
    # Use test area for development
    output_dir = Path("data/raw/ortho")
    
    logger.info("=" * 60)
    logger.info("NYC Orthoimagery Tile Downloader")
    logger.info("=" * 60)
    
    # Check and download
    total_valid = check_and_download_tiles(
        output_dir=output_dir,
        bbox=TEST_BBOX,
        zoom=19
    )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Complete! {total_valid} valid tiles in {output_dir}")
    logger.info("=" * 60)
    
    if total_valid == 0:
        logger.error("")
        logger.error("No valid tiles downloaded. Alternative options:")
        logger.error("")
        logger.error("1. Use QGIS to download tiles manually:")
        logger.error("   - Add XYZ tile layer: https://maps.nyc.gov/xyz/1.0.0/photo/{z}/{x}/{y}.png8")
        logger.error("   - Export to GeoTiff for your area of interest")
        logger.error("")
        logger.error("2. Download orthoimagery from NYC GIS:")
        logger.error("   - https://gis.ny.gov/new-york-city-orthoimagery-downloads")
        logger.error("   - Select Brooklyn and download GeoTiff")


if __name__ == "__main__":
    main()