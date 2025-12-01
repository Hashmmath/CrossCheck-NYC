"""
Process Manually Downloaded LION Data
======================================

This script processes the NYC LION dataset (Shapefile or Geodatabase format)
and filters it to Brooklyn only.

Usage:
    1. Download LION from NYC Open Data:
       https://data.cityofnewyork.us/City-Government/LION/2v4z-66xt
       Click Export -> Shapefile (downloads nyclion.zip ~44.5 MB)
       
       OR download from NYC Planning:
       https://www.nyc.gov/site/planning/data-maps/open-data/dwn-lion.page
    
    2. Extract the zip file to: data/raw/lion/
       You should have: data/raw/lion/lion/lion.shp (or similar structure)
    
    3. Run this script:
       python process_lion.py

The script will:
- Find the LION shapefile or geodatabase
- Filter to Brooklyn (BoroCode = 3)
- Save as lion_brooklyn.geojson
"""

import sys
import os
from pathlib import Path
import logging
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_lion_file(base_dir: Path):
    """
    Find LION data file in various possible locations/formats.
    
    Returns:
        Path to the LION file (shapefile, geodatabase, or geojson)
    """
    base_dir = Path(base_dir)
    
    # Check for zip file first and extract if found
    zip_files = list(base_dir.glob("*.zip")) + list(base_dir.glob("**/nyclion.zip"))
    for zip_path in zip_files:
        logger.info(f"Found zip file: {zip_path}")
        extract_dir = base_dir / "extracted"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Extracted to: {extract_dir}")
    
    # Search patterns for LION data
    search_patterns = [
        # Shapefiles
        "**/lion.shp",
        "**/LION.shp",
        "**/lion/lion.shp",
        "**/nyclion/lion.shp",
        "**/*lion*.shp",
        # Geodatabases
        "**/lion.gdb",
        "**/LION.gdb",
        "**/nyclion.gdb",
        "**/*.gdb",
        # GeoJSON (if already converted)
        "**/lion*.geojson",
        "**/lion*.json",
    ]
    
    for pattern in search_patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            # Prefer .shp over .gdb for easier processing
            shp_matches = [m for m in matches if m.suffix == '.shp']
            if shp_matches:
                return shp_matches[0]
            return matches[0]
    
    return None


def process_lion():
    """Find, load, filter LION data to Brooklyn and save."""
    
    try:
        import geopandas as gpd
    except ImportError:
        logger.error("geopandas not installed. Run: pip install geopandas")
        sys.exit(1)
    
    # Paths
    lion_dir = Path("data/raw/lion")
    output_path = lion_dir / "lion_brooklyn.geojson"
    
    # Check if output already exists
    if output_path.exists():
        logger.info(f"Brooklyn LION data already exists: {output_path}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            logger.info("Skipping processing.")
            return output_path
    
    # Create directory if needed
    lion_dir.mkdir(parents=True, exist_ok=True)
    
    # Find LION file
    logger.info(f"Searching for LION data in {lion_dir}...")
    lion_path = find_lion_file(lion_dir)
    
    if lion_path is None:
        logger.error("=" * 60)
        logger.error("LION data not found!")
        logger.error("=" * 60)
        logger.error("")
        logger.error("Please download LION data:")
        logger.error("")
        logger.error("Option 1 - NYC Open Data:")
        logger.error("  1. Go to: https://data.cityofnewyork.us/City-Government/LION/2v4z-66xt")
        logger.error("  2. Click 'Export' -> 'Shapefile'")
        logger.error("  3. Save nyclion.zip to: data/raw/lion/")
        logger.error("")
        logger.error("Option 2 - NYC Planning (BYTES of Big Apple):")
        logger.error("  1. Go to: https://www.nyc.gov/site/planning/data-maps/open-data/dwn-lion.page")
        logger.error("  2. Download the LION zip file")
        logger.error("  3. Save to: data/raw/lion/")
        logger.error("")
        logger.error("After downloading, run this script again.")
        sys.exit(1)
    
    logger.info(f"Found LION data: {lion_path}")
    
    # Get file size
    if lion_path.is_file():
        file_size_mb = lion_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.1f} MB")
    
    # Load data
    logger.info("Loading LION data (this may take 1-2 minutes)...")
    
    try:
        if lion_path.suffix == '.gdb' or str(lion_path).endswith('.gdb'):
            # Geodatabase - need to specify layer
            import fiona
            layers = fiona.listlayers(lion_path)
            logger.info(f"Geodatabase layers: {layers}")
            # Usually the main layer is 'lion' or first layer
            layer_name = 'lion' if 'lion' in layers else layers[0]
            gdf = gpd.read_file(lion_path, layer=layer_name)
        else:
            # Shapefile or GeoJSON
            gdf = gpd.read_file(lion_path)
    except Exception as e:
        logger.error(f"Failed to load LION data: {e}")
        logger.error("")
        logger.error("If you're getting a driver error, try installing additional libraries:")
        logger.error("  pip install pyogrio fiona")
        sys.exit(1)
    
    logger.info(f"Loaded {len(gdf):,} total street segments")
    
    # Show available columns
    logger.info(f"Columns: {list(gdf.columns)[:10]}...")  # First 10 columns
    
    # Filter to Brooklyn
    logger.info("Filtering to Brooklyn (BoroCode = 3)...")
    
    brooklyn_gdf = None
    
    # Try different column names for borough filtering
    # LION uses various column names depending on version
    borough_columns = ['BoroCode', 'borocode', 'BOROCODE', 'LBoro', 'RBoro', 'LBORO', 'RBORO']
    
    for col in borough_columns:
        if col in gdf.columns:
            # Brooklyn is BoroCode 3
            # Try both string and integer comparison
            try:
                brooklyn_gdf = gdf[gdf[col] == '3'].copy()
                if len(brooklyn_gdf) == 0:
                    brooklyn_gdf = gdf[gdf[col] == 3].copy()
            except:
                brooklyn_gdf = gdf[gdf[col].astype(str) == '3'].copy()
            
            if len(brooklyn_gdf) > 0:
                logger.info(f"Filtered by {col}='3': {len(brooklyn_gdf):,} segments")
                break
    
    # If no borough column found, use spatial filter
    if brooklyn_gdf is None or len(brooklyn_gdf) == 0:
        logger.warning("No borough code column found, using spatial filter...")
        from shapely.geometry import box
        
        # Brooklyn approximate bounding box (WGS84)
        brooklyn_bbox = box(-74.0424, 40.5707, -73.8334, 40.7395)
        
        # Check CRS and reproject if needed
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            logger.info(f"Reprojecting from {gdf.crs} to WGS84...")
            gdf_wgs84 = gdf.to_crs('EPSG:4326')
            brooklyn_gdf = gdf_wgs84[gdf_wgs84.intersects(brooklyn_bbox)].copy()
        else:
            brooklyn_gdf = gdf[gdf.intersects(brooklyn_bbox)].copy()
        
        logger.info(f"Filtered by spatial intersection: {len(brooklyn_gdf):,} segments")
    
    if brooklyn_gdf is None or len(brooklyn_gdf) == 0:
        logger.error("No Brooklyn data found after filtering!")
        sys.exit(1)
    
    # Ensure output is in WGS84 (EPSG:4326) for compatibility
    if brooklyn_gdf.crs and brooklyn_gdf.crs.to_epsg() != 4326:
        logger.info(f"Converting from {brooklyn_gdf.crs} to WGS84...")
        brooklyn_gdf = brooklyn_gdf.to_crs('EPSG:4326')
    
    # Save filtered data
    logger.info(f"Saving Brooklyn data to {output_path}...")
    brooklyn_gdf.to_file(output_path, driver='GeoJSON')
    
    # Report file size
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("LION data processing complete!")
    logger.info("=" * 60)
    logger.info(f"Output file: {output_path}")
    logger.info(f"Total segments: {len(brooklyn_gdf):,}")
    logger.info(f"File size: {output_size_mb:.1f} MB")
    logger.info("=" * 60)
    
    # Offer to clean up
    print("")
    cleanup = input("Remove original downloaded files to save space? (y/n): ").strip().lower()
    if cleanup == 'y':
        # Remove extracted folder and zip files
        import shutil
        extracted_dir = lion_dir / "extracted"
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
            logger.info(f"Removed {extracted_dir}")
        
        for zip_file in lion_dir.glob("*.zip"):
            zip_file.unlink()
            logger.info(f"Removed {zip_file}")
        
        # Remove shapefile components if they exist outside extracted
        for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg', '.sbn', '.sbx']:
            for f in lion_dir.glob(f"*{ext}"):
                if 'brooklyn' not in f.name.lower():
                    f.unlink()
    
    return output_path


if __name__ == "__main__":
    process_lion()