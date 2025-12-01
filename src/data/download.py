"""
Data Download Utilities for Brooklyn Crosswalk QA
=================================================

Downloads and prepares the following datasets:
1. NYC Orthoimagery (Brooklyn tiles)
2. LION Street Centerlines
3. OSM Crossings
4. Vision Zero Enhanced Crossings (optional)

Usage:
    python download.py --all                    # Download all datasets
    python download.py --ortho --area downtown  # Download ortho for specific area
    python download.py --lion                   # Download LION centerlines only
    python download.py --osm                    # Download OSM crossings only
"""

import os
import sys
import json
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import yaml
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box container."""
    west: float
    south: float
    east: float
    north: float
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)
    
    def to_polygon(self):
        return box(self.west, self.south, self.east, self.north)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'BoundingBox':
        return cls(
            west=d['west'],
            south=d['south'],
            east=d['east'],
            north=d['north']
        )


class DataDownloader:
    """
    Main class for downloading all required datasets.
    
    Handles:
    - NYC Orthoimagery tiles via NYC Map Tiles API
    - LION street centerlines from NYC Open Data
    - OSM crossing data from Geofabrik
    - Vision Zero enhanced crossings from Data.gov
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize downloader with configuration."""
        self.config = self._load_config(config_path)
        self.paths = self.config['paths']
        self._setup_directories()
        
        # Brooklyn bounding box
        self.brooklyn_bbox = BoundingBox.from_dict(
            self.config['location']['bbox']
        )
        
        # Test area for development
        self.test_bbox = BoundingBox.from_dict(
            self.config['location']['test_area']['bbox']
        )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """Create necessary directories."""
        for key, path in self.paths['raw'].items():
            Path(path).mkdir(parents=True, exist_ok=True)
        for key, path in self.paths['processed'].items():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # NYC Orthoimagery Download
    # =========================================================================
    
    def download_ortho_tiles(
        self,
        bbox: Optional[BoundingBox] = None,
        zoom: int = 19,
        output_dir: Optional[str] = None,
        max_tiles: int = 1000
    ) -> List[Path]:
        """
        Download orthoimagery tiles from NYC Map Tiles service.
        
        Args:
            bbox: Bounding box (defaults to test area)
            zoom: Zoom level (18-20, recommend 19)
            output_dir: Output directory
            max_tiles: Maximum number of tiles to download
            
        Returns:
            List of downloaded tile paths
        """
        bbox = bbox or self.test_bbox
        output_dir = Path(output_dir or self.paths['raw']['ortho'])
        
        # Calculate tile indices for bounding box
        tiles = self._get_tile_indices(bbox, zoom)
        
        if len(tiles) > max_tiles:
            logger.warning(
                f"Requested {len(tiles)} tiles exceeds max_tiles={max_tiles}. "
                f"Limiting to {max_tiles} tiles."
            )
            tiles = tiles[:max_tiles]
        
        logger.info(f"Downloading {len(tiles)} tiles at zoom {zoom}")
        
        # NYC aerial tile URL template
        # Using NYC's ArcGIS tile service
        base_url = "https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_-_2022/MapServer/tile"
        
        downloaded = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for z, x, y in tiles:
                url = f"{base_url}/{z}/{y}/{x}"
                tile_path = output_dir / f"tile_{z}_{x}_{y}.png"
                
                if tile_path.exists():
                    downloaded.append(tile_path)
                    continue
                
                future = executor.submit(
                    self._download_tile, url, tile_path
                )
                futures[future] = tile_path
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading tiles"):
                tile_path = futures[future]
                try:
                    if future.result():
                        downloaded.append(tile_path)
                except Exception as e:
                    logger.error(f"Failed to download {tile_path}: {e}")
        
        logger.info(f"Downloaded {len(downloaded)} tiles to {output_dir}")
        
        # Save tile index
        self._save_tile_index(downloaded, output_dir / "tile_index.json", zoom)
        
        return downloaded
    
    def _get_tile_indices(
        self, 
        bbox: BoundingBox, 
        zoom: int
    ) -> List[Tuple[int, int, int]]:
        """
        Convert bounding box to tile indices (z, x, y).
        
        Uses Web Mercator tiling scheme.
        """
        import math
        
        def lat_lon_to_tile(lat, lon, zoom):
            n = 2 ** zoom
            x = int((lon + 180.0) / 360.0 * n)
            lat_rad = math.radians(lat)
            y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            return x, y
        
        # Get tile range
        x_min, y_max = lat_lon_to_tile(bbox.south, bbox.west, zoom)
        x_max, y_min = lat_lon_to_tile(bbox.north, bbox.east, zoom)
        
        tiles = []
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                tiles.append((zoom, x, y))
        
        return tiles
    
    def _download_tile(self, url: str, output_path: Path) -> bool:
        """Download a single tile."""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                output_path.write_bytes(response.content)
                return True
            else:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return False
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _save_tile_index(
        self, 
        tile_paths: List[Path], 
        index_path: Path,
        zoom: int
    ):
        """Save tile index with metadata."""
        index = {
            "zoom": zoom,
            "count": len(tile_paths),
            "tiles": [
                {
                    "path": str(p),
                    "name": p.name,
                    "z": int(p.stem.split('_')[1]),
                    "x": int(p.stem.split('_')[2]),
                    "y": int(p.stem.split('_')[3])
                }
                for p in tile_paths
            ]
        }
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    # =========================================================================
    # LION Centerlines Download
    # =========================================================================
    
    def download_lion(
        self,
        output_dir: Optional[str] = None,
        filter_brooklyn: bool = True
    ) -> Path:
        """
        Download NYC LION street centerlines.
        
        Args:
            output_dir: Output directory
            filter_brooklyn: Whether to filter to Brooklyn only
            
        Returns:
            Path to downloaded/filtered shapefile
        """
        output_dir = Path(output_dir or self.paths['raw']['lion'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading LION street centerlines...")
        
        output_path = output_dir / "lion_brooklyn.geojson"
        
        if output_path.exists():
            logger.info(f"LION data already exists at {output_path}")
            return output_path
        
        # Try multiple methods to get LION data
        gdf = None
        
        # Method 1: NYC Open Data Socrata API (JSON format with pagination)
        try:
            logger.info("Trying NYC Open Data Socrata API...")
            gdf = self._download_lion_socrata(filter_brooklyn)
        except Exception as e:
            logger.warning(f"Socrata API failed: {e}")
        
        # Method 2: Direct GeoJSON export (may have size limits)
        if gdf is None:
            try:
                logger.info("Trying direct GeoJSON export...")
                gdf = self._download_lion_geojson_export(output_dir, filter_brooklyn)
            except Exception as e:
                logger.warning(f"GeoJSON export failed: {e}")
        
        # Method 3: Use NYC Centerline (CSCL) as alternative
        if gdf is None:
            try:
                logger.info("Trying NYC Centerline (CSCL) as alternative...")
                gdf = self._download_centerline_alternative(filter_brooklyn)
            except Exception as e:
                logger.warning(f"Centerline alternative failed: {e}")
        
        # Method 4: Create minimal placeholder from bbox
        if gdf is None:
            logger.warning("All download methods failed. Creating placeholder...")
            gdf = self._create_lion_placeholder()
        
        # Save result
        if gdf is not None and len(gdf) > 0:
            gdf.to_file(output_path, driver='GeoJSON')
            logger.info(f"Saved {len(gdf)} street segments to {output_path}")
        else:
            logger.error("Failed to download LION data")
            return None
        
        return output_path
    
    def _download_lion_socrata(self, filter_brooklyn: bool) -> gpd.GeoDataFrame:
        """Download LION via Socrata API with pagination."""
        
        # Socrata API endpoint for LION
        base_url = "https://data.cityofnewyork.us/resource/2v4z-66xt.geojson"
        
        # Build query - filter to Brooklyn (BoroCode = 3) to reduce data size
        if filter_brooklyn:
            # Try with borough filter
            params = {
                "$where": "borocode = '3'",
                "$limit": 50000
            }
        else:
            params = {"$limit": 50000}
        
        response = requests.get(base_url, params=params, timeout=300)
        
        if response.status_code != 200:
            raise Exception(f"API returned status {response.status_code}")
        
        # Check if response is valid GeoJSON
        try:
            data = response.json()
            if 'type' not in data or data.get('type') != 'FeatureCollection':
                raise Exception("Response is not valid GeoJSON")
            
            if len(data.get('features', [])) == 0:
                raise Exception("No features returned")
            
            gdf = gpd.GeoDataFrame.from_features(data['features'], crs='EPSG:4326')
            logger.info(f"Downloaded {len(gdf)} features via Socrata API")
            return gdf
            
        except json.JSONDecodeError:
            raise Exception("Response is not valid JSON")
    
    def _download_lion_geojson_export(
        self, 
        output_dir: Path,
        filter_brooklyn: bool
    ) -> gpd.GeoDataFrame:
        """Download via GeoJSON export endpoint."""
        
        # Try the export endpoint
        lion_url = "https://data.cityofnewyork.us/api/geospatial/2v4z-66xt?method=export&format=GeoJSON"
        
        response = requests.get(lion_url, stream=True, timeout=300)
        
        if response.status_code != 200:
            raise Exception(f"Export returned status {response.status_code}")
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'html' in content_type.lower():
            raise Exception("Received HTML instead of GeoJSON")
        
        temp_path = output_dir / "lion_temp.geojson"
        
        # Download to temp file
        total_size = 0
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                total_size += len(chunk)
        
        # Check file size (should be at least a few KB for valid GeoJSON)
        if total_size < 1000:
            temp_path.unlink()
            raise Exception(f"Downloaded file too small ({total_size} bytes)")
        
        # Try to read
        try:
            gdf = gpd.read_file(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        
        # Filter to Brooklyn if needed
        if filter_brooklyn and len(gdf) > 0:
            if 'borocode' in gdf.columns:
                gdf = gdf[gdf['borocode'] == '3']
            elif 'boro' in gdf.columns:
                gdf = gdf[gdf['boro'].str.upper().str.contains('BROOKLYN', na=False)]
            else:
                brooklyn_poly = self.brooklyn_bbox.to_polygon()
                gdf = gdf[gdf.intersects(brooklyn_poly)]
        
        return gdf
    
    def _download_centerline_alternative(self, filter_brooklyn: bool) -> gpd.GeoDataFrame:
        """Download NYC Centerline (CSCL) as alternative to LION."""
        
        # CSCL endpoint
        base_url = "https://data.cityofnewyork.us/resource/3mf9-qshr.geojson"
        
        if filter_brooklyn:
            params = {
                "$where": "borocode = '3'",
                "$limit": 50000
            }
        else:
            params = {"$limit": 50000}
        
        response = requests.get(base_url, params=params, timeout=300)
        
        if response.status_code != 200:
            raise Exception(f"CSCL API returned status {response.status_code}")
        
        data = response.json()
        
        if 'type' not in data or len(data.get('features', [])) == 0:
            raise Exception("No valid features in CSCL response")
        
        gdf = gpd.GeoDataFrame.from_features(data['features'], crs='EPSG:4326')
        logger.info(f"Downloaded {len(gdf)} features from CSCL")
        return gdf
    
    def _create_lion_placeholder(self) -> gpd.GeoDataFrame:
        """Create a minimal placeholder GeoDataFrame."""
        from shapely.geometry import LineString
        
        # Create a simple grid of lines covering Brooklyn bbox
        bbox = self.brooklyn_bbox
        
        lines = []
        # Create some horizontal lines
        for i in range(5):
            lat = bbox.south + (bbox.north - bbox.south) * i / 4
            line = LineString([(bbox.west, lat), (bbox.east, lat)])
            lines.append({'geometry': line, 'type': 'placeholder', 'id': f'h_{i}'})
        
        # Create some vertical lines
        for i in range(5):
            lon = bbox.west + (bbox.east - bbox.west) * i / 4
            line = LineString([(lon, bbox.south), (lon, bbox.north)])
            lines.append({'geometry': line, 'type': 'placeholder', 'id': f'v_{i}'})
        
        gdf = gpd.GeoDataFrame(lines, crs='EPSG:4326')
        
        logger.warning(
            "Created placeholder LION data. For accurate analysis, please download "
            "LION data manually from: https://data.cityofnewyork.us/City-Government/LION/2v4z-66xt"
        )
        
        return gdf
    
    # =========================================================================
    # OSM Crossings Download
    # =========================================================================
    
    def download_osm_crossings(
        self,
        output_dir: Optional[str] = None,
        use_osmnx: bool = True
    ) -> Path:
        """
        Download OpenStreetMap crossing data for Brooklyn.
        
        Args:
            output_dir: Output directory
            use_osmnx: Whether to use OSMnx (simpler) or Geofabrik
            
        Returns:
            Path to crossings GeoJSON
        """
        output_dir = Path(output_dir or self.paths['raw']['osm'])
        output_path = output_dir / "osm_crossings_brooklyn.geojson"
        
        if output_path.exists():
            logger.info(f"OSM crossings already exist at {output_path}")
            return output_path
        
        if use_osmnx:
            return self._download_osm_via_osmnx(output_path)
        else:
            return self._download_osm_via_geofabrik(output_dir, output_path)
    
    def _download_osm_via_osmnx(self, output_path: Path) -> Path:
        """Download OSM crossings using OSMnx (Overpass API)."""
        try:
            import osmnx as ox
        except ImportError:
            logger.error("OSMnx not installed. Run: pip install osmnx")
            raise
        
        logger.info("Downloading OSM crossings via Overpass API...")
        
        # Create polygon for Brooklyn
        brooklyn_poly = self.brooklyn_bbox.to_polygon()
        
        # Query for crossings
        # OSMnx custom query for highway=crossing nodes
        tags = {'highway': 'crossing'}
        
        try:
            # Get crossing nodes
            crossings = ox.features_from_polygon(
                brooklyn_poly,
                tags=tags
            )
            
            logger.info(f"Found {len(crossings)} crossings")
            
            # Convert to GeoDataFrame if needed
            if not isinstance(crossings, gpd.GeoDataFrame):
                crossings = gpd.GeoDataFrame(crossings)
            
            # Ensure CRS
            if crossings.crs is None:
                crossings = crossings.set_crs('EPSG:4326')
            
            # Save
            crossings.to_file(output_path, driver='GeoJSON')
            
            logger.info(f"Saved {len(crossings)} crossings to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"OSMnx download failed: {e}")
            logger.info("Trying alternative method...")
            return self._download_osm_via_overpass_direct(output_path)
    
    def _download_osm_via_overpass_direct(self, output_path: Path) -> Path:
        """Direct Overpass API query for crossings."""
        
        bbox = self.brooklyn_bbox
        
        # Overpass QL query for crossings
        query = f"""
        [out:json][timeout:300];
        (
          node["highway"="crossing"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});
          way["highway"="crossing"]({bbox.south},{bbox.west},{bbox.north},{bbox.east});
        );
        out body;
        >;
        out skel qt;
        """
        
        logger.info("Querying Overpass API...")
        
        response = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query,
            timeout=300
        )
        
        if response.status_code != 200:
            raise Exception(f"Overpass API error: {response.status_code}")
        
        data = response.json()
        
        # Convert to GeoDataFrame
        features = []
        for element in data.get('elements', []):
            if element['type'] == 'node':
                features.append({
                    'geometry': Point(element['lon'], element['lat']),
                    'osm_id': element['id'],
                    'osm_type': 'node',
                    **element.get('tags', {})
                })
        
        gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')
        gdf.to_file(output_path, driver='GeoJSON')
        
        logger.info(f"Saved {len(gdf)} crossings to {output_path}")
        
        return output_path
    
    def _download_osm_via_geofabrik(
        self, 
        output_dir: Path,
        output_path: Path
    ) -> Path:
        """Download from Geofabrik and extract crossings."""
        
        # Download New York state extract
        geofabrik_url = "https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf"
        pbf_path = output_dir / "new-york-latest.osm.pbf"
        
        if not pbf_path.exists():
            logger.info("Downloading New York OSM extract from Geofabrik...")
            response = requests.get(geofabrik_url, stream=True, timeout=600)
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(pbf_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Extract crossings using osmium
        logger.info("Extracting crossings from PBF...")
        
        try:
            import osmium
            
            class CrossingHandler(osmium.SimpleHandler):
                def __init__(self):
                    super().__init__()
                    self.crossings = []
                
                def node(self, n):
                    if 'highway' in n.tags and n.tags['highway'] == 'crossing':
                        self.crossings.append({
                            'geometry': Point(n.location.lon, n.location.lat),
                            'osm_id': n.id,
                            **dict(n.tags)
                        })
            
            handler = CrossingHandler()
            handler.apply_file(str(pbf_path), locations=True)
            
            # Filter to Brooklyn
            gdf = gpd.GeoDataFrame(handler.crossings, crs='EPSG:4326')
            brooklyn_poly = self.brooklyn_bbox.to_polygon()
            gdf = gdf[gdf.within(brooklyn_poly)]
            
            gdf.to_file(output_path, driver='GeoJSON')
            
            logger.info(f"Saved {len(gdf)} Brooklyn crossings to {output_path}")
            
            return output_path
            
        except ImportError:
            logger.warning("osmium not available, falling back to Overpass API")
            return self._download_osm_via_overpass_direct(output_path)
    
    # =========================================================================
    # Vision Zero Enhanced Crossings
    # =========================================================================
    
    def download_vision_zero(self, output_dir: Optional[str] = None) -> Path:
        """
        Download NYC Vision Zero enhanced crossings data.
        
        This is a subset of high-visibility crossings - useful for validation.
        """
        output_dir = Path(output_dir or self.paths['raw']['vision_zero'])
        output_path = output_dir / "vision_zero_crossings.geojson"
        
        if output_path.exists():
            logger.info(f"Vision Zero data already exists at {output_path}")
            return output_path
        
        # Vision Zero enhanced crossings from NYC Open Data
        vz_url = "https://data.cityofnewyork.us/api/geospatial/bc4v-7aum?method=export&format=GeoJSON"
        
        logger.info("Downloading Vision Zero enhanced crossings...")
        
        response = requests.get(vz_url, timeout=120)
        
        if response.status_code != 200:
            logger.warning(f"Failed to download Vision Zero data: {response.status_code}")
            # Try alternative endpoint
            vz_url_alt = "https://data.cityofnewyork.us/resource/bc4v-7aum.geojson?$limit=50000"
            response = requests.get(vz_url_alt, timeout=120)
        
        if response.status_code == 200:
            # Load and filter to Brooklyn
            gdf = gpd.GeoDataFrame.from_features(response.json()['features'], crs='EPSG:4326')
            
            brooklyn_poly = self.brooklyn_bbox.to_polygon()
            gdf = gdf[gdf.within(brooklyn_poly)]
            
            gdf.to_file(output_path, driver='GeoJSON')
            
            logger.info(f"Saved {len(gdf)} Vision Zero crossings to {output_path}")
            
            return output_path
        else:
            logger.warning("Could not download Vision Zero data")
            return None
    
    # =========================================================================
    # Tile2Net Weights Download
    # =========================================================================
    
    def download_tile2net_weights(
        self, 
        output_dir: Optional[str] = None
    ) -> Path:
        """
        Download Tile2Net model weights.
        
        Note: Weights are hosted on Google Drive, may require manual download.
        """
        output_dir = Path(output_dir or "models/tile2net_weights")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "Tile2Net weights need to be downloaded from Google Drive:\n"
            "https://drive.google.com/drive/folders/1cu-MATHgekWUYqj9TFr12utl6VB-XKSu\n"
            f"Please download and place in: {output_dir}"
        )
        
        # Check if weights exist
        weight_files = list(output_dir.glob("*.pth")) + list(output_dir.glob("*.pt"))
        
        if weight_files:
            logger.info(f"Found existing weights: {[f.name for f in weight_files]}")
            return output_dir
        
        return output_dir
    
    # =========================================================================
    # Download All
    # =========================================================================
    
    def download_all(
        self,
        use_test_area: bool = True,
        skip_ortho: bool = False
    ):
        """
        Download all required datasets.
        
        Args:
            use_test_area: Use smaller test area for ortho tiles
            skip_ortho: Skip orthoimagery download (large)
        """
        logger.info("=" * 60)
        logger.info("Starting full data download for Brooklyn Crosswalk QA")
        logger.info("=" * 60)
        
        results = {}
        
        # 1. Orthoimagery
        if not skip_ortho:
            logger.info("\n[1/4] Downloading orthoimagery tiles...")
            bbox = self.test_bbox if use_test_area else self.brooklyn_bbox
            results['ortho'] = self.download_ortho_tiles(bbox=bbox)
        
        # 2. LION centerlines
        logger.info("\n[2/4] Downloading LION centerlines...")
        results['lion'] = self.download_lion()
        
        # 3. OSM crossings
        logger.info("\n[3/4] Downloading OSM crossings...")
        results['osm'] = self.download_osm_crossings()
        
        # 4. Vision Zero (optional)
        logger.info("\n[4/4] Downloading Vision Zero crossings...")
        results['vision_zero'] = self.download_vision_zero()
        
        # 5. Model weights
        logger.info("\n[5/4] Checking Tile2Net weights...")
        results['weights'] = self.download_tile2net_weights()
        
        logger.info("\n" + "=" * 60)
        logger.info("Download Summary:")
        logger.info("=" * 60)
        
        for key, value in results.items():
            if value is not None:
                if isinstance(value, list):
                    logger.info(f"  {key}: {len(value)} files")
                else:
                    logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: FAILED")
        
        return results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for data download."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download datasets for Brooklyn Crosswalk QA"
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Download all datasets'
    )
    
    parser.add_argument(
        '--ortho',
        action='store_true',
        help='Download orthoimagery tiles'
    )
    
    parser.add_argument(
        '--lion',
        action='store_true',
        help='Download LION centerlines'
    )
    
    parser.add_argument(
        '--osm',
        action='store_true',
        help='Download OSM crossings'
    )
    
    parser.add_argument(
        '--vision-zero',
        action='store_true',
        help='Download Vision Zero crossings'
    )
    
    parser.add_argument(
        '--weights',
        action='store_true',
        help='Download/check Tile2Net weights'
    )
    
    parser.add_argument(
        '--area',
        default='test',
        choices=['test', 'full'],
        help='Area to download: test (small) or full (all Brooklyn)'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DataDownloader(args.config)
    
    if args.all:
        downloader.download_all(use_test_area=(args.area == 'test'))
    else:
        if args.ortho:
            bbox = downloader.test_bbox if args.area == 'test' else downloader.brooklyn_bbox
            downloader.download_ortho_tiles(bbox=bbox)
        
        if args.lion:
            downloader.download_lion()
        
        if args.osm:
            downloader.download_osm_crossings()
        
        if args.vision_zero:
            downloader.download_vision_zero()
        
        if args.weights:
            downloader.download_tile2net_weights()
        
        # If no specific option, show help
        if not any([args.ortho, args.lion, args.osm, args.vision_zero, args.weights]):
            parser.print_help()


if __name__ == "__main__":
    main()