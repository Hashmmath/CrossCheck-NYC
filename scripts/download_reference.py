#!/usr/bin/env python3
"""
CrossCheck NYC - Reference Data Downloader
==========================================
Download reference datasets for crosswalk comparison.

Datasets:
- OpenStreetMap crossings
- NYC Vision Zero Enhanced Crossings
- NYC Raised Crosswalk Locations

Usage:
    python download_reference.py --all
    python download_reference.py --dataset osm
"""

import argparse
import logging
import requests
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

# NYC Bounding Box (covers Manhattan + Brooklyn study areas)
NYC_BBOX = box(-74.05, 40.55, -73.85, 40.85)


class ReferenceDownloader:
    """Download and prepare reference datasets."""
    
    def __init__(self):
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        
    def download_osm_crossings(self) -> Optional[gpd.GeoDataFrame]:
        """Download crossing nodes from OpenStreetMap using OSMnx."""
        try:
            import osmnx as ox
            
            logger.info("Downloading OSM crossings for NYC...")
            
            # Download for both boroughs
            areas = [
                "Manhattan, New York City, New York, USA",
                "Brooklyn, New York City, New York, USA",
                "Queens, New York City, New York, USA"
            ]
            
            all_crossings = []
            
            for area in areas:
                logger.info(f"  Fetching: {area}")
                try:
                    gdf = ox.features_from_place(area, tags={"highway": "crossing"})
                    gdf['source_area'] = area
                    all_crossings.append(gdf)
                    logger.info(f"    Found {len(gdf)} crossings")
                except Exception as e:
                    logger.warning(f"    Error: {e}")
            
            if not all_crossings:
                logger.error("No OSM crossings found")
                return None
            
            # Combine
            combined = gpd.GeoDataFrame(pd.concat(all_crossings, ignore_index=True))
            combined['source'] = 'osm'
            
            # Save
            output_path = REFERENCE_DIR / "osm_crossings_nyc.geojson"
            combined.to_file(output_path, driver='GeoJSON')
            logger.info(f"Saved {len(combined)} OSM crossings to {output_path}")
            
            return combined
            
        except ImportError:
            logger.error("osmnx not installed. Run: pip install osmnx")
            return None
        except Exception as e:
            logger.error(f"Error downloading OSM data: {e}")
            return None
    
    def download_vision_zero(self) -> Optional[gpd.GeoDataFrame]:
        """Download NYC Vision Zero Enhanced Crossings."""
        logger.info("Downloading Vision Zero Enhanced Crossings...")
        
        # NYC Open Data API endpoint
        url = "https://data.cityofnewyork.us/api/geospatial/c27w-7ciz?method=export&format=GeoJSON"
        
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Save raw
            raw_path = REFERENCE_DIR / "vision_zero_raw.geojson"
            with open(raw_path, 'w') as f:
                f.write(response.text)
            
            # Load and process
            gdf = gpd.read_file(raw_path)
            gdf['source'] = 'vision_zero'
            
            # Filter to NYC bbox
            gdf = gdf.to_crs(epsg=4326)
            gdf = gdf[gdf.geometry.intersects(NYC_BBOX)]
            
            # Save processed
            output_path = REFERENCE_DIR / "vision_zero_crossings.geojson"
            gdf.to_file(output_path, driver='GeoJSON')
            logger.info(f"Saved {len(gdf)} Vision Zero crossings to {output_path}")
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error downloading Vision Zero data: {e}")
            return None
    
    def download_raised_crosswalks(self) -> Optional[gpd.GeoDataFrame]:
        """Download NYC Raised Crosswalk Locations."""
        logger.info("Downloading Raised Crosswalk Locations...")
        
        url = "https://data.cityofnewyork.us/resource/uh2s-ftgh.geojson"
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            gdf = gpd.read_file(response.text)
            gdf['source'] = 'raised_crosswalks'
            
            output_path = REFERENCE_DIR / "raised_crosswalks.geojson"
            gdf.to_file(output_path, driver='GeoJSON')
            logger.info(f"Saved {len(gdf)} raised crosswalks to {output_path}")
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error downloading raised crosswalks: {e}")
            return None
    
    def create_combined_reference(self) -> gpd.GeoDataFrame:
        """Combine all reference datasets."""
        logger.info("Creating combined reference dataset...")
        
        all_data = []
        
        for geojson_file in REFERENCE_DIR.glob("*.geojson"):
            if "combined" in geojson_file.name or "raw" in geojson_file.name:
                continue
            try:
                gdf = gpd.read_file(geojson_file)
                gdf['source_file'] = geojson_file.name
                all_data.append(gdf)
                logger.info(f"  Added {len(gdf)} from {geojson_file.name}")
            except Exception as e:
                logger.warning(f"  Error loading {geojson_file}: {e}")
        
        if not all_data:
            logger.warning("No reference data to combine")
            return gpd.GeoDataFrame()
        
        combined = gpd.GeoDataFrame(pd.concat(all_data, ignore_index=True))
        combined = combined.set_crs(epsg=4326, allow_override=True)
        
        output_path = REFERENCE_DIR / "combined_reference.geojson"
        combined.to_file(output_path, driver='GeoJSON')
        logger.info(f"Saved combined reference ({len(combined)} records) to {output_path}")
        
        return combined
    
    def download_all(self):
        """Download all reference datasets."""
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOADING REFERENCE DATASETS")
        logger.info("=" * 60 + "\n")
        
        results = {}
        
        # OSM
        gdf = self.download_osm_crossings()
        results['osm'] = len(gdf) if gdf is not None else 0
        
        # Vision Zero
        gdf = self.download_vision_zero()
        results['vision_zero'] = len(gdf) if gdf is not None else 0
        
        # Raised Crosswalks
        gdf = self.download_raised_crosswalks()
        results['raised_crosswalks'] = len(gdf) if gdf is not None else 0
        
        # Combine
        combined = self.create_combined_reference()
        results['combined'] = len(combined)
        
        # Summary
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        for name, count in results.items():
            print(f"  {name}: {count} records")
        print(f"\nFiles saved to: {REFERENCE_DIR}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Download reference datasets")
    parser.add_argument('--all', '-a', action='store_true', help='Download all')
    parser.add_argument('--dataset', '-d', choices=['osm', 'vision_zero', 'raised_crosswalks'])
    parser.add_argument('--combine', action='store_true', help='Combine existing files')
    
    args = parser.parse_args()
    
    downloader = ReferenceDownloader()
    
    if args.all:
        downloader.download_all()
    elif args.dataset == 'osm':
        downloader.download_osm_crossings()
    elif args.dataset == 'vision_zero':
        downloader.download_vision_zero()
    elif args.dataset == 'raised_crosswalks':
        downloader.download_raised_crosswalks()
    elif args.combine:
        downloader.create_combined_reference()
    else:
        print("Usage: python download_reference.py --all")


if __name__ == "__main__":
    main()
