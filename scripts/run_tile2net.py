#!/usr/bin/env python3
"""
CrossCheck NYC - Tile2Net Runner
================================
Run tile2net inference on the 4 study locations.

Usage:
    python run_tile2net.py --location financial_district
    python run_tile2net.py --all
    python run_tile2net.py --list
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "locations.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"


def load_locations():
    """Load location configurations from JSON."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return {loc['id']: loc for loc in config['locations']}


def run_tile2net(location_id: str, location_data: dict, output_dir: Path) -> dict:
    """
    Run tile2net generate | inference pipeline for a location.
    
    Args:
        location_id: Location identifier
        location_data: Location configuration dict
        output_dir: Output directory path
        
    Returns:
        Dict with run results
    """
    result = {
        "location_id": location_id,
        "location": location_data["location"],
        "status": "pending",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "error": None
    }
    
    logger.info("=" * 60)
    logger.info(f"Processing: {location_data['display_name']}")
    logger.info(f"Location: {location_data['location']}")
    logger.info(f"Category: {location_data['category']}")
    logger.info("=" * 60)
    
    # Build command
    cmd = (
        f'python -m tile2net generate '
        f'-l "{location_data["location"]}" '
        f'-n {location_id} '
        f'-o {output_dir} '
        f'| python -m tile2net inference --dump_percent 100'
    )
    
    logger.info(f"Command: {cmd}")
    
    try:
        # Run the pipeline
        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )
        
        if process.returncode == 0:
            result["status"] = "success"
            logger.info(f"✓ Successfully completed: {location_id}")
            
            # Log output paths
            loc_output = output_dir / location_id
            if loc_output.exists():
                logger.info(f"  Output directory: {loc_output}")
                crosswalk_path = loc_output / "polygons" / "crosswalk.geojson"
                if crosswalk_path.exists():
                    logger.info(f"  Crosswalk file: {crosswalk_path}")
        else:
            result["status"] = "failed"
            result["error"] = process.stderr[:500]  # First 500 chars of error
            logger.error(f"✗ Failed: {location_id}")
            logger.error(f"  Error: {process.stderr[:200]}")
            
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Process timed out after 30 minutes"
        logger.error(f"✗ Timeout: {location_id}")
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"✗ Exception: {location_id} - {e}")
    
    result["end_time"] = datetime.now().isoformat()
    return result


def run_all_locations():
    """Run tile2net on all 4 locations."""
    locations = load_locations()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    logger.info("\n" + "=" * 60)
    logger.info("CROSSCHECK NYC - BATCH INFERENCE")
    logger.info(f"Running {len(locations)} locations")
    logger.info("=" * 60 + "\n")
    
    for loc_id, loc_data in locations.items():
        result = run_tile2net(loc_id, loc_data, OUTPUT_DIR)
        results.append(result)
        print()  # Blank line between locations
    
    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - success
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Succeeded: {success}/{len(results)}")
    logger.info(f"  Failed: {failed}/{len(results)}")
    
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        logger.info(f"  {status_icon} {r['location_id']}: {r['status']}")
    
    # Save results
    results_file = PROJECT_ROOT / "data" / "results" / "inference_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    return results


def list_locations():
    """Print available locations."""
    locations = load_locations()
    
    print("\n" + "=" * 60)
    print("CROSSCHECK NYC - STUDY LOCATIONS")
    print("=" * 60)
    
    for loc_id, loc_data in locations.items():
        print(f"\n  {loc_id}:")
        print(f"    Display: {loc_data['display_name']}")
        print(f"    Borough: {loc_data['borough']}")
        print(f"    Category: {loc_data['category']}")
        print(f"    Location: {loc_data['location']}")
    
    print("\n" + "-" * 60)
    print("Usage:")
    print("  python run_tile2net.py --location <location_id>")
    print("  python run_tile2net.py --all")
    print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="CrossCheck NYC - Run tile2net inference"
    )
    parser.add_argument(
        '--location', '-l',
        type=str,
        help='Run on specific location (e.g., financial_district)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run on all 4 locations'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available locations'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=str(OUTPUT_DIR),
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_locations()
        return
    
    if args.all:
        run_all_locations()
        return
    
    if args.location:
        locations = load_locations()
        if args.location not in locations:
            logger.error(f"Unknown location: {args.location}")
            list_locations()
            sys.exit(1)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_tile2net(args.location, locations[args.location], output_dir)
        return
    
    # No arguments - show help
    list_locations()
    print("Please specify --location <id> or --all\n")


if __name__ == "__main__":
    main()
