"""
Windows-Compatible Tile2Net Runner
"""

import os
import sys

# CRITICAL: Set these BEFORE importing torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Fix Windows multiprocessing
if sys.platform == "win32":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

import torch

# Force CPU if no CUDA
if not torch.cuda.is_available():
    print("INFO: CUDA not available, using CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set number of threads for CPU
torch.set_num_threads(4)

# Now import tile2net
from tile2net import Raster
from pathlib import Path
import shutil

def main():
    print("=" * 60)
    print("TILE2NET: WINDOWS CPU INFERENCE")
    print("=" * 60)
    
    # Configuration
    OUTPUT_DIR = Path("./tile2net_output")
    LOCATION = "Washington Square Park, Manhattan, NY, USA"
    PROJECT_NAME = "manhattan_crosswalks"
    
    # Clean previous output
    if OUTPUT_DIR.exists():
        print("Cleaning previous output...")
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Location: " + LOCATION)
    print("Output: " + str(OUTPUT_DIR))
    
    # Create Raster
    print("")
    print("Creating Raster object...")
    raster = Raster(
        location=LOCATION,
        name=PROJECT_NAME,
        output_dir=str(OUTPUT_DIR)
    )
    
    print("Source: " + str(raster.source if hasattr(raster, "source") else "N/A"))
    
    # Generate tiles
    print("")
    print("=" * 60)
    print("STEP 1: Generating/Downloading tiles...")
    print("=" * 60)
    raster.generate(step=1)
    print("[OK] Tiles generated")
    
    # Stitch tiles
    print("")
    print("=" * 60)
    print("STEP 2: Stitching tiles...")
    print("=" * 60)
    raster.stitch(step=1)
    print("[OK] Tiles stitched")
    
    # Run inference
    print("")
    print("=" * 60)
    print("STEP 3: Running inference...")
    print("=" * 60)
    print("This will take 30-60 minutes on CPU. Please wait...")
    raster.inference()
    print("[OK] Inference complete!")
    
    # Check results
    print("")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    geojson_files = list(OUTPUT_DIR.glob("**/*.geojson"))
    png_files = list(OUTPUT_DIR.glob("**/*.png"))
    
    print("GeoJSON files: " + str(len(geojson_files)))
    print("PNG files: " + str(len(png_files)))
    
    if geojson_files:
        print("")
        print("Detected feature files:")
        for f in geojson_files[:5]:
            print("  - " + str(f))
    
    print("")
    print("=" * 60)
    print("SUCCESS! Tile2Net completed.")
    print("=" * 60)
    print("")
    print("Next steps:")
    print("1. Run: python process_tile2net_results.py")
    print("2. Then: streamlit run app.py")

if __name__ == "__main__":
    main()
