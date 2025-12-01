"""
Tile2Net Diagnostic Script
==========================

Run this to understand what Tile2Net can do on your machine.

Usage: python check_tile2net.py
"""

import sys
print(f"Python: {sys.version}")
print()

# Step 1: Check tile2net import
print("=" * 60)
print("STEP 1: Checking Tile2Net Installation")
print("=" * 60)

try:
    import tile2net
    print(f"✓ tile2net imported successfully")
    print(f"  Version: {getattr(tile2net, '__version__', 'unknown')}")
    print(f"  Location: {tile2net.__file__}")
except ImportError as e:
    print(f"✗ Failed to import tile2net: {e}")
    print("  Run: pip install tile2net")
    sys.exit(1)

# Step 2: Check available modules
print()
print("=" * 60)
print("STEP 2: Checking Available Modules")
print("=" * 60)

modules_to_check = [
    'tile2net.Raster',
    'tile2net.tileseg',
    'tile2net.tileseg.inference',
    'tile2net.namespace',
]

for mod_name in modules_to_check:
    try:
        parts = mod_name.split('.')
        obj = tile2net
        for part in parts[1:]:
            obj = getattr(obj, part)
        print(f"✓ {mod_name} available")
    except AttributeError:
        print(f"✗ {mod_name} NOT available")

# Step 3: Check Raster class
print()
print("=" * 60)
print("STEP 3: Checking Raster Class")
print("=" * 60)

try:
    from tile2net import Raster
    print(f"✓ Raster class imported")
    
    # Check Raster methods
    methods = ['generate', 'inference', 'stitch', 'dump']
    for method in methods:
        if hasattr(Raster, method):
            print(f"  ✓ Raster.{method}() available")
        else:
            print(f"  ✗ Raster.{method}() NOT available")
            
except Exception as e:
    print(f"✗ Could not import Raster: {e}")

# Step 4: Check available sources
print()
print("=" * 60)
print("STEP 4: Checking Available Imagery Sources")
print("=" * 60)

try:
    # Try different ways to get sources
    sources_found = False
    
    # Method 1
    try:
        from tile2net.tileseg.source import get_sources
        sources = get_sources()
        print(f"✓ Available sources (via get_sources):")
        for s in sources:
            print(f"    - {s}")
        sources_found = True
    except:
        pass
    
    # Method 2
    if not sources_found:
        try:
            from tile2net import sources
            print(f"✓ Available sources (via tile2net.sources):")
            print(f"    {sources}")
            sources_found = True
        except:
            pass
    
    # Method 3
    if not sources_found:
        try:
            from tile2net.namespace import sources
            print(f"✓ Available sources (via namespace):")
            for name, source in sources.items():
                print(f"    - {name}: {source}")
            sources_found = True
        except:
            pass
    
    # Method 4: Check source module directly
    if not sources_found:
        try:
            import tile2net.tileseg.source as source_module
            print(f"✓ Source module contents:")
            for attr in dir(source_module):
                if not attr.startswith('_'):
                    print(f"    - {attr}")
            sources_found = True
        except:
            pass
    
    if not sources_found:
        print("✗ Could not find imagery sources")
        print("  Tile2Net may need to download source configuration")
        
except Exception as e:
    print(f"✗ Error checking sources: {e}")

# Step 5: Test creating a Raster object
print()
print("=" * 60)
print("STEP 5: Testing Raster Object Creation")
print("=" * 60)

test_locations = [
    ("Manhattan Address", "Washington Square Park, Manhattan, NY, USA"),
    ("Manhattan Coords", "-74.000,40.730,-73.995,40.735"),
    ("Washington DC", "The White House, Washington, DC, USA"),
    ("Boston", "Boston Common, Boston, MA, USA"),
]

for name, location in test_locations:
    print(f"\nTrying: {name}")
    print(f"  Location: {location}")
    
    try:
        from tile2net import Raster
        
        raster = Raster(
            location=location,
            name="test_project",
            output_dir="./test_output"
        )
        
        print(f"  ✓ Raster created successfully!")
        
        # Check if source was found
        if hasattr(raster, 'source') and raster.source:
            print(f"  ✓ Source found: {raster.source}")
        else:
            print(f"  ⚠ No source found for this location")
        
        # Check grid info
        if hasattr(raster, 'grid'):
            print(f"  ✓ Grid: {raster.grid}")
        
        # Try to see project info
        if hasattr(raster, 'project'):
            print(f"  ✓ Project: {raster.project}")
            
        # This location works - we can use it!
        print(f"\n  >>> This location appears to work! <<<")
        
        # Clean up test directory
        import shutil
        if os.path.exists("./test_output"):
            shutil.rmtree("./test_output", ignore_errors=True)
            
        break  # Found a working location
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Step 6: Check model weights
print()
print("=" * 60)
print("STEP 6: Checking Model Weights")
print("=" * 60)

import os
from pathlib import Path

# Common locations for tile2net weights
possible_weight_locations = [
    Path.home() / ".tile2net",
    Path.home() / ".cache" / "tile2net",
    Path("./weights"),
    Path("./models"),
]

weights_found = False
for loc in possible_weight_locations:
    if loc.exists():
        print(f"✓ Found tile2net directory: {loc}")
        # List contents
        for f in loc.glob("**/*"):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    - {f.name} ({size_mb:.1f} MB)")
                if size_mb > 100:  # Model weights are usually >100MB
                    weights_found = True
                    print(f"      ^ This looks like model weights!")

if not weights_found:
    print("⚠ No model weights found")
    print("  Tile2Net should download weights automatically on first run")
    print("  This requires internet connection")

# Summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
To run Tile2Net successfully, you need:
1. ✓ tile2net package installed
2. A supported location (Manhattan, DC, Boston, Cambridge)
3. Internet connection to download tiles
4. Model weights (downloaded automatically)

Next steps:
1. Run this script and share the output
2. If a location works, use that for your project
3. If no location works, we may need to provide custom imagery
""")

# Cleanup
import shutil
if os.path.exists("./test_output"):
    shutil.rmtree("./test_output", ignore_errors=True)