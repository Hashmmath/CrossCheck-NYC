"""
Fix ALL CUDA Calls in Tile2Net
==============================

This script finds and patches ALL .cuda() calls in tile2net
to support CPU-only systems.

Usage: python fix_all_cuda.py
"""

import sys
from pathlib import Path
import re
import shutil

def find_tile2net_path():
    """Find where tile2net is installed."""
    try:
        import tile2net
        return Path(tile2net.__file__).parent
    except ImportError:
        print("ERROR: tile2net not installed")
        return None

def backup_file(file_path):
    """Create backup if not exists."""
    backup_path = file_path.with_suffix('.py.original')
    if not backup_path.exists():
        shutil.copy(file_path, backup_path)
        return True
    return False

def patch_cuda_calls(file_path):
    """Patch all .cuda() calls in a file to support CPU."""
    
    content = file_path.read_text(encoding='utf-8')
    original_content = content
    
    # Skip if already patched
    if "# CUDA_CPU_PATCHED" in content:
        return False, "already patched"
    
    changes = []
    
    # Pattern 1: variable.cuda() -> variable.to(device)
    # Example: inputs = {k: v.cuda() for k, v in inputs.items()}
    pattern1 = r'(\w+)\.cuda\(\)'
    
    # Pattern 2: net = net.cuda() -> handled separately
    # Pattern 3: tensor.cuda() in various contexts
    
    # First, add device detection at the top of the file (after imports)
    # Find a good place to insert device detection
    
    # Check if torch is imported
    has_torch_import = 'import torch' in content or 'from torch' in content
    
    # Find all .cuda() calls
    cuda_calls = re.findall(r'\.cuda\(\)', content)
    
    if not cuda_calls:
        return False, "no cuda calls found"
    
    # Replace .cuda() with .to(_device)
    # But first we need to define _device
    
    # Add device definition after imports
    device_code = '''
# CUDA_CPU_PATCHED: Auto-detect device
import torch as _torch_device_check
_device = _torch_device_check.device('cuda' if _torch_device_check.cuda.is_available() else 'cpu')
'''
    
    # Find the end of imports section
    lines = content.split('\n')
    insert_line = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip empty lines, comments, and imports at the start
        if stripped.startswith('from __future__'):
            insert_line = i + 1
        elif stripped.startswith('import ') or stripped.startswith('from '):
            insert_line = i + 1
        elif stripped and not stripped.startswith('#') and insert_line > 0:
            # Found first non-import line
            break
    
    # Insert device detection
    lines.insert(insert_line, device_code)
    content = '\n'.join(lines)
    changes.append("added device detection")
    
    # Now replace all .cuda() calls
    # Pattern: something.cuda() -> something.to(_device)
    new_content = re.sub(r'\.cuda\(\)', '.to(_device)', content)
    
    if new_content != content:
        content = new_content
        changes.append(f"replaced {len(cuda_calls)} .cuda() calls with .to(_device)")
    
    # Write back
    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        return True, ", ".join(changes)
    
    return False, "no changes needed"

def find_all_python_files(tile2net_path):
    """Find all Python files in tile2net."""
    return list(tile2net_path.glob("**/*.py"))

def main():
    print("=" * 60)
    print("FIX ALL CUDA CALLS IN TILE2NET")
    print("=" * 60)
    
    tile2net_path = find_tile2net_path()
    
    if not tile2net_path:
        return False
    
    print("")
    print("Tile2Net location: " + str(tile2net_path))
    
    # Find all Python files
    python_files = find_all_python_files(tile2net_path)
    print(f"Found {len(python_files)} Python files")
    
    # First, find which files have .cuda() calls
    print("")
    print("=" * 60)
    print("SCANNING FOR .cuda() CALLS")
    print("=" * 60)
    
    files_with_cuda = []
    for py_file in python_files:
        try:
            content = py_file.read_text(encoding='utf-8')
            cuda_count = len(re.findall(r'\.cuda\(\)', content))
            if cuda_count > 0:
                files_with_cuda.append((py_file, cuda_count))
                print(f"  {py_file.relative_to(tile2net_path)}: {cuda_count} calls")
        except Exception as e:
            pass
    
    print(f"\nTotal files with .cuda(): {len(files_with_cuda)}")
    
    # Backup and patch each file
    print("")
    print("=" * 60)
    print("PATCHING FILES")
    print("=" * 60)
    
    patched = 0
    failed = 0
    
    for py_file, count in files_with_cuda:
        try:
            # Backup
            backup_file(py_file)
            
            # Patch
            changed, reason = patch_cuda_calls(py_file)
            
            rel_path = py_file.relative_to(tile2net_path)
            if changed:
                print(f"[OK] {rel_path}: {reason}")
                patched += 1
            else:
                print(f"[SKIP] {rel_path}: {reason}")
                
        except Exception as e:
            print(f"[ERROR] {py_file.name}: {str(e)}")
            failed += 1
    
    # Also fix num_workers in inference.py
    print("")
    print("=" * 60)
    print("FIXING DATALOADER (num_workers=0)")
    print("=" * 60)
    
    inference_path = tile2net_path / "tileseg" / "inference" / "inference.py"
    if inference_path.exists():
        content = inference_path.read_text(encoding='utf-8')
        new_content = re.sub(r'num_workers\s*=\s*\d+', 'num_workers=0', content)
        if new_content != content:
            inference_path.write_text(new_content, encoding='utf-8')
            print("[OK] Set num_workers=0 in inference.py")
        else:
            print("[OK] num_workers already set to 0")
    
    # Summary
    print("")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Files patched: {patched}")
    print(f"  Files failed: {failed}")
    
    if patched > 0 and failed == 0:
        print("")
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print("")
        print("Now run:")
        print("  python run_tile2net_windows.py")
        return True
    else:
        print("")
        print("[WARNING] Some issues occurred. Check errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)