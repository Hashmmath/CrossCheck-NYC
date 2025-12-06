#!/bin/bash
# =============================================================================
# CrossCheck NYC - Environment Setup Script
# =============================================================================
# This script sets up everything you need:
#   1. Creates conda environment
#   2. Installs PyTorch with CUDA
#   3. Clones and installs tile2net
#   4. Installs all project requirements
#
# Usage:
#   cd crosscheck_nyc
#   bash setup_environment.sh
#   conda activate crosscheck
# =============================================================================

set -e

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "      CrossCheck NYC - Environment Setup"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Configuration
ENV_NAME="crosscheck"
PYTHON_VERSION="3.11"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Step 0: Check prerequisites
# =============================================================================
print_step "Checking prerequisites..."

if ! command -v conda &> /dev/null; then
    print_error "conda not found!"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
print_success "conda found"

if ! command -v git &> /dev/null; then
    print_error "git not found!"
    echo "Please install git first"
    exit 1
fi
print_success "git found"

# =============================================================================
# Step 1: Create conda environment
# =============================================================================
print_step "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"

if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        conda env remove -n $ENV_NAME -y
        conda create -y -n $ENV_NAME python=$PYTHON_VERSION
    fi
else
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

# =============================================================================
# Step 2: Activate environment
# =============================================================================
print_step "Activating environment..."

# Initialize conda for this shell
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

print_success "Environment activated: $ENV_NAME"

# =============================================================================
# Step 3: Install PyTorch with CUDA
# =============================================================================
print_step "Installing PyTorch with CUDA support..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

print_success "PyTorch installed"

# =============================================================================
# Step 4: Clone and install tile2net
# =============================================================================
print_step "Setting up tile2net..."

TILE2NET_DIR="./tile2net"

if [ -d "$TILE2NET_DIR" ]; then
    print_warning "tile2net directory already exists. Updating..."
    cd $TILE2NET_DIR
    git pull || print_warning "Could not pull updates"
    cd ..
else
    print_step "Cloning tile2net repository..."
    git clone https://github.com/VIDA-NYU/tile2net.git
fi

print_step "Installing tile2net..."
cd $TILE2NET_DIR
pip install -e .
cd ..

print_success "tile2net installed"

# =============================================================================
# Step 5: Install project requirements
# =============================================================================
print_step "Installing CrossCheck NYC requirements..."

pip install -r requirements.txt

print_success "All requirements installed"

# =============================================================================
# Step 6: Verify installation
# =============================================================================
print_step "Verifying installation..."

echo ""
python << 'EOF'
import sys
print(f"Python version: {sys.version}")
print()

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch: NOT INSTALLED - {e}")

# Check tile2net
try:
    import tile2net
    print(f"✓ tile2net: Installed")
except ImportError as e:
    print(f"✗ tile2net: NOT INSTALLED - {e}")

# Check geopandas
try:
    import geopandas
    print(f"✓ GeoPandas: {geopandas.__version__}")
except ImportError as e:
    print(f"✗ GeoPandas: NOT INSTALLED - {e}")

# Check other key packages
packages = ['shapely', 'rasterio', 'opencv-python', 'matplotlib', 'numpy', 'pandas']
for pkg in ['shapely', 'rasterio', 'cv2', 'matplotlib', 'numpy', 'pandas']:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'installed')
        print(f"✓ {pkg}: {ver}")
    except ImportError:
        print(f"✗ {pkg}: NOT INSTALLED")

print()
print("="*50)
print("Setup verification complete!")
print("="*50)
EOF

# =============================================================================
# Step 7: Create directory structure
# =============================================================================
print_step "Creating project directories..."

mkdir -p data/outputs/financial_district
mkdir -p data/outputs/central_park_south
mkdir -p data/outputs/bay_ridge
mkdir -p data/outputs/downtown_brooklyn
mkdir -p data/reference
mkdir -p data/results/stage_a
mkdir -p data/results/stage_b
mkdir -p data/results/figures/segmentation_samples
mkdir -p data/results/figures/comparison_maps

print_success "Directories created"

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "      Setup Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "IMPORTANT: You need to activate the environment in your terminal:"
echo ""
echo "    conda activate $ENV_NAME"
echo ""
echo "Then run tile2net on your locations:"
echo ""
echo "    python -m tile2net generate -l \"Financial District, Manhattan, NY, USA\" -n financial_district -o ./data/outputs | python -m tile2net inference --dump_percent 100"
echo ""
echo "Or use the batch script:"
echo ""
echo "    python scripts/run_tile2net.py --all"
echo ""
echo "═══════════════════════════════════════════════════════════"
