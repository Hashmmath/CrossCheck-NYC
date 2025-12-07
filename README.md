# CrossCheck NYC
## A Visual Analytics Tool for Assessing the Trustworthiness of AI-Mapped Crosswalk Data

## Quick Start
### Download the .ipynb file and execute each cell one by one, but before that download the the files as "crosscheck_nyc.zip" file and add it to the files section of the Google Colab, and remember you have to execute these commands in "GPU", the session restarts even though don't woory the files won't be lost and once you have executed all the codes you will get a data folder with outputs as a "all_outputs.zip" file download it and add it to your code folder and then execute "streamlit run app.py" in a local conda environment, but before activate an environment and execute "pip install -r requirements.txt"
### Step 1: Environment Setup

```bash
!pip install -q condacolab

import condacolab

condacolab.install()

!mamba create -n crosscheck python=3.11 -y

# Install tile2net
pip install -e .

# Install additional dependencies
cd ../crosscheck_nyc
pip install -r requirements.txt
```

### Step 2: Run Tile2Net Inference

**Option A: Command Line (One Location)**
```bash
python -m tile2net generate \
    -l "Financial District, Manhattan, NY, USA" \
    -n financial_district \
    -o ./data/outputs \
    | python -m tile2net inference --dump_percent 100
```

**Option B: Python Script (All Locations)**
```bash
python scripts/run_tile2net.py --all
```

**Option C: Single Location via Script**
```bash
python scripts/run_tile2net.py --location financial_district
```

### Step 3: Download Reference Data
```bash
python scripts/download_reference.py --all
```

### Step 4: Run Analysis
```bash
# Stage A: Pixel-level analysis
python analysis/stage_a_pixel_analysis.py --all

# Stage B: Network & comparison analysis  
python analysis/stage_b_network_analysis.py --all
```

### Step 5: Interactive Analysis
```bash
jupyter notebook notebooks/crosscheck_analysis.ipynb
```

---

## Tile2Net Commands Reference

### Basic Command Format
```bash
python -m tile2net generate -l "<location>" -n <name> -o <output_dir> | python -m tile2net inference --dump_percent 100
```

### Your 4 Locations

```bash
# 1. Financial District (Dense Urban)
python -m tile2net generate \
    -l "Financial District, Manhattan, NY, USA" \
    -n financial_district \
    -o ./data/outputs \
    | python -m tile2net inference --dump_percent 100

# 2. Central Park South (Open Area)
python -m tile2net generate \
    -l "Central Park South, Manhattan, NY, USA" \
    -n central_park_south \
    -o ./data/outputs \
    | python -m tile2net inference --dump_percent 100

# 3. Bay Ridge, Brooklyn (Residential)
python -m tile2net generate \
    -l "Bay Ridge, Brooklyn, NY, USA" \
    -n bay_ridge \
    -o ./data/outputs \
    | python -m tile2net inference --dump_percent 100

# 4. Downtown Brooklyn (Urban Commercial)
python -m tile2net generate \
    -l "Downtown Brooklyn, Brooklyn, NY, USA" \
    -n downtown_brooklyn \
    -o ./data/outputs \
    | python -m tile2net inference --dump_percent 100
```

---

## Key Output Files

| File | Description |
|------|-------------|
| `polygons/crosswalk.geojson` | **Primary output** - Detected crosswalk polygons |
| `polygons/sidewalk.geojson` | Detected sidewalk polygons |
| `polygons/road.geojson` | Detected road polygons |
| `segmentation/seg_results/*.png` | Segmentation images (for visual QA) |
| `network/ntw.geojson` | Centerline network |
| `*_info.json` | Project metadata |

### Segmentation Class Labels
- **0**: Background (Black)
- **1**: Sidewalk (Blue)
- **2**: Crosswalk (Red) ← **FOCUS**
- **3**: Road (Green)

---

## Requirements

- Python 3.11
- CUDA-enabled GPU (for inference)
- ~5-15 minutes per location (depending on area size)

---

## Notes

- NYC and Brooklyn are **supported regions** in tile2net (tiles download automatically)
- Always use `--dump_percent 100` to save segmentation images for visual QA
- Reference data comparison requires downloading OSM/Vision Zero data first
