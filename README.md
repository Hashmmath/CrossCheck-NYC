# CrossCheck NYC
## A Visual Analytics Tool for Assessing the Trustworthiness of AI-Mapped Crosswalk Data

---

## Project Overview

This project builds a **two-stage visual QA system** for AI-derived crosswalk maps using the **tile2net** model.

### Stage A — Segmentation Detective
- Error overlays (FP/FN visualization)
- Pixel-level confidence analysis
- Morphological operation effects

### Stage B — Network Quality Inspector  
- Topology analysis (connectivity, components)
- Comparison with reference data (OSM, Vision Zero)
- Spatial context validation

---

## Research Questions

1. **Perception & Patterns**: When does the model detect crosswalks correctly?
2. **Confidence & Choice**: How do threshold changes affect predictions?
3. **Placement & Plausibility**: Are crosswalks in logical locations?
4. **Agreement & Disagreement**: How well do results match reference data?
5. **From Evidence to Trust**: What visual cues help users trust predictions?
6. **Usability**: Which cues help decision-making?
7. **Edge Cases**: What confuses the model?

---

## Study Locations (4 Sites)

| Location | Borough | Type | Why Selected |
|----------|---------|------|--------------|
| **Financial District** | Manhattan | Dense Urban | Narrow streets, irregular grid, shadows from tall buildings |
| **Central Park South** | Manhattan | Open Area | Wide avenues, park edges, clear visibility, regular grid |
| **Bay Ridge** | Brooklyn | Residential | Suburban-style, lower buildings, tests generalization |
| **Downtown Brooklyn** | Brooklyn | Urban Commercial | Transit hub, wide intersections, Brooklyn's urban core |

### Location Comparison Strategy
- **Financial District vs Central Park South**: Dense vs Open (Manhattan)
- **Financial District vs Downtown Brooklyn**: Manhattan vs Brooklyn dense cores
- **Bay Ridge vs Downtown Brooklyn**: Residential vs Commercial (Brooklyn)
- **Central Park South vs Bay Ridge**: Manhattan open vs Brooklyn residential

---

## Folder Structure

```
crosscheck_nyc/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── config/
│   └── locations.json                  # Location configurations
│
├── scripts/
│   ├── run_tile2net.py                # Main inference runner
│   ├── extract_crosswalks.py          # Extract crosswalk data
│   ├── download_reference.py          # Download OSM/Vision Zero data
│   └── compare_with_reference.py      # Compare predictions vs reference
│
├── analysis/
│   ├── stage_a_pixel_analysis.py      # Stage A: Pixel-level QA
│   └── stage_b_network_analysis.py    # Stage B: Network & comparison
│
├── notebooks/
│   └── crosscheck_analysis.ipynb      # Interactive Jupyter analysis
│
├── data/
│   ├── outputs/                        # Tile2net outputs (per location)
│   │   ├── financial_district/
│   │   │   ├── tiles/
│   │   │   │   └── static/
│   │   │   │       └── 256_19/        # Downloaded tiles
│   │   │   ├── stitched/              # Stitched larger tiles
│   │   │   ├── segmentation/
│   │   │   │   ├── inference/         # Raw model outputs
│   │   │   │   └── seg_results/       # Segmentation images (dump_percent=100)
│   │   │   ├── polygons/
│   │   │   │   ├── crosswalk.geojson  # ← PRIMARY OUTPUT
│   │   │   │   ├── sidewalk.geojson
│   │   │   │   ├── road.geojson
│   │   │   │   └── footpath.geojson
│   │   │   ├── network/
│   │   │   │   └── ntw.geojson        # Centerline network
│   │   │   └── financial_district_info.json
│   │   │
│   │   ├── central_park_south/        # Same structure
│   │   ├── bay_ridge/                 # Same structure
│   │   └── downtown_brooklyn/         # Same structure
│   │
│   ├── reference/                      # Reference datasets
│   │   ├── osm_crossings_nyc.geojson
│   │   ├── vision_zero_crossings.geojson
│   │   ├── raised_crosswalks.geojson
│   │   └── combined_reference.geojson
│   │
│   └── results/                        # Analysis outputs
│       ├── stage_a/
│       │   ├── financial_district_pixel_analysis.json
│       │   ├── central_park_south_pixel_analysis.json
│       │   ├── bay_ridge_pixel_analysis.json
│       │   └── downtown_brooklyn_pixel_analysis.json
│       ├── stage_b/
│       │   ├── financial_district_network_analysis.json
│       │   ├── comparison_with_osm.json
│       │   └── cross_location_comparison.json
│       └── figures/
│           ├── segmentation_samples/
│           ├── error_overlays/
│           └── comparison_maps/
│
└── docs/
    └── methodology.md                  # Analysis methodology notes
```

---

## Quick Start

### Step 1: Environment Setup

```bash
# Clone tile2net
git clone https://github.com/VIDA-NYU/tile2net.git
cd tile2net

# Create conda environment
conda create --name crosscheck python=3.11
conda activate crosscheck

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
