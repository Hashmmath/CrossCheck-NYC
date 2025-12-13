# 🚶 CrossCheck-NYC

**A Visual Analytics Tool for Assessing the Trustworthiness of AI-Mapped Crosswalk Data**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/](https://colab.research.google.com/drive/18aVD8YVABGXXtgqshS3staPWUSGc3xOx?usp=sharing))

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
  - [Phase 1: Google Colab (Inference & Analysis)](#phase-1-google-colab-inference--analysis)
  - [Phase 2: Local Machine (Dashboard)](#phase-2-local-machine-dashboard)
- [Research Questions](#-research-questions)
- [Team](#-team)
- [References](#-references)

---

## 🎯 Overview

CrossCheck-NYC is a two-stage visual analytics system for evaluating AI-derived crosswalk maps from [Tile2Net](https://github.com/VIDA-NYU/tile2net), a semantic segmentation model developed by NYU VIDA Lab.

### The Problem
- **7,148 pedestrians** killed in the US in 2024 (48% increase since 2014)
- Cities lack comprehensive crosswalk data for safety planning
- AI models can help, but need quality assessment tools

### Our Solution
- **Stage A (Segmentation Detective)**: Pixel-level analysis of model predictions
- **Stage B (Network Inspector)**: Geographic comparison with OpenStreetMap ground truth

---

## 📁 Project Structure

```
crosscheck_nyc/
│
├── app.py                          # 🖥️ Streamlit dashboard (run locally)
├── requirements.txt                # Python dependencies for dashboard
├── README.md                       # This file
│
├── notebooks/
│   └── CrossCheck_NYC.ipynb        # 📓 Colab notebook (inference + analysis)
│
├── analysis/                       # Analysis scripts (run on Colab)
│   ├── complete_stage_analysis.py  # Stage A + B metrics computation
│   ├── feature_impact_analysis.py  # Environmental factor analysis
│   └── tsne_feature_analysis.py    # t-SNE dimensionality reduction
│
├── scripts/                        # Utility scripts (run on Colab)
│   ├── download_reference.py       # Download OSM ground truth
│   └── extract_osm_features.py     # Extract OSM features
│
└── data/                           # Generated data (from Colab → Local)
    ├── outputs/                    # Tile2Net inference results
    ├── reference/                  # OSM ground truth
    ├── features/                   # Extracted OSM features
    └── results/                    # Analysis metrics & visualizations
```

---

## 🚀 Setup & Installation

This project uses a **two-phase workflow**:

| Phase | Platform | Purpose | GPU Required |
|-------|----------|---------|--------------|
| **Phase 1** | Google Colab | Run Tile2Net inference + analysis scripts | ✅ Yes (T4 GPU) |
| **Phase 2** | Local Machine | Run Streamlit dashboard | ❌ No |

---

### Phase 1: Google Colab (Inference & Analysis)

> ⚠️ **Important**: This phase requires a GPU. Use Google Colab with T4 GPU (free tier available).

#### Step 1: Open the Notebook in Colab

1. Upload `CrossCheck_NYC.ipynb` to Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU**

#### Step 2: Run All Cells in Order

The notebook executes the following steps:

```python
# 1. Install Conda in Colab
!pip install -q condacolab
import condacolab
condacolab.install()

# 2. Create conda environment
!mamba create -n crosscheck python=3.11 -y

# 3. Install Tile2Net and dependencies
%cd /content
!git clone https://github.com/VIDA-NYU/tile2net.git
%cd tile2net
!pip install -e .
!pip install geopandas osmnx folium matplotlib seaborn

# 4. Run inference for each location (5 NYC locations)
!python -m tile2net generate -l "Financial District, Manhattan, NY, USA" -n financial_district -o ./outputs | python -m tile2net inference --dump_percent 100
!python -m tile2net generate -l "East Village, Manhattan, NY, USA" -n east_village -o ./outputs | python -m tile2net inference --dump_percent 100
!python -m tile2net generate -l "Bay Ridge, Brooklyn, NY, USA" -n bay_ridge -o ./outputs | python -m tile2net inference --dump_percent 100
!python -m tile2net generate -l "Downtown Brooklyn, NY, USA" -n downtown_brooklyn -o ./outputs | python -m tile2net inference --dump_percent 100
!python -m tile2net generate -l "Kew Gardens, Queens, NY, USA" -n kew_gardens -o ./outputs | python -m tile2net inference --dump_percent 100

# 5. Organize outputs into data directory
!mkdir -p data/outputs data/reference data/results/metrics
!mv outputs/* data/outputs/

# 6. Run analysis scripts
!python analysis/complete_stage_analysis.py
!python analysis/feature_impact_analysis.py
!python analysis/tsne_feature_analysis.py

# 7. Zip all outputs for download
!zip -r all_outputs.zip ./data/outputs ./data/results ./data/reference ./data/features
```

#### Step 3: Download the Results

After all cells complete, download `all_outputs.zip` from Colab:
- Click the **Files** icon (📁) in the left sidebar
- Right-click `all_outputs.zip` → **Download**

---

### Phase 2: Local Machine (Dashboard)

> 💻 Run this on your local machine (Windows/Mac/Linux). No GPU needed.

#### Step 1: Clone/Download the Project

```bash
# Option A: Clone from GitHub
git clone https://github.com/yourusername/crosscheck-nyc.git
cd crosscheck-nyc

# Option B: Extract from zip
unzip crosscheck_nyc.zip
cd crosscheck_nyc
```

#### Step 2: Create Conda Environment

```bash
# Create new environment with Python 3.11
conda create -n crosscheck python=3.11 -y

# Activate the environment
conda activate crosscheck
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Extract Data from Colab

Copy the `all_outputs.zip` file (downloaded from Colab) into the project folder, then:

```bash
# Extract the data
unzip all_outputs.zip
```

Your folder structure should now look like:
```
crosscheck_nyc/
├── app.py
├── data/
│   ├── outputs/
│   │   ├── financial_district/
│   │   ├── east_village/
│   │   ├── bay_ridge/
│   │   ├── downtown_brooklyn/
│   │   └── kew_gardens/
│   ├── reference/
│   ├── features/
│   └── results/
│       └── metrics/
│           ├── results.json
│           ├── feature_impact_metrics.json
│           └── tsne_embeddings.json
└── ...
```

#### Step 5: Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically at **http://localhost:8501**

---

## 📦 Requirements (Local Machine)

### requirements.txt
```txt
# Core Framework
streamlit>=1.28.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0
altair>=5.0.0
pillow>=9.5.0

# Interactive Maps
folium>=0.14.0
streamlit-folium>=0.15.0

# Geospatial
geopandas>=0.13.0
shapely>=2.0.0

# Machine Learning
scikit-learn>=1.2.0
```

### Quick Install
```bash
pip install streamlit numpy pandas scipy matplotlib altair pillow folium streamlit-folium geopandas shapely scikit-learn
```

---

## 🔬 Research Questions

| RQ | Question | Stage |
|----|----------|-------|
| **RQ1** | When does the model detect crosswalks correctly? | A |
| **RQ2** | How do threshold changes affect detection? | A |
| **RQ3** | Are detections in logical places (near sidewalks/roads)? | B |
| **RQ4** | How well do results match OSM ground truth? | B |
| **RQ5** | What visual cues help users trust the model? | A |
| **RQ6** | Which cues help users question outputs? | A, B |
| **RQ7** | What edge cases confuse the model? | A |

---

## 🗺️ Locations Analyzed

| Location | Borough | Characteristics |
|----------|---------|-----------------|
| Financial District | Manhattan | Dense high-rises, heavy shadows |
| East Village | Manhattan | Mixed residential, tree-lined streets |
| Bay Ridge | Brooklyn | Suburban feel, residential roads |
| Downtown Brooklyn | Brooklyn | Transit hub, wide roads |
| Kew Gardens | Queens | Residential, different urban fabric |

---

## 🛠️ Troubleshooting

### Colab Issues

**"Runtime disconnected"**
- Colab has usage limits. Wait and retry, or use Colab Pro.

**"CUDA out of memory"**
- Restart runtime and run cells again. The T4 GPU should handle this.

### Local Issues

**"No data found"**
```bash
# Make sure data is extracted
ls ./data/outputs/
# Should show: financial_district, east_village, etc.
```

**"ModuleNotFoundError"**
```bash
# Make sure environment is activated
conda activate crosscheck
pip install -r requirements.txt
```

**"Streamlit not opening"**
```bash
# Try specifying port
streamlit run app.py --server.port 8502
# Then open http://localhost:8502
```

---

## 👥 Team

**ZebraSense Lads** - NYU CS-GY 9223 Visual Analytics, Fall 2025
**Hashmmath Shaik - hs5544**
**Gaurav Wadwa - gw2467**
**Naman Vshishta - nv2375**
---

## 📚 References

### Core Model
- Hosseini, M., et al. (2023). *Mapping the walk: A scalable computer vision approach for generating sidewalk network datasets from aerial imagery.* Computers, Environment and Urban Systems. [GitHub](https://github.com/VIDA-NYU/tile2net)

### Data Sources
- NYC Orthoimagery: [gis.ny.gov](https://gis.ny.gov/new-york-city-orthoimagery-downloads)
- OpenStreetMap: [geofabrik.de](https://download.geofabrik.de/north-america/us/new-york.html)

### Statistics
- GHSA (2024). *Pedestrian Traffic Fatalities by State*
- NHTSA (2024). *Pedestrians: 2023 Data*

---

## 📄 License

MIT License

---

<p align="center">
  <b>CrossCheck-NYC</b> • Visual Analytics for Pedestrian Safety<br>
  Made with ❤️ by ZebraSense Lads
</p>
