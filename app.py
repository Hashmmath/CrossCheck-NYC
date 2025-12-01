"""
CrossCheck-NYC: Interactive Crosswalk Detection QA Application
==============================================================

Final fixed version with:
- Working Tab A navigation
- Fixed Tab B map (no page reload issues)
- Fixed Stage A Metrics chart
- Professional styling (no animations)

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="CrossCheck-NYC",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports
try:
    import altair as alt
    import matplotlib.pyplot as plt
    from PIL import Image
    import geopandas as gpd
    from shapely.geometry import Point
    import rasterio
    from skimage import morphology
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.stop()

# Optional: Folium
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

class AppConfig:
    BASE_DIR = Path(".")
    DATA_DIR = BASE_DIR / "data"
    TILES_DIR = DATA_DIR / "raw" / "ortho"
    GT_DIR = DATA_DIR / "processed" / "ground_truth"
    PROB_DIR = DATA_DIR / "processed" / "predictions" / "probabilities"
    
    OUTPUT_DIR = BASE_DIR / "outputs"
    METRICS_DIR = OUTPUT_DIR / "metrics"
    NETWORK_DIR = OUTPUT_DIR / "network"
    
    BROOKLYN_CENTER = [40.678, -73.944]
    BROOKLYN_BOUNDS = {'min_lat': 40.57, 'max_lat': 40.74, 'min_lon': -74.04, 'max_lon': -73.85}
    
    EDGE_CASE_IOU_THRESHOLD = 0.3
    EDGE_CASE_CONFIDENCE_VAR = 0.15


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_tile_list():
    """Load available tiles with coordinates."""
    config = AppConfig()
    tiles = []
    
    gt_files = {f.stem.replace("_gt", ""): f for f in config.GT_DIR.glob("*_gt.tif")} if config.GT_DIR.exists() else {}
    prob_files = {f.stem.replace("_prob", ""): f for f in config.PROB_DIR.glob("*_prob.tif")} if config.PROB_DIR.exists() else {}
    
    for tile_name in sorted(set(gt_files.keys()) & set(prob_files.keys())):
        tile_file = None
        for ext in ['.png', '.tif', '.jpg']:
            candidate = config.TILES_DIR / f"{tile_name}{ext}"
            if candidate.exists():
                tile_file = candidate
                break
        
        if tile_file:
            coords = extract_tile_coords(tile_name)
            tiles.append({
                'name': tile_name,
                'tile_path': str(tile_file),
                'gt_path': str(gt_files[tile_name]),
                'prob_path': str(prob_files[tile_name]),
                'lat': coords[0] if coords else 40.678 + np.random.uniform(-0.05, 0.05),
                'lon': coords[1] if coords else -73.944 + np.random.uniform(-0.05, 0.05)
            })
    
    return tiles


def extract_tile_coords(tile_name: str) -> Optional[Tuple[float, float]]:
    """Extract lat/lon from tile name."""
    try:
        parts = tile_name.replace("tile_", "").replace("stitched_", "").split("_")
        if len(parts) >= 3:
            z, x, y = int(parts[0]), int(parts[1]), int(parts[2])
            n = 2 ** z
            lon = x / n * 360.0 - 180.0
            lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
            return (lat, lon)
    except:
        pass
    return None


@st.cache_data
def load_tile_data(tile_path: str, gt_path: str, prob_path: str):
    """Load tile data."""
    try:
        original = np.array(Image.open(tile_path).convert('RGB'))
        
        with rasterio.open(gt_path) as src:
            gt = src.read(1)
        with rasterio.open(prob_path) as src:
            probs = src.read(1)
        
        h, w = original.shape[:2]
        if gt.shape != (h, w):
            gt = np.array(Image.fromarray(gt).resize((w, h), Image.NEAREST))
        if probs.shape != (h, w):
            probs = np.array(Image.fromarray((probs * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255
        
        return {'original': original, 'ground_truth': gt, 'probabilities': probs}
    except Exception as e:
        return None


@st.cache_data
def load_metrics():
    """Load metrics."""
    config = AppConfig()
    metrics = {'aggregate': {}, 'per_tile': [], 'calibration': {}}
    
    for key, filename in [('aggregate', 'aggregate_metrics.json'), 
                          ('per_tile', 'tile_metrics.json'),
                          ('calibration', 'calibration.json')]:
        path = config.METRICS_DIR / filename
        if path.exists():
            with open(path) as f:
                metrics[key] = json.load(f)
    return metrics


@st.cache_data
def load_stage_b_results():
    """Load Stage B results."""
    path = AppConfig.NETWORK_DIR / "stage_b_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_baseline_data():
    """Load baselines."""
    config = AppConfig()
    baselines = {}
    
    osm_path = config.DATA_DIR / "raw" / "osm" / "osm_crossings_brooklyn.geojson"
    if osm_path.exists():
        try:
            gdf = gpd.read_file(osm_path)
            bounds = config.BROOKLYN_BOUNDS
            baselines['osm'] = gdf.cx[bounds['min_lon']:bounds['max_lon'], 
                                      bounds['min_lat']:bounds['max_lat']]
        except:
            pass
    
    lion_path = config.DATA_DIR / "raw" / "lion" / "lion_brooklyn.geojson"
    if lion_path.exists():
        try:
            baselines['lion'] = gpd.read_file(lion_path)
        except:
            pass
    
    return baselines


# =============================================================================
# COMPUTATION FUNCTIONS
# =============================================================================

def compute_metrics(gt: np.ndarray, probs: np.ndarray, threshold: float) -> Dict:
    """Compute metrics."""
    pred = (probs >= threshold).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    
    tp = np.sum((pred == 1) & (gt_bin == 1))
    fp = np.sum((pred == 1) & (gt_bin == 0))
    fn = np.sum((pred == 0) & (gt_bin == 1))
    tn = np.sum((pred == 0) & (gt_bin == 0))
    
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'iou': float(iou), 'f1': float(f1),
        'precision': float(precision), 'recall': float(recall),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }


def compute_calibration(probs: np.ndarray, gt: np.ndarray, n_bins: int = 10) -> Dict:
    """Compute calibration."""
    probs_flat = probs.flatten()
    labels_flat = (gt.flatten() > 0).astype(float)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accs, bin_counts = [], []
    
    for i in range(n_bins):
        mask = (probs_flat >= bin_edges[i]) & (probs_flat < bin_edges[i + 1])
        count = np.sum(mask)
        bin_counts.append(int(count))
        bin_accs.append(float(np.mean(labels_flat[mask])) if count > 0 else np.nan)
    
    ece = sum((bin_counts[i] / len(probs_flat)) * abs(bin_accs[i] - bin_centers[i])
              for i in range(n_bins) if bin_counts[i] > 0 and not np.isnan(bin_accs[i]))
    
    return {'ece': ece, 'bin_centers': bin_centers.tolist(), 
            'bin_accuracies': bin_accs, 'bin_counts': bin_counts}


def compute_risk_coverage(probs: np.ndarray, gt: np.ndarray) -> Dict:
    """Compute risk-coverage."""
    probs_flat = probs.flatten()
    labels_flat = (gt.flatten() > 0).astype(float)
    
    thresholds = np.linspace(0, 1, 20)
    coverages, accuracies = [], []
    
    for t in thresholds:
        mask = probs_flat >= t
        coverages.append(np.mean(mask))
        accuracies.append(np.mean(labels_flat[mask] == 1) if np.sum(mask) > 0 else np.nan)
    
    return {'thresholds': thresholds.tolist(), 'coverages': coverages, 'accuracies': accuracies}


def detect_edge_cases(tile_data: dict, metrics: dict) -> Dict:
    """Detect edge cases."""
    flags = []
    
    if metrics['iou'] < AppConfig.EDGE_CASE_IOU_THRESHOLD:
        flags.append({'type': 'Low IoU', 'description': f"IoU={metrics['iou']:.2f}", 'severity': 'high'})
    
    probs = tile_data['probabilities']
    uncertain = np.mean((probs > 0.3) & (probs < 0.7))
    if uncertain > AppConfig.EDGE_CASE_CONFIDENCE_VAR:
        flags.append({'type': 'High Uncertainty', 'description': f"{uncertain:.1%} uncertain", 'severity': 'medium'})
    
    if metrics['fp'] > 0 and metrics['fn'] > 0:
        ratio = max(metrics['fp'], metrics['fn']) / min(metrics['fp'], metrics['fn'])
        if ratio > 5:
            flags.append({'type': 'Error Imbalance', 'description': f"Ratio {ratio:.1f}:1", 'severity': 'medium'})
    
    gt_pixels = np.sum(tile_data['ground_truth'] > 0)
    if gt_pixels < 100:
        flags.append({'type': 'Sparse GT', 'description': f"{gt_pixels} pixels", 'severity': 'low'})
    
    return {'is_edge_case': len(flags) > 0, 'flags': flags,
            'score': sum(3 if f['severity'] == 'high' else 2 if f['severity'] == 'medium' else 1 for f in flags)}


def compute_quality_score(stage_a: dict, stage_b: dict) -> Dict:
    """Compute quality score."""
    # Handle both 'iou' and 'mean_iou' key formats
    iou = stage_a.get('mean_iou', stage_a.get('iou', 0))
    f1 = stage_a.get('mean_f1', stage_a.get('f1', 0))
    recall = stage_a.get('mean_recall', stage_a.get('recall', 0))
    precision = stage_a.get('mean_precision', stage_a.get('precision', 0))
    coverage = min(stage_b.get('detection_coverage', 0), 1.0)
    
    stage_a_score = 0.20 * iou + 0.20 * f1 + 0.10 * recall + 0.10 * precision
    stage_b_score = 0.25 * coverage + 0.15 * coverage
    total = min(stage_a_score + stage_b_score, 1.0)
    
    return {
        'total': total,
        'stage_a_score': stage_a_score / 0.6 if stage_a_score > 0 else 0,
        'stage_b_score': stage_b_score / 0.4 if stage_b_score > 0 else 0,
        'breakdown': {'iou': iou, 'f1': f1, 'recall': recall, 'precision': precision, 'coverage': coverage}
    }


def apply_morphology(mask: np.ndarray, op: str, kernel_size: int = 3) -> np.ndarray:
    """Apply morphology."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    ops = {'opening': morphology.opening, 'closing': morphology.closing,
           'dilation': morphology.dilation, 'erosion': morphology.erosion}
    return ops.get(op, lambda m, k: m)(mask, kernel) if op != 'none' else mask


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_error_overlay(gt, pred, original, alpha):
    """Create error overlay."""
    gt_bin = (gt > 0).astype(np.uint8)
    overlay = original.copy().astype(float)
    
    overlay[(gt_bin == 1) & (pred == 1)] = overlay[(gt_bin == 1) & (pred == 1)] * (1-alpha) + np.array([46, 204, 113]) * alpha
    overlay[(gt_bin == 0) & (pred == 1)] = overlay[(gt_bin == 0) & (pred == 1)] * (1-alpha) + np.array([231, 76, 60]) * alpha
    overlay[(gt_bin == 1) & (pred == 0)] = overlay[(gt_bin == 1) & (pred == 0)] * (1-alpha) + np.array([52, 152, 219]) * alpha
    
    return overlay.astype(np.uint8)


def create_confidence_heatmap(probs):
    """Create heatmap."""
    cmap = plt.cm.viridis
    return (cmap(probs)[:, :, :3] * 255).astype(np.uint8)


def create_interactive_reliability_diagram(cal: Dict) -> alt.Chart:
    """Create interactive reliability diagram with Altair."""
    df = pd.DataFrame({
        'Bin Center': cal['bin_centers'],
        'Accuracy': cal['bin_accuracies'],
        'Count': cal['bin_counts']
    }).dropna()
    
    bars = alt.Chart(df).mark_bar(opacity=0.8).encode(
        x=alt.X('Bin Center:Q', scale=alt.Scale(domain=[0, 1]), title='Predicted Probability'),
        y=alt.Y('Accuracy:Q', scale=alt.Scale(domain=[0, 1]), title='Actual Accuracy'),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title='Sample Count')),
        tooltip=['Bin Center', 'Accuracy', 'Count']
    ).properties(width=300, height=250, title=f"Reliability Diagram (ECE={cal['ece']:.4f})")
    
    line_df = pd.DataFrame({'x': [0, 1], 'y': [0, 1]})
    line = alt.Chart(line_df).mark_line(strokeDash=[5, 5], color='black').encode(x='x:Q', y='y:Q')
    
    return bars + line


def create_interactive_risk_coverage(rc: Dict) -> alt.Chart:
    """Create interactive risk-coverage with Altair."""
    df = pd.DataFrame({
        'Threshold': rc['thresholds'],
        'Coverage': rc['coverages'],
        'Accuracy': rc['accuracies']
    }).dropna()
    
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('Coverage:Q', scale=alt.Scale(domain=[0, 1]), title='Coverage'),
        y=alt.Y('Accuracy:Q', scale=alt.Scale(domain=[0, 1]), title='Accuracy'),
        tooltip=['Threshold', 'Coverage', 'Accuracy']
    ).properties(width=300, height=250, title='Risk-Coverage Curve')
    
    valid_df = df[df['Coverage'] > 0.1]
    if len(valid_df) > 0:
        best = valid_df.loc[valid_df['Accuracy'].idxmax()]
        point = alt.Chart(pd.DataFrame([best])).mark_point(size=200, color='red').encode(
            x='Coverage:Q', y='Accuracy:Q'
        )
        return chart + point
    
    return chart


def create_interactive_threshold_sweep(gt, probs) -> alt.Chart:
    """Create interactive threshold sweep with Altair."""
    thresholds = np.linspace(0, 1, 21)
    data = []
    for t in thresholds:
        m = compute_metrics(gt, probs, t)
        data.extend([
            {'Threshold': t, 'Metric': 'IoU', 'Value': m['iou']},
            {'Threshold': t, 'Metric': 'F1', 'Value': m['f1']},
            {'Threshold': t, 'Metric': 'Precision', 'Value': m['precision']},
            {'Threshold': t, 'Metric': 'Recall', 'Value': m['recall']}
        ])
    
    df = pd.DataFrame(data)
    best_f1 = df[df['Metric'] == 'F1'].loc[df[df['Metric'] == 'F1']['Value'].idxmax()]
    
    lines = alt.Chart(df).mark_line().encode(
        x=alt.X('Threshold:Q', title='Threshold (τ)'),
        y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Metric:N', scale=alt.Scale(
            domain=['IoU', 'F1', 'Precision', 'Recall'],
            range=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        )),
        tooltip=['Threshold', 'Metric', 'Value']
    ).properties(width=300, height=250, title='Metrics vs Threshold')
    
    rule = alt.Chart(pd.DataFrame([{'x': best_f1['Threshold']}])).mark_rule(
        strokeDash=[3, 3], color='gray'
    ).encode(x='x:Q')
    
    return lines + rule


def create_stage_a_metrics_chart(agg: Dict) -> alt.Chart:
    """Create Stage A metrics bar chart - handles both key formats."""
    # Handle both 'iou' and 'mean_iou' key formats
    iou = agg.get('mean_iou', agg.get('iou', 0))
    f1 = agg.get('mean_f1', agg.get('f1', 0))
    precision = agg.get('mean_precision', agg.get('precision', 0))
    recall = agg.get('mean_recall', agg.get('recall', 0))
    
    df = pd.DataFrame({
        'Metric': ['IoU', 'F1', 'Precision', 'Recall'],
        'Value': [iou, f1, precision, recall]
    })
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Value:Q', scale=alt.Scale(domain=[0, 1]), title='Score'),
        y=alt.Y('Metric:N', sort=['IoU', 'F1', 'Precision', 'Recall'], title=''),
        color=alt.Color('Metric:N', scale=alt.Scale(
            domain=['IoU', 'F1', 'Precision', 'Recall'],
            range=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        ), legend=None),
        tooltip=['Metric', alt.Tooltip('Value:Q', format='.3f')]
    ).properties(height=200, title='Stage A Metrics')
    
    return chart


def plot_quality_gauge(quality: float) -> plt.Figure:
    """Plot quality gauge."""
    fig, ax = plt.subplots(figsize=(4, 3))
    
    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=20, solid_capstyle='round')
    
    q_theta = np.linspace(0, np.pi * quality, 100)
    color = '#2ecc71' if quality > 0.7 else '#f39c12' if quality > 0.5 else '#e74c3c'
    ax.plot(np.cos(q_theta), np.sin(q_theta), color, linewidth=20, solid_capstyle='round')
    
    ax.text(0, 0.15, f'{quality:.0%}', ha='center', va='center', fontsize=28, fontweight='bold')
    label = 'Excellent' if quality > 0.8 else 'Good' if quality > 0.7 else 'Moderate' if quality > 0.5 else 'Needs Work'
    ax.text(0, -0.2, label, ha='center', va='center', fontsize=12, color=color, fontweight='bold')
    
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.4, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig


# =============================================================================
# TAB A: SEGMENTATION DETECTIVE
# =============================================================================

def render_tab_a():
    """Render Tab A."""
    
    st.header("🔍 Segmentation Detective")
    st.caption("Analyze pixel-level model performance and diagnose errors")
    
    tiles = load_tile_list()
    metrics_data = load_metrics()
    
    if not tiles:
        st.warning("⚠️ No tiles found. Run `python run_complete_pipeline.py` first.")
        return
    
    tile_names = [t['name'] for t in tiles]
    num_tiles = len(tiles)
    
    # Initialize state if not exists
    if 'current_tile_name' not in st.session_state:
        st.session_state.current_tile_name = tile_names[0]
    if 'tile_idx' not in st.session_state:
        st.session_state.tile_idx = 0
    
    # Ensure current_tile_name is valid
    if st.session_state.current_tile_name not in tile_names:
        st.session_state.current_tile_name = tile_names[0]
        st.session_state.tile_idx = 0
    
    # Ensure tile_idx matches current_tile_name (current_tile_name is the source of truth)
    expected_idx = tile_names.index(st.session_state.current_tile_name)
    if st.session_state.tile_idx != expected_idx:
        st.session_state.tile_idx = expected_idx
    
    current_idx = st.session_state.tile_idx
    
    # Navigation
    col1, col2, col3, col4, col5 = st.columns([1, 1, 4, 1, 1])
    
    with col1:
        if st.button("⏮️ First", use_container_width=True, disabled=(current_idx == 0)):
            st.session_state.current_tile_name = tile_names[0]
            st.session_state.tile_idx = 0
            st.rerun()
    
    with col2:
        if st.button("◀️ Prev", use_container_width=True, disabled=(current_idx == 0)):
            new_idx = current_idx - 1
            st.session_state.current_tile_name = tile_names[new_idx]
            st.session_state.tile_idx = new_idx
            st.rerun()
    
    with col3:
        selected = st.selectbox(
            "Select Tile", tile_names, index=current_idx,
            label_visibility="collapsed",
            key=f"dropdown_{st.session_state.current_tile_name}"
        )
        if selected != st.session_state.current_tile_name:
            st.session_state.current_tile_name = selected
            st.session_state.tile_idx = tile_names.index(selected)
            st.rerun()
    
    with col4:
        if st.button("Next ▶️", use_container_width=True, disabled=(current_idx >= num_tiles - 1)):
            new_idx = current_idx + 1
            st.session_state.current_tile_name = tile_names[new_idx]
            st.session_state.tile_idx = new_idx
            st.rerun()
    
    with col5:
        if st.button("Last ⏭️", use_container_width=True, disabled=(current_idx >= num_tiles - 1)):
            st.session_state.current_tile_name = tile_names[-1]
            st.session_state.tile_idx = num_tiles - 1
            st.rerun()
    
    st.progress((current_idx + 1) / num_tiles, text=f"Tile {current_idx + 1} of {num_tiles}")
    
    # Sidebar
    with st.sidebar:
        st.subheader("🎛️ Controls")
        threshold = st.slider("Threshold (τ)", 0.0, 1.0, 0.5, 0.05)
        
        st.divider()
        morph_op = st.selectbox("Morphology", ['none', 'opening', 'closing', 'dilation', 'erosion'])
        kernel_size = st.slider("Kernel Size", 3, 9, 3, 2) if morph_op != 'none' else 3
        
        st.divider()
        opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.1)
        
        st.divider()
        st.subheader("📌 Bookmarks")
        if 'bookmarks' not in st.session_state:
            st.session_state.bookmarks = []
        
        if st.button("Bookmark Current", use_container_width=True):
            current = tile_names[current_idx]
            if current not in st.session_state.bookmarks:
                st.session_state.bookmarks.append(current)
                st.toast("Bookmarked!")
        
        for i, bm in enumerate(st.session_state.bookmarks[:5]):
            c1, c2 = st.columns([4, 1])
            with c1:
                if st.button(f"📍 {bm[:15]}...", key=f"bm_{i}", use_container_width=True):
                    st.session_state.current_tile_name = bm
                    st.session_state.tile_idx = tile_names.index(bm)
                    st.rerun()
            with c2:
                if st.button("✕", key=f"del_{i}"):
                    st.session_state.bookmarks.remove(bm)
                    st.rerun()
    
    # Load tile
    tile_info = tiles[current_idx]
    tile_data = load_tile_data(tile_info['tile_path'], tile_info['gt_path'], tile_info['prob_path'])
    
    if not tile_data:
        st.error("Failed to load tile")
        return
    
    original = tile_data['original']
    gt = tile_data['ground_truth']
    probs = tile_data['probabilities']
    
    pred = (probs >= threshold).astype(np.uint8)
    if morph_op != 'none':
        pred = apply_morphology(pred, morph_op, kernel_size)
    
    metrics = compute_metrics(gt, probs, threshold)
    edge_info = detect_edge_cases(tile_data, metrics)
    
    # Edge Case Alert
    if edge_info['is_edge_case']:
        with st.expander("⚠️ **EDGE CASE DETECTED**", expanded=True):
            for flag in edge_info['flags']:
                icon = "🔴" if flag['severity'] == 'high' else "🟡" if flag['severity'] == 'medium' else "🟢"
                st.markdown(f"{icon} **{flag['type']}**: {flag['description']}")
    
    # Linked Views
    st.subheader("📷 Linked View Strip")
    cols = st.columns(5)
    
    with cols[0]:
        st.markdown("**1. Original**")
        st.image(original, use_container_width=True)
    
    with cols[1]:
        st.markdown("**2. Ground Truth**")
        gt_vis = original.copy()
        gt_vis[gt > 0] = [255, 140, 0]
        st.image(gt_vis, use_container_width=True)
    
    with cols[2]:
        st.markdown(f"**3. Prediction (τ={threshold:.2f})**")
        pred_vis = original.copy()
        pred_vis[pred > 0] = [0, 255, 0]
        st.image(pred_vis, use_container_width=True)
    
    with cols[3]:
        st.markdown("**4. Confidence**")
        st.image(create_confidence_heatmap(probs), use_container_width=True)
    
    with cols[4]:
        st.markdown("**5. Errors**")
        st.image(create_error_overlay(gt, pred, original, opacity), use_container_width=True)
    
    st.markdown("🟢 True Positive | 🔴 False Positive | 🔵 False Negative")
    
    # Metrics
    st.divider()
    st.subheader("📊 Live Metrics")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IoU", f"{metrics['iou']:.3f}", "Good" if metrics['iou'] > 0.5 else "Low",
              delta_color="normal" if metrics['iou'] > 0.5 else "inverse")
    m2.metric("F1", f"{metrics['f1']:.3f}")
    m3.metric("Precision", f"{metrics['precision']:.3f}")
    m4.metric("Recall", f"{metrics['recall']:.3f}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ TP", f"{metrics['tp']:,}")
    c2.metric("❌ FP", f"{metrics['fp']:,}")
    c3.metric("❌ FN", f"{metrics['fn']:,}")
    c4.metric("✅ TN", f"{metrics['tn']:,}")
    
    # Interactive Charts
    st.divider()
    st.subheader("📈 Analysis Charts")
    
    ch1, ch2, ch3 = st.columns(3)
    
    with ch1:
        cal = compute_calibration(probs, gt)
        chart = create_interactive_reliability_diagram(cal)
        st.altair_chart(chart, use_container_width=True)
    
    with ch2:
        rc = compute_risk_coverage(probs, gt)
        chart = create_interactive_risk_coverage(rc)
        st.altair_chart(chart, use_container_width=True)
    
    with ch3:
        chart = create_interactive_threshold_sweep(gt, probs)
        st.altair_chart(chart, use_container_width=True)
    
    # Research Questions
    st.divider()
    with st.expander("🔬 Research Questions Analysis", expanded=False):
        q1, q2 = st.columns(2)
        
        with q1:
            st.markdown("**Q1: Perception & Patterns**")
            fp_rate = metrics['fp'] / (metrics['fp'] + metrics['tp'] + 1)
            fn_rate = metrics['fn'] / (metrics['fn'] + metrics['tp'] + 1)
            st.write(f"• FP Rate: {fp_rate:.1%} | FN Rate: {fn_rate:.1%}")
            st.write("→ Model **misses**" if fn_rate > fp_rate else "→ Model **over-detects**")
            
            st.markdown("**Q2: Confidence & Choice**")
            st.write(f"• Threshold: τ = {threshold}")
            
            st.markdown("**Q5: Evidence to Trust**")
            st.write(f"• ECE: {cal['ece']:.4f} " + ("✅" if cal['ece'] < 0.1 else "⚠️"))
        
        with q2:
            st.markdown("**Q6: Usability Cues**")
            st.write("🟢=Correct 🔴=Error 🔵=Missed")
            
            st.markdown("**Q7: Edge Cases**")
            if edge_info['is_edge_case']:
                st.write(f"⚠️ Edge case (score: {edge_info['score']})")
            else:
                st.write("✅ Normal case")
    
    # Aggregate Metrics
    if metrics_data.get('per_tile'):
        st.divider()
        st.subheader("📊 Aggregate Metrics (All Tiles)")
        
        agg_col1, agg_col2 = st.columns(2)
        
        with agg_col1:
            df = pd.DataFrame(metrics_data['per_tile'])
            if len(df) > 0 and 'iou' in df.columns:
                melt_df = df[['iou', 'f1', 'precision', 'recall']].melt(var_name='Metric', value_name='Value')
                chart = alt.Chart(melt_df).mark_boxplot().encode(
                    x='Metric:N',
                    y='Value:Q',
                    color='Metric:N'
                ).properties(title='Metrics Distribution', height=300)
                st.altair_chart(chart, use_container_width=True)
        
        with agg_col2:
            df = pd.DataFrame(metrics_data['per_tile'])
            if 'tile' in df.columns:
                st.dataframe(df[['tile', 'iou', 'f1', 'precision', 'recall']].round(3), height=300)


# =============================================================================
# TAB B: NETWORK QUALITY INSPECTOR
# =============================================================================

def render_tab_b():
    """Render Tab B with all visualizations."""
    
    st.header("🗺️ Network Quality Inspector")
    st.caption("Evaluate network quality and baseline agreement")
    
    stage_b = load_stage_b_results()
    metrics_data = load_metrics()
    baselines = load_baseline_data()
    tiles = load_tile_list()
    
    if not stage_b:
        st.warning("⚠️ No Stage B results. Run `python run_complete_pipeline.py` first.")
        return
    
    tile_names = [t['name'] for t in tiles]
    
    # Sidebar
    with st.sidebar:
        st.subheader("🗺️ Map Controls")
        show_osm = st.checkbox("Show OSM Crossings", value=True)
        show_lion = st.checkbox("Show LION Streets", value=False)
        show_tiles = st.checkbox("Show Project Tiles", value=True)
        buffer_dist = st.slider("Buffer (m)", 1.0, 10.0, 2.0, 0.5)
    
    # Quality Score
    stage_a_agg = stage_b.get('aggregate_metrics', metrics_data.get('aggregate', {}))
    quality_info = compute_quality_score(stage_a_agg, stage_b)
    
    agg = stage_a_agg
    detected = stage_b.get('total_detected_pixels', 0)
    gt_pixels = stage_b.get('total_gt_pixels', 0)
    coverage = stage_b.get('detection_coverage', 0)
    quality = quality_info['total']
    
    # ==========================================================================
    # QUALITY OVERVIEW
    # ==========================================================================
    
    st.subheader("📊 Quality Overview")
    
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Detected Pixels", f"{detected:,}")
    q2.metric("Ground Truth Pixels", f"{gt_pixels:,}")
    q3.metric("Coverage", f"{min(coverage * 100, 100):.1f}%")
    q4.metric("Quality Score", f"{quality:.0%}", "Good" if quality > 0.6 else "Moderate")
    
    # ==========================================================================
    # NETWORK MAP - SIMPLIFIED (No auto-rerun on click)
    # ==========================================================================
    
    st.divider()
    st.subheader("🗺️ Network Map")
    
    if FOLIUM_AVAILABLE:
        m = folium.Map(location=AppConfig.BROOKLYN_CENTER, zoom_start=13, tiles='CartoDB positron')
        
        # OSM Crossings (orange)
        if show_osm and 'osm' in baselines:
            osm = baselines['osm']
            if len(osm) > 0:
                for _, row in osm.sample(min(500, len(osm))).iterrows():
                    if row.geometry.geom_type == 'Point':
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=3, color='#f39c12', fill=True, fillOpacity=0.6
                        ).add_to(m)
        
        # LION Streets (gray)
        if show_lion and 'lion' in baselines:
            lion = baselines['lion']
            if len(lion) > 0:
                folium.GeoJson(
                    lion.sample(min(300, len(lion))).to_json(),
                    style_function=lambda x: {'color': '#7f8c8d', 'weight': 2, 'opacity': 0.5}
                ).add_to(m)
        
        # Project Tiles (CYAN markers)
        if show_tiles:
            for i, t in enumerate(tiles):
                folium.CircleMarker(
                    location=[t['lat'], t['lon']],
                    radius=10,
                    color='#008B8B',
                    fill=True,
                    fillColor='#00CED1',
                    fillOpacity=0.9,
                    weight=2,
                    popup=t['name'],
                    tooltip=t['name']
                ).add_to(m)
        
        # Render map (no click handling that causes reloads)
        st_folium(m, width=None, height=400, returned_objects=[])
        
        st.caption("🟠 OSM Crossings | 🔵 CYAN = Project Tiles | Gray = LION Streets")
        
        # Initialize Tab B selected tile in session state
        if 'tab_b_selected' not in st.session_state:
            st.session_state.tab_b_selected = tile_names[0]
        
        # Ensure it's valid
        if st.session_state.tab_b_selected not in tile_names:
            st.session_state.tab_b_selected = tile_names[0]
        
        # Get current index for dropdown
        tab_b_current_idx = tile_names.index(st.session_state.tab_b_selected)
        
        st.markdown("---")
        st.markdown("**🎯 Select Tile to View in Tab A:**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Dropdown with proper index from session state
            selected_tile = st.selectbox(
                "Choose a tile:",
                tile_names,
                index=tab_b_current_idx,
                key="tab_b_tile_selector",
                label_visibility="collapsed"
            )
            
            # Update session state if dropdown changed
            if selected_tile != st.session_state.tab_b_selected:
                st.session_state.tab_b_selected = selected_tile
        
        with col2:
            if st.button("🔍 View in Tab A", type="primary", use_container_width=True):
                # Set BOTH variables that Tab A uses
                selected_idx = tile_names.index(selected_tile)
                st.session_state.tile_idx = selected_idx
                st.session_state.current_tile_name = selected_tile
                st.rerun()  # Force rerun so Tab A picks up the change
        
        # Show confirmation of current selection
        st.info(f"📍 Selected: **{selected_tile}** — Click the button, then switch to Tab A")
    
    else:
        st.warning("Install folium: `pip install folium streamlit-folium`")
    
    # ==========================================================================
    # AGREEMENT PANEL
    # ==========================================================================
    
    st.divider()
    st.subheader("📏 Agreement Analysis")
    
    ag1, ag2 = st.columns(2)
    
    with ag1:
        st.markdown("**A→B Linkage**")
        
        # Get values with fallbacks
        iou_val = agg.get('mean_iou', agg.get('iou', 0))
        f1_val = agg.get('mean_f1', agg.get('f1', 0))
        
        linkage_df = pd.DataFrame({
            'Stage': ['A (Pixel)', 'A (Pixel)', 'B (Network)', 'B (Network)'],
            'Metric': ['IoU', 'F1', 'Coverage', 'Quality'],
            'Value': [iou_val, f1_val, min(coverage, 1), quality]
        })
        
        chart = alt.Chart(linkage_df).mark_bar().encode(
            x=alt.X('Value:Q', scale=alt.Scale(domain=[0, 1])),
            y='Metric:N',
            color=alt.Color('Stage:N', scale=alt.Scale(
                domain=['A (Pixel)', 'B (Network)'],
                range=['#3498db', '#2ecc71']
            )),
            tooltip=['Stage', 'Metric', alt.Tooltip('Value:Q', format='.3f')]
        ).properties(height=200)
        
        st.altair_chart(chart, use_container_width=True)
    
    with ag2:
        st.markdown("**Interpretation**")
        
        if quality > 0.7:
            st.success("✅ **Strong Quality**: Network is well-connected.")
        elif quality > 0.5:
            st.warning("⚠️ **Moderate Quality**: Some gaps may exist.")
        else:
            st.error("❌ **Needs Improvement**: Significant gaps.")
        
        iou_pct = agg.get('mean_iou', agg.get('iou', 0))
        f1_pct = agg.get('mean_f1', agg.get('f1', 0))
        
        st.markdown(f"""
        **Summary:**
        - Pixel IoU: {iou_pct:.1%}
        - Pixel F1: {f1_pct:.1%}
        - Network Coverage: {min(coverage * 100, 100):.1f}%
        - Baseline Features: {stage_b.get('baseline_features', 0):,}
        """)
    
    # ==========================================================================
    # NETWORK STATISTICS (3 charts)
    # ==========================================================================
    
    st.divider()
    st.subheader("📊 Network Statistics")
    
    ns1, ns2, ns3 = st.columns(3)
    
    with ns1:
        st.markdown("**Detection vs Ground Truth**")
        det_df = pd.DataFrame({
            'Category': ['Detected', 'Ground Truth'],
            'Pixels': [detected, gt_pixels]
        })
        chart = alt.Chart(det_df).mark_bar().encode(
            x='Category:N',
            y='Pixels:Q',
            color=alt.Color('Category:N', scale=alt.Scale(
                domain=['Detected', 'Ground Truth'],
                range=['#3498db', '#2ecc71']
            )),
            tooltip=['Category', 'Pixels']
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)
    
    with ns2:
        st.markdown("**Quality Score**")
        fig = plot_quality_gauge(quality)
        st.pyplot(fig)
        plt.close(fig)
    
    with ns3:
        st.markdown("**Stage A Metrics**")
        chart = create_stage_a_metrics_chart(agg)
        st.altair_chart(chart, use_container_width=True)
    
    # ==========================================================================
    # RESEARCH QUESTIONS
    # ==========================================================================
    
    st.divider()
    with st.expander("🔬 Research Questions (Tab B)", expanded=False):
        rq1, rq2 = st.columns(2)
        
        with rq1:
            st.markdown("**Q3: Placement & Plausibility** ✅")
            st.write("• Map shows detections in geographic context")
            st.write("• Cyan markers = project tile locations")
            
            st.markdown("**Q4: Agreement & Disagreement** ✅")
            st.write("• OSM overlay enables comparison")
            st.write("• Coverage metric shows match rate")
        
        with rq2:
            st.markdown("**Q5: From Evidence to Trust** ✅")
            st.write("• Quality score synthesizes A+B metrics")
            st.write("• A→B linkage shows causal relationship")
    
    # ==========================================================================
    # EXPORT & PROVENANCE
    # ==========================================================================
    
    st.divider()
    st.subheader("📥 Export & Provenance")
    
    ex1, ex2 = st.columns(2)
    
    with ex1:
        st.markdown("**Export Data**")
        
        st.download_button(
            "📥 Stage B Results (JSON)",
            json.dumps(stage_b, indent=2),
            "stage_b_results.json",
            "application/json"
        )
        
        if metrics_data.get('per_tile'):
            df = pd.DataFrame(metrics_data['per_tile'])
            st.download_button(
                "📥 Tile Metrics (CSV)",
                df.to_csv(index=False),
                "tile_metrics.csv",
                "text/csv"
            )
    
    with ex2:
        st.markdown("**Provenance**")
        st.code(f"""
Data Sources:
  • Tiles: {len(tiles)} processed
  • OSM: {len(baselines.get('osm', [])) if baselines else 0} crossings
  • LION: {len(baselines.get('lion', [])) if baselines else 0} segments

Processing:
  • Threshold (τ): 0.5
  • Buffer: {buffer_dist}m
  • CRS: EPSG:4326

Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.title("🚶 CrossCheck-NYC")
    st.caption("Visual QA System for AI-Derived Crosswalk Maps | Brooklyn, NY")
    
    tab_a, tab_b = st.tabs(["🔍 Tab A: Segmentation Detective", "🗺️ Tab B: Network Inspector"])
    
    with tab_a:
        render_tab_a()
    
    with tab_b:
        render_tab_b()
    
    st.divider()
    st.caption("CrossCheck-NYC | Powered by Tile2Net")


if __name__ == "__main__":
    main()