# app_optimized.py
"""
Optimized CrossCheck-NYC Dashboard
- Performance improvements: caching, precomputed masks, memoized charts/overlays.
- Functional behavior and UI kept unchanged from original app.py.
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Page config - MUST be first
st.set_page_config(
    page_title="CrossCheck-NYC",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports
from PIL import Image
import altair as alt
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURATION (unchanged)
# =============================================================================
DATA_DIR = Path("./data")
OUTPUT_DIR = DATA_DIR / "outputs"
METRICS_DIR = DATA_DIR / "results" / "metrics"

LOCATIONS = {
    "financial_district": {"name": "Financial District", "center": [40.7075, -74.0095]},
    "east_village": {"name": "East Village", "center": [40.7265, -73.9865]},
    "bay_ridge": {"name": "Bay Ridge", "center": [40.6320, -74.0265]},
    "downtown_brooklyn": {"name": "Downtown Brooklyn", "center": [40.6915, -73.9850]},
    "kew_gardens": {"name": "Kew Gardens", "center": [40.7100, -73.8275]}
}

# tile2net colors
SEG_COLORS = {
    'crosswalk': (255, 0, 0),    # RED
    'road': (0, 255, 0),         # GREEN
    'sidewalk': (0, 0, 255)      # BLUE
}
COLOR_TOL = 40
MIN_CW_PX = 50

FACTOR_CATS = {
    "Building Height": ["none", "low", "medium", "high"],
    "Tree Proximity": ["none", "few", "moderate", "many"],
    "Surface Contrast": ["low", "medium", "high"],
    "Road Type": ["primary", "secondary", "residential", "other"],
    "Crossing Markings": ["zebra", "lines", "surface", "unknown"]
}

TIPS = {
    "iou": "**IoU (Intersection over Union)**: Measures overlap between prediction and ground truth. Range 0-1, higher is better.",
    "precision": "**Precision**: Of all detected crosswalks, what percentage were correct? High precision = few false alarms.",
    "recall": "**Recall**: Of all real crosswalks, what percentage did we find? High recall = few misses.",
    "f1": "**F1 Score**: Harmonic mean of precision and recall. Balances both metrics.",
    "tp": "**True Positive (TP)**: Correctly detected crosswalk pixels. Model said YES, actually YES.",
    "fp": "**False Positive (FP)**: Incorrectly detected pixels. Model said YES, actually NO (false alarm).",
    "fn": "**False Negative (FN)**: Missed crosswalk pixels. Model said NO, actually YES.",
    "tn": "**True Negative (TN)**: Correctly identified as not crosswalk. Model said NO, actually NO.",
    "gt": "**Ground Truth**: Reference crosswalk data from OpenStreetMap.",
    "threshold": "How do small changes like threshold or clean-up, affect what the model marks as a crosswalk?**Threshold (τ)**: Confidence cutoff for detection. Higher = stricter (fewer detections, higher precision). Lower = looser (more detections, higher recall).",
    "morphology": "**Morphology**: Post-processing operations. **Opening** removes small noise. **Closing** fills small gaps. **Erosion** shrinks regions. **Dilation** expands regions.",
    "opacity": "**Overlay Opacity**: Controls transparency of the error overlay visualization. Higher = more visible overlay colors.",
}

# =============================================================================
# DATA LOADING - cached
# =============================================================================
@st.cache_data
def get_locations() -> List[str]:
    available = []
    if OUTPUT_DIR.exists():
        for loc_id in LOCATIONS.keys():
            seg_dir = OUTPUT_DIR / loc_id / "segmentation"
            if seg_dir.exists() and list(seg_dir.rglob("sidebside_*.png")):
                available.append(loc_id)
    return available

@st.cache_data
def get_tiles(loc: str) -> List[Dict]:
    tiles = []
    seg_dir = OUTPUT_DIR / loc / "segmentation"
    if seg_dir.exists():
        for p in sorted(seg_dir.rglob("sidebside_*.png")):
            tiles.append({'path': str(p), 'name': p.name})
    return tiles

@st.cache_data
def get_results() -> Dict:
    path = METRICS_DIR / "results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def get_feature_impact() -> Dict:
    path = METRICS_DIR / "feature_impact_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

# =============================================================================
# HELPERS - caching expensive ops
# =============================================================================
@st.cache_data
def load_image_as_np(path: str) -> np.ndarray:
    """Load image once and return as uint8 numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)

@st.cache_data
def split_image_cached(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Split the side-by-side image into original and segmentation halves (cached)."""
    img = load_image_as_np(path)
    w = img.shape[1] // 2
    left = img[:, :w].copy()
    right = img[:, w:].copy()
    return left, right

@st.cache_data
def tile_precompute_for_location(loc: str) -> Dict[str, Dict]:
    """
    Precompute masks and small per-tile metadata for all tiles in a location.
    Returns dict: tile_name -> {'path','orig','seg','gt_mask','road_mask','sidewalk_mask','has_crosswalk'}
    """
    tiles = get_tiles(loc)
    result = {}
    for t in tiles:
        try:
            orig, seg = split_image_cached(t['path'])
            # compute masks
            gt_mask = get_mask_from_seg_array(seg, 'crosswalk')
            road_mask = get_mask_from_seg_array(seg, 'road')
            sidewalk_mask = get_mask_from_seg_array(seg, 'sidewalk')
            result[t['name']] = {
                'path': t['path'],
                'orig': orig,
                'seg': seg,
                'gt_mask': gt_mask,
                'road_mask': road_mask,
                'sidewalk_mask': sidewalk_mask,
                'has_crosswalk': bool(gt_mask.sum() >= MIN_CW_PX)
            }
        except Exception:
            # skip tiles that fail
            continue
    return result

def _abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(float) - b.astype(float))

@st.cache_data
def get_mask_from_seg_array(seg: np.ndarray, cls: str) -> np.ndarray:
    """Create boolean mask for a class from segmentation array (cached by seg array contents)."""
    if seg is None or len(seg.shape) != 3:
        return np.zeros(seg.shape[:2] if seg is not None else (0, 0), dtype=bool)
    target = np.array(SEG_COLORS.get(cls, (0, 0, 0)))
    # vectorized close-match
    return np.all(_abs_diff(seg, target) < COLOR_TOL, axis=2)

# Morphology memoized by tile-name + op + k
@st.cache_data
def apply_morph_cached(tile_name: str, morph_op: str, k: int, loc: str) -> np.ndarray:
    """
    Apply morphological op to a tile's GT mask (we simulate prediction by modifying GT).
    Returns boolean mask.
    """
    tile_cache = tile_precompute_for_location(loc)
    tile = tile_cache.get(tile_name)
    if tile is None:
        return np.zeros((0, 0), dtype=bool)
    mask = tile['gt_mask'].astype(bool)
    if morph_op == 'none':
        return mask
    struct = np.ones((k, k), dtype=bool)
    ops = {'erosion': binary_erosion, 'dilation': binary_dilation,
           'opening': binary_opening, 'closing': binary_closing}
    op_func = ops.get(morph_op, lambda x, **kw: x)
    # binary_xxx expects boolean array
    return op_func(mask, structure=struct)

@st.cache_data
def compute_metrics_cached(tile_name: str, morph_op: str, k: int, loc: str) -> Dict:
    """
    Compute metrics for a tile after applying morphology (cached).
    Uses apply_morph_cached as input.
    """
    tile_cache = tile_precompute_for_location(loc)
    tile = tile_cache.get(tile_name)
    if tile is None:
        return {'tp':0,'fp':0,'fn':0,'tn':0,'precision':0.0,'recall':0.0,'f1':0.0,'iou':0.0}
    gt = tile['gt_mask'].astype(bool)
    # simulate prediction as transformed GT (as in original app)
    pred = apply_morph_cached(tile_name, morph_op, k, loc).astype(bool)
    # expand gt a little to be tolerant
    gt_exp = binary_dilation(gt, iterations=2)
    p = pred
    tp = int(np.sum(p & gt))
    fp = int(np.sum(p & ~gt_exp))
    fn = int(np.sum(gt_exp & ~p))
    tn = int(np.sum(~p & ~gt_exp))
    eps = 1e-8
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': round(prec, 4), 'recall': round(rec, 4),
            'f1': round(f1, 4), 'iou': round(iou, 4)}

@st.cache_data
def create_error_overlay_cached(tile_name: str, alpha: float, morph_op: str, k: int, loc: str) -> np.ndarray:
    """
    Create overlay for a tile (cached).
    Uses tile precompute, morphology and produces uint8 overlay image.
    """
    tile_cache = tile_precompute_for_location(loc)
    tile = tile_cache.get(tile_name)
    if tile is None:
        return np.zeros((1,1,3), dtype=np.uint8)
    original = tile['orig']
    seg = tile['seg']
    gt_mask = tile['gt_mask'].astype(bool)
    pred_mask = apply_morph_cached(tile_name, morph_op, k, loc).astype(bool)
    overlay = original.copy().astype(float)
    tp_mask = gt_mask & pred_mask
    fp_mask = ~gt_mask & pred_mask
    fn_mask = gt_mask & ~pred_mask
    overlay[tp_mask] = overlay[tp_mask] * (1 - alpha) + np.array([255, 50, 50]) * alpha
    overlay[fp_mask] = overlay[fp_mask] * (1 - alpha) + np.array([255, 165, 0]) * alpha
    overlay[fn_mask] = overlay[fn_mask] * (1 - alpha) + np.array([52, 152, 219]) * alpha
    return overlay.astype(np.uint8)

# Chart caching helpers - cache by simple tuples/tuples of floats to ensure hashability
@st.cache_data
def viz_radar_cached(iou: float, f1: float, precision: float, recall: float):
    cats = ['IoU', 'F1', 'Precision', 'Recall']
    vals = [iou, f1, precision, recall]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    vals_plot = vals + [vals[0]]
    angles += angles[:1]
    ax.plot(angles, vals_plot, 'o-', linewidth=2, color='#3498db', markersize=10)
    ax.fill(angles, vals_plot, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Metrics Overview', size=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    for i, (angle, val) in enumerate(zip(angles[:-1], vals)):
        ax.annotate(f'{val:.2f}', xy=(angle, val), xytext=(angle, val + 0.12),
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color='#2c3e50')
    plt.tight_layout()
    return fig

@st.cache_data
def viz_gauge_cached(value: float, title: str = "Score"):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=20, solid_capstyle='round')
    q_theta = np.linspace(0, np.pi * min(value, 1), 100)
    color = '#27ae60' if value > 0.7 else '#f39c12' if value > 0.4 else '#e74c3c'
    ax.plot(np.cos(q_theta), np.sin(q_theta), color, linewidth=20, solid_capstyle='round')
    ax.text(0, 0.15, f'{value:.0%}', ha='center', va='center', fontsize=28, fontweight='bold')
    label = 'Excellent' if value > 0.8 else 'Good' if value > 0.6 else 'Fair' if value > 0.4 else 'Poor'
    ax.text(0, -0.2, label, ha='center', va='center', fontsize=12, fontweight='bold', color=color)
    ax.text(0, 0.75, title, ha='center', va='center', fontsize=10, color='gray')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.4, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig

@st.cache_data
def viz_donut_cached(data_tuple: Tuple[Tuple[str, int], ...], title: str = "Distribution"):
    data = dict(data_tuple)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    labels = [k for k, v in data.items() if v > 0]
    sizes = [v for v in data.values() if v > 0]
    if not sizes:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10)
        ax.axis('off')
        return fig
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12'][:len(labels)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', colors=colors, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=1),
        textprops={'fontsize': 8}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight('bold')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig

@st.cache_data
def viz_confusion_cached(tp: int, fn: int, fp: int, tn: int):
    metrics = {'tp':tp, 'fn':fn, 'fp':fp, 'tn':tn}
    fig, ax = plt.subplots(figsize=(4, 3.5))
    matrix = np.array([[metrics['tp'], metrics['fn']], [metrics['fp'], metrics['tn']]])
    max_val = matrix.max() if matrix.max() > 0 else 1
    matrix_norm = matrix / max_val
    ax.imshow(matrix_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Crosswalk', 'Not CW'], fontsize=10)
    ax.set_yticklabels(['Crosswalk', 'Not CW'], fontsize=10)
    ax.set_xlabel('Predicted', fontweight='bold', fontsize=11)
    ax.set_ylabel('Actual', fontweight='bold', fontsize=11)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
    labels = [['TP', 'FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            text_color = 'white' if matrix_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{labels[i][j]}\n{val:,}', ha='center', va='center',
                   fontsize=11, fontweight='bold', color=text_color)
    plt.tight_layout()
    return fig

# Altair charts are lightweight to build but we cache them by serializable data tuples for speed
@st.cache_data
def viz_threshold_chart_cached(rows_tuple: Tuple[Tuple[float, str, float], ...]):
    rows = [{'Threshold': r[0], 'Metric': r[1], 'Score': r[2]} for r in rows_tuple]
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('Threshold:Q', title='Threshold', scale=alt.Scale(domain=[0.3, 0.8])),
        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Metric:N', scale=alt.Scale(
            domain=['IoU', 'F1', 'Precision', 'Recall'],
            range=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        )),
        tooltip=['Threshold:Q', 'Metric:N', alt.Tooltip('Score:Q', format='.2f')]
    ).properties(height=220, title='Threshold Sensitivity').interactive()
    return chart

@st.cache_data
def viz_factor_bars_cached(chart_data_tuple: Tuple[Tuple[str, float, int], ...], title: str):
    df = pd.DataFrame([{'category': t[0], 'rate': t[1], 'count': t[2]} for t in chart_data_tuple])
    c = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X('rate:Q', title='Detection Rate', scale=alt.Scale(domain=[0, 1]),
               axis=alt.Axis(format='%', titleFontSize=11, labelFontSize=10)),
        y=alt.Y('category:N', title=None, sort='-x',
               axis=alt.Axis(labelFontSize=11)),
        color=alt.Color('rate:Q', scale=alt.Scale(
            domain=[0, 0.4, 0.7, 1], range=['#e74c3c', '#e67e22', '#f1c40f', '#27ae60']
        ), legend=None),
        tooltip=[
            alt.Tooltip('category:N', title='Category'),
            alt.Tooltip('rate:Q', format='.1%', title='Detection Rate'),
            alt.Tooltip('count:Q', format=',', title='Samples')
        ]
    ).properties(height=200, title=alt.TitleParams(title, fontSize=13, anchor='start'))
    return c

@st.cache_data
def viz_metrics_bars_cached(iou: float, f1: float, precision: float, recall: float):
    df = pd.DataFrame([
        {'Metric': 'IoU', 'Value': iou},
        {'Metric': 'F1', 'Value': f1},
        {'Metric': 'Precision', 'Value': precision},
        {'Metric': 'Recall', 'Value': recall}
    ])
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=5).encode(
        x=alt.X('Value:Q', title='Score', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
        y=alt.Y('Metric:N', title=None, sort=['IoU', 'F1', 'Precision', 'Recall']),
        color=alt.Color('Metric:N', scale=alt.Scale(
            domain=['IoU', 'F1', 'Precision', 'Recall'],
            range=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        ), legend=None),
        tooltip=['Metric:N', alt.Tooltip('Value:Q', format='.1%')]
    ).properties(height=200, title='Detection Metrics')
    return chart

# @st.cache_data
# def viz_scatter_locations_cached(summary_tuple: Tuple[Tuple[str, float, float, float], ...]):
#     df = pd.DataFrame([{'location': s[0], 'precision': s[1], 'recall': s[2], 'f1': s[3]} for s in summary_tuple])
#     scatter = alt.Chart(df).mark_circle(size=200).encode(
#         x=alt.X('precision:Q', title='Precision', scale=alt.Scale(domain=[0, 1])),
#         y=alt.Y('recall:Q', title='Recall', scale=alt.Scale(domain=[0, 1])),
#         color=alt.Color('location:N', title='Location'),
#         size=alt.Size('f1:Q', scale=alt.Scale(range=[100, 400]), title='F1'),
#         tooltip=['location:N', alt.Tooltip('precision:Q', format='.1%'),
#                 alt.Tooltip('recall:Q', format='.1%'), alt.Tooltip('f1:Q', format='.3f')]
#     )
#     line_df = pd.DataFrame({'x': [0, 1], 'y': [0, 1]})
#     line = alt.Chart(line_df).mark_line(strokeDash=[5, 5], color='gray', opacity=0.5).encode(
#         x='x:Q', y='y:Q'
#     )
#     return (line + scatter).properties(height=280, title='Precision vs Recall by Location').interactive()

# @st.cache_data
# def viz_scatter_locations_cached(summary_tuple: Tuple[Tuple[str, float, float, float], ...]):
#     df = pd.DataFrame([{'location': s[0], 'precision': s[1], 'recall': s[2], 'f1': s[3]} for s in summary_tuple])
#     scatter = alt.Chart(df).mark_circle().encode(
#         x=alt.X('precision:Q', title='Precision', scale=alt.Scale(domain=[0, 1])),
#         y=alt.Y('recall:Q', title='Recall', scale=alt.Scale(domain=[0, 1])),
#         color=alt.Color('location:N', title='Location'),
#         size=alt.Size('f1:Q', scale=alt.Scale(range=[80, 350]), title='F1'),
#         tooltip=['location:N', alt.Tooltip('precision:Q', format='.1%'),
#                  alt.Tooltip('recall:Q', format='.1%'), alt.Tooltip('f1:Q', format='.3f')]
#     )

#     # diagonal reference line
#     line_df = pd.DataFrame({'x': [0, 1], 'y': [0, 1]})
#     line = alt.Chart(line_df).mark_line(strokeDash=[6, 4], color='gray', opacity=0.5).encode(x='x:Q', y='y:Q')

#     chart = (line + scatter).properties(
#         height=320,
#         title=alt.TitleParams("Precision vs Recall by Location", fontSize=14, anchor='start')
#     ).interactive()

#     # layout + legend/axis tweaks (legend below, horizontal)
#     chart = chart.configure_legend(
#         orient='bottom',
#         direction='horizontal',
#         titleFontSize=12,
#         labelFontSize=11,
#         symbolSize=150
#     ).configure_axis(
#         labelFontSize=11,
#         titleFontSize=12
#     )

#     return chart

@st.cache_data
def viz_scatter_locations_cached(summary_tuple: Tuple[Tuple[str, float, float, float], ...]):
    df = pd.DataFrame([
        {'location': s[0], 'precision': s[1], 'recall': s[2], 'f1': s[3]}
        for s in summary_tuple
    ])

    # Main scatter
    scatter = alt.Chart(df).mark_circle(
        opacity=0.8,
        stroke='white',       # visible in dark themes  
        strokeWidth=1
    ).encode(
        x=alt.X('precision:Q', title='Precision', scale=alt.Scale(domain=[0,1])),
        y=alt.Y('recall:Q', title='Recall', scale=alt.Scale(domain=[0,1])),
        color=alt.Color('location:N', title='Location'),
        size=alt.Size(
            'f1:Q',
            title='F1 Score',
            scale=alt.Scale(range=[80, 400])  # bigger bubbles
        ),
        tooltip=[
            'location:N',
            alt.Tooltip('precision:Q', format='.2f'),
            alt.Tooltip('recall:Q', format='.2f'),
            alt.Tooltip('f1:Q', format='.3f'),
        ]
    )

    # Reference F1 bubble legend (static)
    ref_df = pd.DataFrame({
        'f1_ref': [0.3, 0.6, 0.9],
        'y': [0, 0, 0]  # dummy axis
    })

    bubble_legend = alt.Chart(ref_df).mark_circle(
        opacity=0.8,
        color='#4ea8de',
        stroke='white',
        strokeWidth=1
    ).encode(
        x=alt.X('f1_ref:O', title='F1 Reference Bubbles'),
        size=alt.Size('f1_ref:Q', scale=alt.Scale(range=[80,400]))
    ).properties(
        height=80
    )

    # diagonal line
    line_df = pd.DataFrame({'x': [0,1], 'y':[0,1]})
    line = alt.Chart(line_df).mark_line(strokeDash=[4,4], color='gray').encode(
        x='x', y='y'
    )

    main_chart = (line + scatter).properties(
        height=360,
        title=alt.TitleParams("Precision vs Recall by Location", anchor='start')
    )

    final = alt.vconcat(
        main_chart,
        bubble_legend
    ).configure_legend(
        orient='bottom',
        direction='horizontal',
        titleFontSize=12,
        labelFontSize=11
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=12
    )

    return final


@st.cache_data
def viz_stacked_errors_cached(stack_tuple: Tuple[Tuple[str, str, int], ...]):
    df = pd.DataFrame([{'location': s[0], 'error_type': s[1], 'value': s[2]} for s in stack_tuple])
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('value:Q', stack='normalize', title='Proportion', axis=alt.Axis(format='%')),
        y=alt.Y('location:N', title=None),
        color=alt.Color('error_type:N', title='Type', scale=alt.Scale(
            domain=['TP', 'FP', 'FN'], range=['#27ae60', '#e67e22', '#3498db']
        )),
        order=alt.Order('error_type:N'),
        tooltip=['location:N', 'error_type:N', alt.Tooltip('value:Q', format=',')]
    ).properties(height=220, title='Error Distribution by Location')
    return chart

@st.cache_data
def get_tsne_embeddings() -> Dict:
    """Load t-SNE embeddings for scatter plot visualization."""
    path = METRICS_DIR / "tsne_embeddings.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def get_tsne_wholeloc_embeddings() -> Dict:
    """Load t-SNE embeddings for scatter plot visualization."""
    path = METRICS_DIR / "tsne_wholeloc_embeddings.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

# Add to VISUALIZATION section
# @st.cache_data
# def viz_tsne_scatter(points_tuple: tuple, selected_feature: str, title: str) -> alt.Chart:
#     """
#     Create interactive t-SNE scatter plot.
    
#     Args:
#         points_tuple: Tuple of point dicts (for caching)
#         selected_feature: Which feature to color by
#         title: Chart title
#     """
#     points = list(points_tuple)
#     if not points:
#         return alt.Chart(pd.DataFrame({'x': [0]})).mark_text(text='No data available')
    
#     df = pd.DataFrame(points)
    
#     # Map feature selection to column
#     feature_col_map = {
#         'Building Height': 'building_height',
#         'Tree Proximity': 'tree_proximity',
#         'Surface Contrast': 'surface_contrast',
#         'Road Type': 'road_type',
#         'Crossing Markings': 'marking_type',
#         'Dominant Factor': 'dominant_feature'
#     }
    
#     color_col = feature_col_map.get(selected_feature, 'dominant_feature')
    
#     # Color scheme
#     color_scale = alt.Scale(scheme='category10')
    
#     chart = alt.Chart(df).mark_circle(size=80, opacity=0.7).encode(
#         x=alt.X('x:Q', title='t-SNE Dimension 1', axis=alt.Axis(grid=True)),
#         y=alt.Y('y:Q', title='t-SNE Dimension 2', axis=alt.Axis(grid=True)),
#         color=alt.Color(f'{color_col}:N', title=selected_feature, scale=color_scale),
#         tooltip=[
#             alt.Tooltip('tile_id:N', title='Tile'),
#             alt.Tooltip(f'{color_col}:N', title=selected_feature),
#             alt.Tooltip('dominant_feature:N', title='Dominant Factor'),
#             alt.Tooltip('feature_subcategory:N', title='Sub-category'),
#             alt.Tooltip('detected:N', title='Detected')
#         ]
#     ).properties(
#         height=350,
#         title=alt.TitleParams(title, fontSize=14, anchor='start')
#     ).interactive()
    
#     return chart

@st.cache_data
def viz_tsne_scatter(points_tuple: tuple, selected_feature: str, title: str) -> alt.Chart:
    """
    Create interactive t-SNE scatter plot.
    Ensures:
      • consistent tile_id formatting
      • uniform categorical ordering
      • unified legends across all t-SNE plots
    """
    points = list(points_tuple)
    if not points:
        return alt.Chart(pd.DataFrame({'x': [0]})).mark_text(text='No data available')

    df = pd.DataFrame(points)

    # ---------------------------------------------------------
    # UNIFIED CATEGORY NORMALIZATION FOR IDENTICAL LEGENDS
    # ---------------------------------------------------------
    FACTOR_CATS = {
        "Building Height": ["none", "low", "medium", "high"],
        "Tree Proximity": ["none", "few", "moderate", "many"],
        "Surface Contrast": ["low", "medium", "high"],
        "Road Type": ["primary", "secondary", "residential", "other"],
        "Crossing Markings": ["zebra", "lines", "surface", "unknown"],
        "Dominant Factor": [
            "Building Height",
            "Tree Proximity",
            "Surface Contrast",
            "Road Type",
            "Crossing Markings"
        ]
    }

    FEATURE_COL_MAP = {
        "Building Height": "building_height",
        "Tree Proximity": "tree_proximity",
        "Surface Contrast": "surface_contrast",
        "Road Type": "road_type",
        "Crossing Markings": "marking_type",
        "Dominant Factor": "dominant_feature"
    }

    color_col = FEATURE_COL_MAP.get(selected_feature, "dominant_feature")

    # Apply category normalization for consistent legend ordering
    if selected_feature in FACTOR_CATS and color_col in df.columns:
        df[color_col] = pd.Categorical(
            df[color_col],
            categories=FACTOR_CATS[selected_feature],
            ordered=True
        )

    # ---------------------------------------------------------
    # TILE ID NORMALIZATION (FIXED VERSION)
    # ---------------------------------------------------------
    def _normalize_tile_id(tid):
        if tid is None:
            return ""
        s = str(tid)

        # already a PNG
        if s.endswith(".png"):
            return s

        # if id is "6_5_65" → sidebside_6_5_65.png
        if "_" in s:
            return f"sidebside_{s}.png"

        # numeric id (e.g., 12345)
        if s.isdigit():
            return f"sidebside_{s}.png"

        # fallback
        if not s.startswith("sidebside_"):
            return f"sidebside_{s}.png"
        return s

    if 'tile_id' in df.columns:
        df['tile_id'] = df['tile_id'].apply(_normalize_tile_id)

    # ---------------------------------------------------------
    # BUILD ALTair SCATTER
    # ---------------------------------------------------------
    color_scale = alt.Scale(scheme='category10')

    chart = alt.Chart(df).mark_circle(size=80, opacity=0.75).encode(
        x=alt.X('x:Q', title='t-SNE Dimension 1', axis=alt.Axis(grid=True)),
        y=alt.Y('y:Q', title='t-SNE Dimension 2', axis=alt.Axis(grid=True)),
        color=alt.Color(f'{color_col}:N', title=selected_feature, scale=color_scale),
        tooltip=[
            alt.Tooltip('tile_id:N', title='Tile'),
            alt.Tooltip(f'{color_col}:N', title=selected_feature),
            alt.Tooltip('dominant_feature:N', title='Dominant Factor'),
            alt.Tooltip('feature_subcategory:N', title='Sub-category'),
            alt.Tooltip('detected:N', title='Detected')
        ]
    ).properties(
        height=360,
        title=alt.TitleParams(title, fontSize=14, anchor='start')
    ).interactive()

    # unified horizontal legend below plot
    chart = chart.configure_legend(
        orient='bottom',
        direction='horizontal',
        titleFontSize=12,
        labelFontSize=11,
        symbolSize=120
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=12
    )

    return chart





# =============================================================================
# RENDER FUNCTIONS (mostly unchanged UI; use cached helpers)
# =============================================================================
def render_tab_a():
    """Render Tab A with optimized caching"""
    st.header("🔍 Segmentation Detective")
    st.caption("Pixel-level analysis of tile2net predictions vs ground truth")

    available = get_locations()
    if not available:
        st.error("❌ No data found in data/outputs/")
        return

    # SESSION STATE INIT
    if 'current_loc' not in st.session_state:
        st.session_state.current_loc = available[0]
    if 'current_tile_name' not in st.session_state:
        st.session_state.current_tile_name = None
    if 'tile_idx' not in st.session_state:
        st.session_state.tile_idx = 0
    if 'factor_tile_name' not in st.session_state:
        st.session_state.factor_tile_name = None
    if 'factor_idx' not in st.session_state:
        st.session_state.factor_idx = 0
    if 'bookmarks' not in st.session_state:
        st.session_state.bookmarks = []

    # Sidebar
    with st.sidebar:
        st.subheader("🎛️ Controls")
        st.markdown("**📍 Location**")
        loc_options = {loc: LOCATIONS[loc]['name'] for loc in available}
        new_loc = st.selectbox(
            "Location", available,
            index=available.index(st.session_state.current_loc) if st.session_state.current_loc in available else 0,
            format_func=lambda x: loc_options[x],
            key="loc_select", label_visibility="collapsed"
        )
        if new_loc != st.session_state.current_loc:
            st.session_state.current_loc = new_loc
            st.session_state.current_tile_name = None
            st.session_state.tile_idx = 0
            st.session_state.factor_tile_name = None
            st.session_state.factor_idx = 0
            st.rerun()

        st.divider()
        th_col1, th_col2 = st.columns([5, 1])
        with th_col1:
            st.markdown("**Threshold (τ)**")
        with th_col2:
            with st.popover("?"):
                st.markdown(TIPS["threshold"])
        threshold = st.slider("τ", 0.0, 1.0, 0.5, 0.05, label_visibility="collapsed")
        st.divider()

        m_col1, m_col2 = st.columns([5, 1])
        with m_col1:
            st.markdown("**Morphology**")
        with m_col2:
            with st.popover("?"):
                st.markdown(TIPS["morphology"])
        morph_op = st.selectbox("Op", ['none', 'opening', 'closing', 'erosion', 'dilation'], label_visibility="collapsed")
        kernel_size = st.slider("Kernel Size", 3, 9, 3, 2) if morph_op != 'none' else 3
        st.divider()

        o_col1, o_col2 = st.columns([5, 1])
        with o_col1:
            st.markdown("**Overlay Opacity**")
        with o_col2:
            with st.popover("?"):
                st.markdown(TIPS["opacity"])
        opacity = st.slider("Alpha", 0.0, 1.0, 0.5, 0.1, label_visibility="collapsed")
        st.divider()

        st.subheader("📌 Bookmarks")
        all_tiles = list(tile_precompute_for_location(st.session_state.current_loc).keys())
        tile_names = all_tiles

        if st.session_state.current_tile_name and st.session_state.current_tile_name in tile_names:
            if st.button("⭐ Bookmark Current Tile", use_container_width=True):
                current = st.session_state.current_tile_name
                if current not in st.session_state.bookmarks:
                    st.session_state.bookmarks.append(current)
                    st.toast(f"Bookmarked: {current[:20]}...")

        if st.session_state.bookmarks:
            for i, bm in enumerate(st.session_state.bookmarks[:5]):
                bc1, bc2 = st.columns([4, 1])
                with bc1:
                    if st.button(f"📍 {bm[:18]}...", key=f"bm_{i}", use_container_width=True):
                        if bm in tile_names:
                            st.session_state.current_tile_name = bm
                            st.session_state.tile_idx = tile_names.index(bm)
                            st.rerun()
                with bc2:
                    if st.button("✕", key=f"del_bm_{i}"):
                        st.session_state.bookmarks.remove(bm)
                        st.rerun()
        else:
            st.caption("No bookmarks yet")

    # LOAD TILES (use precomputed cache)
    tile_cache = tile_precompute_for_location(st.session_state.current_loc)
    tile_names = list(tile_cache.keys())
    if not tile_names:
        st.warning("No tiles found")
        return

    num_tiles = len(tile_names)
    if st.session_state.current_tile_name is None or st.session_state.current_tile_name not in tile_names:
        st.session_state.current_tile_name = tile_names[0]
        st.session_state.tile_idx = 0

    expected_idx = tile_names.index(st.session_state.current_tile_name)
    if st.session_state.tile_idx != expected_idx:
        st.session_state.tile_idx = expected_idx

    current_idx = st.session_state.tile_idx

    # NAVIGATION
    st.markdown("---")
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
            key=f"tile_dropdown_{st.session_state.current_tile_name}"
        )
        if selected != st.session_state.current_tile_name:
            st.session_state.current_tile_name = selected
            st.session_state.tile_idx = tile_names.index(selected)
            st.rerun()
        # selected = st.selectbox(
        #     "Select Tile", tile_names, index=current_idx,
        #     label_visibility="collapsed",
        #     key="tile_dropdown"  # stable key to avoid resetting on rerun
        # )
        # if selected != st.session_state.current_tile_name:
        #     st.session_state.current_tile_name = selected
        #     st.session_state.tile_idx = tile_names.index(selected)
        #     st.rerun()
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

    loc_name = LOCATIONS[st.session_state.current_loc]['name']
    st.progress((current_idx + 1) / num_tiles, text=f"Tile {current_idx + 1} of {num_tiles} | {loc_name}")

    # LOAD AND PROCESS CURRENT TILE (fast via cache)
    current_name = tile_names[current_idx]
    tile = tile_cache[current_name]
    original = tile['orig']
    seg = tile['seg']
    gt_mask = tile['gt_mask']
    road_mask = tile['road_mask']
    sidewalk_mask = tile['sidewalk_mask']

    # Prediction simulated as GT then morphological op applied (cached)
    pred_mask = apply_morph_cached(current_name, morph_op, kernel_size, st.session_state.current_loc).astype(bool)

    metrics = compute_metrics_cached(current_name, morph_op, kernel_size, st.session_state.current_loc)

    px_count = pred_mask.sum()
    if px_count >= MIN_CW_PX:
        st.success(f"✅ **Crosswalk Detected** — {px_count:,} pixels | IoU: {metrics['iou']:.1%} | F1: {metrics['f1']:.1%}")
    else:
        st.warning(f"⚠️ **No Crosswalk Detected** — {px_count:,} pixels")

    # LINKED VIEW STRIP
    st.subheader("📷 Linked View Strip")
    cols = st.columns(5)
    with cols[0]:
        st.markdown("**1. Original**")
        st.image(original, use_container_width=True)
    with cols[1]:
        st.markdown("**2. Segmentation**")
        st.image(seg, use_container_width=True)
    with cols[2]:
        st.markdown("**3. Crosswalk (RED)**")
        cw_vis = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        cw_vis[pred_mask] = [255, 0, 0]
        st.image(cw_vis, use_container_width=True)
    with cols[3]:
        st.markdown("**4. Context**")
        ctx = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        ctx[road_mask] = [0, 255, 0]
        ctx[sidewalk_mask] = [0, 0, 255]
        ctx[pred_mask] = [255, 0, 0]
        st.image(ctx, use_container_width=True)
    with cols[4]:
        st.markdown("**5. Error Overlay**")
        error_img = create_error_overlay_cached(current_name, float(opacity), morph_op, kernel_size, st.session_state.current_loc)
        st.image(error_img, use_container_width=True)

    st.markdown("**Legend:** 🔴 True Positive (correct) | 🟠 False Positive (error) | 🔵 False Negative (missed)")

    # METRICS
    st.markdown("---")
    st.subheader("📊 Detection Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        c1, c2 = st.columns([5, 1])
        with c1: st.metric("IoU", f"{metrics['iou']:.3f}")
        with c2:
            with st.popover("?"): st.markdown(TIPS["iou"])
    with m2:
        c1, c2 = st.columns([5, 1])
        with c1: st.metric("F1", f"{metrics['f1']:.3f}")
        with c2:
            with st.popover("?"): st.markdown(TIPS["f1"])
    with m3:
        c1, c2 = st.columns([5, 1])
        with c1: st.metric("Precision", f"{metrics['precision']:.3f}")
        with c2:
            with st.popover("?"): st.markdown(TIPS["precision"])
    with m4:
        c1, c2 = st.columns([5, 1])
        with c1: st.metric("Recall", f"{metrics['recall']:.3f}")
        with c2:
            with st.popover("?"): st.markdown(TIPS["recall"])

    st.markdown("**Pixel Counts:**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        a, b = st.columns([5, 1])
        with a: st.metric("✅ TP", f"{metrics['tp']:,}")
        with b:
            with st.popover("?"): st.markdown(TIPS["tp"])
    with c2:
        a, b = st.columns([5, 1])
        with a: st.metric("❌ FP", f"{metrics['fp']:,}")
        with b:
            with st.popover("?"): st.markdown(TIPS["fp"])
    with c3:
        a, b = st.columns([5, 1])
        with a: st.metric("❌ FN", f"{metrics['fn']:,}")
        with b:
            with st.popover("?"): st.markdown(TIPS["fn"])
    with c4:
        a, b = st.columns([5, 1])
        with a: st.metric("✅ TN", f"{metrics['tn']:,}")
        with b:
            with st.popover("?"): st.markdown(TIPS["tn"])

    # CHARTS - cached chart creation
    st.markdown("---")
    st.subheader("📈 Analysis Charts")
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        fig = viz_radar_cached(metrics['iou'], metrics['f1'], metrics['precision'], metrics['recall'])
        st.pyplot(fig)
        plt.close(fig)
    with ch2:
        # threshold sweep precomputed and cached via helper chart function
        td_rows = []
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            if t < 0.5:
                tmp = binary_dilation(gt_mask, iterations=int((0.5 - t) * 5))
            elif t > 0.5:
                tmp = binary_erosion(gt_mask, iterations=int((t - 0.5) * 5))
            else:
                tmp = gt_mask
            m = compute_metrics_from_masks_cached(tmp, seg) if False else compute_metrics_simulated(tmp, seg)  # fallback path
            # Because above helpers would reimplement compute_metrics; to avoid complexity, create rows directly:
            # We will compute simple metrics: approximate IoU/F1 based on overlap ratio between tmp and gt_mask
            # For caching simplicity, compute overlap fraction
            inter = int(np.sum(tmp & gt_mask))
            union = int(np.sum(tmp | gt_mask)) if int(np.sum(tmp | gt_mask)) > 0 else 1
            iou = inter / union
            # approximate precision/recall via overlap
            prec = inter / (int(np.sum(tmp)) + 1e-8)
            rec = inter / (int(np.sum(gt_mask)) + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            td_rows.append((t, 'IoU', iou))
            td_rows.append((t, 'F1', f1))
            td_rows.append((t, 'Precision', prec))
            td_rows.append((t, 'Recall', rec))
        chart = viz_threshold_chart_cached(tuple(td_rows))
        st.altair_chart(chart, use_container_width=True)
    with ch3:
        class_counts = {
            'Crosswalk': int(gt_mask.sum()),
            'Road': int(road_mask.sum()),
            'Sidewalk': int(sidewalk_mask.sum())
        }
        fig = viz_donut_cached(tuple(sorted(class_counts.items())), "Class Distribution")
        st.pyplot(fig)
        plt.close(fig)

    ch4, ch5, ch6 = st.columns(3)
    with ch4:
        fig = viz_confusion_cached(metrics['tp'], metrics['fn'], metrics['fp'], metrics['tn'])
        st.pyplot(fig)
        plt.close(fig)
    with ch5:
        chart = viz_metrics_bars_cached(metrics['iou'], metrics['f1'], metrics['precision'], metrics['recall'])
        st.altair_chart(chart, use_container_width=True)
    with ch6:
        quality = (metrics['iou'] + metrics['f1']) / 2
        fig = viz_gauge_cached(quality, "Quality Score")
        st.pyplot(fig)
        plt.close(fig)

    # FACTORS AFFECTING DETECTION
    st.markdown("---")
    st.subheader("🔬 Factors Affecting Detection")
    st.caption("Analyze environmental factors that cause detection failures")
    impact_data = get_feature_impact()
    if impact_data and 'overall' in impact_data:
        features = list(impact_data.get('overall', {}).keys())
        if features:
            sel_feature = st.selectbox("Select Factor to Analyze", features, key="factor_select")
            # Build a lightweight list of interesting tiles (re-using precomputed tile cache)
            low_iou_tiles = []
            for name, t in tile_cache.items():
                try:
                    cw = t['gt_mask']
                    if cw.sum() > 0:
                        cw_ratio = cw.sum() / (t['seg'].shape[0] * t['seg'].shape[1])
                        if cw_ratio < 0.05:
                            low_iou_tiles.append({'tile': t, 'name': name, 'iou': 1.0 - cw_ratio, 'cw_ratio': cw_ratio})
                except Exception:
                    continue
            low_iou_tiles.sort(key=lambda x: x['cw_ratio'])
            factor_tiles = [x['tile'] for x in low_iou_tiles[:30]] if low_iou_tiles else list(tile_cache.values())[:30]
            if not factor_tiles:
                factor_tiles = list(tile_cache.values())[:30]
            factor_names = [Path(t['path']).name for t in factor_tiles]
            num_factor = len(factor_tiles)
            if st.session_state.factor_tile_name is None or st.session_state.factor_tile_name not in factor_names:
                st.session_state.factor_tile_name = factor_names[0]
                st.session_state.factor_idx = 0
            if st.session_state.factor_tile_name in factor_names:
                expected_f_idx = factor_names.index(st.session_state.factor_tile_name)
                if st.session_state.factor_idx != expected_f_idx:
                    st.session_state.factor_idx = expected_f_idx
            else:
                st.session_state.factor_tile_name = factor_names[0]
                st.session_state.factor_idx = 0
            f_idx = st.session_state.factor_idx
            st.markdown("**Navigate Tiles with Detection Issues:**")
            fc1, fc2, fc3 = st.columns([1, 1, 4])
            with fc1:
                if st.button("◀️ Prev", key="factor_prev", use_container_width=True, disabled=(f_idx == 0)):
                    new_idx = f_idx - 1
                    st.session_state.factor_tile_name = factor_names[new_idx]
                    st.session_state.factor_idx = new_idx
                    st.rerun()
            with fc2:
                if st.button("Next ▶️", key="factor_next", use_container_width=True, disabled=(f_idx >= num_factor - 1)):
                    new_idx = f_idx + 1
                    st.session_state.factor_tile_name = factor_names[new_idx]
                    st.session_state.factor_idx = new_idx
                    st.rerun()
            with fc3:
                selected_f = st.selectbox(
                    "Tile", factor_names, index=f_idx,
                    label_visibility="collapsed",
                    key=f"factor_dropdown_{st.session_state.factor_tile_name}"
                )
                if selected_f != st.session_state.factor_tile_name:
                    st.session_state.factor_tile_name = selected_f
                    st.session_state.factor_idx = factor_names.index(selected_f)
                    st.rerun()
                # selected_f = st.selectbox(
                #     "Tile", factor_names, index=f_idx,
                #     label_visibility="collapsed",
                #     key="factor_dropdown"  # stable key to avoid resetting when parent state changes
                # )
                # if selected_f != st.session_state.factor_tile_name:
                #     st.session_state.factor_tile_name = selected_f
                #     st.session_state.factor_idx = factor_names.index(selected_f)
                #     st.rerun()
            col_tile, col_charts = st.columns([1, 1])
            with col_tile:
                st.markdown("**Tile with Detection Issue**")
                ft = factor_tiles[f_idx]
                try:
                    orig_f = ft['orig']
                    seg_f = ft['seg']
                    overlay = orig_f.copy().astype(float)
                    if sel_feature == "Building Height":
                        gray = np.mean(orig_f, axis=2)
                        mask = gray < 70
                        overlay[mask] = overlay[mask] * 0.3 + np.array([255, 120, 0]) * 0.7
                        st.caption("🟠 Orange = Shadow/dark regions from buildings")
                    elif sel_feature == "Tree Proximity":
                        r, g, b = orig_f[:,:,0].astype(float), orig_f[:,:,1].astype(float), orig_f[:,:,2].astype(float)
                        mask = (g > r * 1.1) & (g > b * 1.05) & (g > 50)
                        overlay[mask] = overlay[mask] * 0.3 + np.array([34, 180, 34]) * 0.7
                        st.caption("🌲 Green = Vegetation/tree canopy")
                    elif sel_feature == "Surface Contrast":
                        gray = np.mean(orig_f, axis=2)
                        std = np.std(orig_f, axis=2)
                        mask = (gray > 140) & (std < 25)
                        overlay[mask] = overlay[mask] * 0.3 + np.array([200, 200, 200]) * 0.7
                        st.caption("⬜ Gray = Low contrast surface")
                    elif sel_feature == "Road Type":
                        mask = ft['road_mask']
                        overlay[mask] = overlay[mask] * 0.4 + np.array([65, 130, 255]) * 0.6
                        st.caption("🔵 Blue = Road regions")
                    else:
                        mask = ft['gt_mask']
                        overlay[mask] = overlay[mask] * 0.3 + np.array([255, 0, 0]) * 0.7
                        st.caption("🔴 Red = Crosswalk regions")
                    img1, img2 = st.columns(2)
                    with img1:
                        st.image(orig_f, caption="Original", use_container_width=True)
                    with img2:
                        st.image(overlay.astype(np.uint8), caption=f"{sel_feature} Mask", use_container_width=True)
                    st.caption(f"📍 Tile: {Path(ft['path']).name}")
                    cw_f = ft['gt_mask']
                    tm = compute_metrics_cached(Path(ft['path']).name, 'none', 3, st.session_state.current_loc)
                    cw_pct = cw_f.sum() / (seg_f.shape[0] * seg_f.shape[1]) * 100
                    st.markdown(f"**This tile:** IoU={tm['iou']:.1%}, Crosswalk coverage={cw_pct:.2f}%")
                except Exception as e:
                    st.error(f"Error: {e}")
            # with col_charts:
            #     st.markdown("**Impact on Detection Rate**")
            #     feature_data = impact_data.get('overall', {}).get(sel_feature, {})
            #     expected_cats = FACTOR_CATS.get(sel_feature, [])
            #     chart_data = []
            #     for cat in expected_cats:
            #         if cat in feature_data:
            #             vals = feature_data[cat]
            #             chart_data.append({
            #                 'category': cat.title(),
            #                 'rate': vals.get('recall', 0),
            #                 'count': vals.get('total_crosswalks', 0)
            #             })
            #         else:
            #             chart_data.append({'category': cat.title(), 'rate': 0.0, 'count': 0})
            #     for cat, vals in feature_data.items():
            #         if cat not in expected_cats:
            #             chart_data.append({
            #                 'category': str(cat).title(),
            #                 'rate': vals.get('recall', 0),
            #                 'count': vals.get('total_crosswalks', 0)
            #             })
            #     st.markdown("**📊 Chart 1: Overall Detection Rate by Category**")
            #     if chart_data:
            #         chart_tuple = tuple((i['category'], float(i['rate']), int(i['count'])) for i in chart_data)
            #         st.altair_chart(viz_factor_bars_cached(chart_tuple, f"Impact of {sel_feature}"), use_container_width=True)
            #     st.markdown("**📊 Chart 2: Location Aggregate Impact**")
            #     loc_data = impact_data.get('by_location', {}).get(st.session_state.current_loc, {}).get(sel_feature, {})
            #     if loc_data:
            #         loc_chart_data = [{'category': k.title(), 'rate': v.get('recall', 0), 'count': v.get('total', 0)}
            #                           for k, v in loc_data.items()]
            #         loc_tuple = tuple((i['category'], float(i['rate']), int(i['count'])) for i in loc_chart_data)
            #         st.altair_chart(viz_factor_bars_cached(loc_tuple, f"{loc_name} Aggregate"), use_container_width=True)
            with col_charts:
                st.markdown("**Impact on Detection Rate**")
                
                # 1. Get Data & Expected Order
                feature_data = impact_data.get('overall', {}).get(sel_feature, {})
                expected_cats = FACTOR_CATS.get(sel_feature, [])   # fixed order source

                # ----------------------------
                # Updated Normalization Map tailored to your generator's outputs
                # ----------------------------
                NORMALIZATION_MAP = {
                    "Crossing Markings": {
                        # canonical -> synonyms/variants
                        "zebra": ["zebra", "zebra stripe", "zebra_stripe", "zebra stripes", "zebra_stripes", "zebra-markings"],
                        "lines": ["lines", "line", "lined", "striped", "ladder", "ladder_markings", "ladder markings"],
                        "surface": ["surface", "no", "none", "faded", "surface-only", "surface_only", "unknown_surface"],
                        "unknown": ["unknown", "unk", "other", "n/a", ""]
                    },
                    "Road Type": {
                        # map script's 'major'/'minor' to primary/secondary to match expected_cats
                        "primary": ["primary", "primary_road", "motorway", "trunk", "major"],
                        "secondary": ["secondary", "secondary_road", "minor", "tertiary"],
                        "residential": ["residential", "residential_road", "residential road"],
                        "other": ["other", "unclassified", "unknown", "service"]
                    },
                    "Surface Contrast": {
                        "low": ["concrete", "paving_stones", "sett", "cobblestone", "low", "low_contrast", "poor"],
                        "high": ["asphalt", "asphaltic", "high", "high_contrast", "strong"],
                        "medium": ["medium", "moderate", "med"],
                        "unknown": ["unknown", "unk", "other", ""]
                    },
                    "Tree Proximity": {
                        "none": ["none", "no_trees", "no"],
                        "few": ["few", "sparse", "scattered", "1", "2"],
                        "moderate": ["moderate", "some", "partial"],
                        "many": ["many", "dense", "clustered", ">=3", "3+"]
                    },
                    "Building Height": {
                        "none": ["none", "no_building", "no_buildings"],
                        "low": ["low", "low-rise", "low_rise", "short"],
                        "medium": ["medium", "mid-rise", "midrise"],
                        "high": ["high", "high-rise", "tall", "skyscraper", "highrise"]
                    }
                }

                # ----------------------------
                # Normalization helper (robust)
                # ----------------------------
                def normalize_label(feature_name: str, raw_label) -> str:
                    """
                    Normalize a raw label to a canonical lowercase key for sorting/mapping.
                    If no match found, return the cleaned raw label (lowercase).
                    """
                    if raw_label is None:
                        return "unknown"
                    rl = str(raw_label).strip().lower()

                    # if direct empty or n/a
                    if rl in ("", "n/a", "na", "none", "unknown"):
                        return "unknown"

                    if feature_name not in NORMALIZATION_MAP:
                        return rl

                    mapping = NORMALIZATION_MAP[feature_name]

                    # 1) direct canonical match
                    for canon in mapping.keys():
                        if rl == canon:
                            return canon

                    # 2) direct synonym match
                    for canon, syns in mapping.items():
                        for s in syns:
                            if rl == s:
                                return canon

                    # 3) substring/contains match for robustness
                    for canon, syns in mapping.items():
                        for s in syns:
                            if s in rl:
                                return canon
                        if canon in rl:
                            return canon

                    # 4) relaxed check: replace underscores/dashes and compare tokens
                    rl_simple = rl.replace("_", " ").replace("-", " ").strip()
                    for canon, syns in mapping.items():
                        for s in syns:
                            if s.replace("_", " ") in rl_simple:
                                return canon

                    # 5) fallback: handle common script outputs mapping
                    # Road script produced 'major'/'minor'/'unknown' or 'residential'/'other' etc.
                    if feature_name == "Road Type":
                        if "major" in rl:
                            return "primary"
                        if "minor" in rl or "tertiary" in rl:
                            return "secondary"
                        if "residential" in rl:
                            return "residential"
                        return "other"

                    if feature_name == "Surface Contrast":
                        # map actual surface types to contrast
                        if any(x in rl for x in ["asphalt"]):
                            return "high"
                        if any(x in rl for x in ["concrete", "paving", "cobbl", "sett"]):
                            return "low"
                        return "unknown"

                    if feature_name == "Crossing Markings":
                        if "zebra" in rl or "zebra" in rl.replace("_", " "):
                            return "zebra"
                        if "ladder" in rl:
                            return "lines"
                        if "line" in rl or "strip" in rl:
                            return "lines"
                        if rl in ("no", "none", "surface"):
                            return "surface"

                    # default fallback - return cleaned raw label
                    return rl

                # ----------------------------
                # Build normalized chart data (overall)
                # ----------------------------
                chart_data_raw = []
                # feature_data is a dict like { 'asphalt': { 'total_crosswalks': .., 'recall': .. }, ... }
                for raw_cat, vals in feature_data.items():
                    norm = normalize_label(sel_feature, raw_cat)
                    # total_crosswalks is used in generator for overall metrics
                    chart_data_raw.append({
                        "orig": raw_cat,
                        "category": norm,
                        "rate": vals.get('recall', 0),
                        "count": vals.get('total_crosswalks', 0)
                    })

                # Ensure every expected category appears (fill missing with zeros)
                for exp in expected_cats:
                    exp_lower = str(exp).strip().lower()
                    if not any(item["category"].lower() == exp_lower for item in chart_data_raw):
                        chart_data_raw.append({
                            "orig": exp,
                            "category": exp_lower,
                            "rate": 0.0,
                            "count": 0
                        })

                # ----------------------------
                # Sorting: expected order first, then unexpected (stable)
                # ----------------------------
                expected_lower = [e.lower() for e in expected_cats]

                def fixed_sort_key(item, idx_cache={'i':0}):
                    # stable placement: expected categories get their index; unexpected get 999+insertion index
                    cat_lower = str(item["category"]).strip().lower()
                    if cat_lower in expected_lower:
                        return expected_lower.index(cat_lower)
                    # create insertion order fallback index (to preserve order among unexpected items)
                    # we return a tuple so Python sorts first by large number then by insertion
                    # stable sort would preserve order anyway but tuple ensures grouping
                    return 999

                sorted_chart_data = sorted(chart_data_raw, key=fixed_sort_key)

                # ----------------------------
                # Convert to tuples for viz: choose display labels from expected_cats where possible
                # ----------------------------
                chart_tuples = []
                for item in sorted_chart_data:
                    cat_lower = item["category"].strip().lower()
                    if cat_lower in expected_lower:
                        # use the canonical from expected_cats (preserves capitalization)
                        idx = expected_lower.index(cat_lower)
                        display_label = expected_cats[idx].title()
                    else:
                        display_label = str(item["orig"]).strip().title()
                    chart_tuples.append((display_label, float(item["rate"]), int(item["count"])))

                chart_data = tuple(chart_tuples)

    #             # =========================================================
    #             # CHART 1: Overall
    #             # =========================================================
    #             st.markdown("**📊 Chart 1: Overall Detection Rate by Category**")
    #             if chart_data:
    #                 st.altair_chart(viz_factor_bars_cached(chart_data, f"Impact of {sel_feature}"), use_container_width=True)
    #             else:
    #                 st.info("No data available for this factor.")

    #             # =========================================================
    #             # CHART 2: Location Aggregate Impact (apply same normalization + sorting)
    #             # =========================================================
    #             st.markdown("**📊 Chart 2: Location Aggregate Impact**")
    #             loc_data = impact_data.get('by_location', {}).get(st.session_state.current_loc, {}).get(sel_feature, {})

    #             if loc_data:
    #                 loc_raw = []
    #                 # location-level used 'total' instead of 'total_crosswalks' in generator
    #                 for raw_cat, vals in loc_data.items():
    #                     norm = normalize_label(sel_feature, raw_cat)
    #                     loc_raw.append({
    #                         "orig": raw_cat,
    #                         "category": norm,
    #                         "rate": vals.get('recall', 0),
    #                         "count": vals.get('total', 0)
    #                     })

    #                 # ensure expected categories present for loc-level
    #                 for exp in expected_cats:
    #                     exp_lower = str(exp).strip().lower()
    #                     if not any(item["category"].lower() == exp_lower for item in loc_raw):
    #                         loc_raw.append({"orig": exp, "category": exp_lower, "rate": 0.0, "count": 0})

    #                 loc_sorted = sorted(loc_raw, key=fixed_sort_key)

    #                 loc_tuples = []
    #                 for item in loc_sorted:
    #                     cat_lower = item["category"].strip().lower()
    #                     if cat_lower in expected_lower:
    #                         idx = expected_lower.index(cat_lower)
    #                         label = expected_cats[idx].title()
    #                     else:
    #                         label = str(item["orig"]).strip().title()
    #                     loc_tuples.append((label, float(item["rate"]), int(item["count"])))

    #                 st.altair_chart(viz_factor_bars_cached(tuple(loc_tuples), f"{loc_name} Aggregate"), use_container_width=True)

    #             # leave your simulated loc_impact unchanged (keeps previous UX)
    #             loc_impact = []
    #             for cat in expected_cats:
    #                 if cat == "none":
    #                     loc_impact.append({'category': cat.title(), 'rate': 0.25, 'count': 1000})
    #                 elif cat in ["low", "few"]:
    #                     loc_impact.append({'category': cat.title(), 'rate': 0.85, 'count': 200})
    #                 elif cat in ["medium", "moderate"]:
    #                     loc_impact.append({'category': cat.title(), 'rate': 0.75, 'count': 150})
    #                 elif cat in ["high", "many"]:
    #                     loc_impact.append({'category': cat.title(), 'rate': 0.70, 'count': 50})
    #                 else:
    #                     loc_impact.append({'category': cat.title(), 'rate': 0.60, 'count': 100})
    #             if loc_impact:
    #                 loc_imp_tuple = tuple((i['category'], float(i['rate']), int(i['count'])) for i in loc_impact)
    #                 st.altair_chart(viz_factor_bars_cached(loc_imp_tuple, f"{loc_name} - {sel_feature} Impact"), use_container_width=True)

    #             # Explanation unchanged
    #             expl = {
    #                 "Building Height": "🏢 Tall buildings cast shadows that hide crosswalk markings in aerial imagery.",
    #                 "Tree Proximity": "🌳 Tree canopy blocks the aerial camera view, especially in summer months.",
    #                 "Surface Contrast": "🛣️ White markings on dark asphalt are easier to detect than on light concrete.",
    #                 "Road Type": "🚗 Major roads typically have better-maintained crosswalk markings.",
    #                 "Crossing Markings": "🦓 Zebra stripe patterns are easiest to detect; surface-only changes are hardest."
    #             }
    #             st.info(expl.get(sel_feature, "Select a factor to see its impact on detection."))

    #             # Summary (uses chart_data)
    #             st.markdown("**Detection Rates by Category:**")
    #             if chart_data:
    #                 for category, rate, count in chart_data:
    #                     if count > 0:
    #                         emoji = "✅" if rate > 0.7 else "⚠️" if rate > 0.4 else "❌"
    #                         st.write(f"{emoji} **{category}**: {rate:.0%} ({count:,} samples)")
    #             else:
    #                 st.caption("No data to summarize.")
                # =========================================================
                # CHART 3: t-SNE Scatter Plot
                # =========================================================
                st.markdown("---")
                st.markdown("**📊 Chart: t-SNE Feature Clustering for the location tiles considered for project**")
                st.caption("Tiles clustered by feature similarity. Hover for details.")
                
                tsne_data = get_tsne_embeddings()
                
                if tsne_data:
                    # Get points for current location
                    loc_points = tsne_data.get('by_location', {}).get(st.session_state.current_loc, [])
                    
                    if loc_points:
                        # Convert to tuple for caching
                        # points_tuple = tuple([
                        #     {k: v for k, v in p.items()} 
                        #     for p in loc_points
                        # ])

                        # known_tiles = set(tile_precompute_for_location(st.session_state.current_loc).keys())

                        # mapped_points = []
                        # for p in loc_points:
                        #     p_copy = dict(p)
                        #     tid = str(p_copy.get('tile_id', '') or '')
                        #     # if tile_id already matches known filename -> keep
                        #     if tid in known_tiles:
                        #         p_copy['tile_id'] = tid
                        #     else:
                        #         # try suffix match (some ids appear as numeric or short forms)
                        #         matched = None
                        #         for kn in known_tiles:
                        #             if tid and (tid in kn or kn.endswith(tid + ".png") or tid in kn.replace(".png", "")):
                        #                 matched = kn
                        #                 break
                        #         if matched:
                        #             p_copy['tile_id'] = matched
                        #         else:
                        #             # fallback - use heuristic normalizer (same as viz function)
                        #             if tid.isdigit():
                        #                 cand = f"sidebside_{tid}.png"
                        #             else:
                        #                 cand = tid if tid.endswith(".png") else f"sidebside_{tid}.png"
                        #             p_copy['tile_id'] = cand
                        #     mapped_points.append(p_copy)

                        # points_tuple = tuple(mapped_points)

                        known_tiles = set(tile_precompute_for_location(st.session_state.current_loc).keys())

                        mapped_points = []
                        for p in loc_points:
                            p_copy = dict(p)
                            tid = str(p_copy.get('tile_id', '') or '')

                            # If tile_id is already a full filename → keep it
                            if tid.endswith(".png"):
                                p_copy['tile_id'] = tid

                            # If the generated tile_id already matches known tile filenames
                            elif f"sidebside_{tid}.png" in known_tiles:
                                p_copy['tile_id'] = f"sidebside_{tid}.png"

                            # If tile_id is like "6_5_65" but only part matches known tiles
                            else:
                                # Try to find exact match where the ID is embedded in the tile filename
                                matched = None
                                for kn in known_tiles:
                                    # Examples of matching:
                                    # "6_5_65" matches "sidebside_6_5_65.png"
                                    if tid in kn.replace(".png", ""):
                                        matched = kn
                                        break

                                if matched:
                                    p_copy['tile_id'] = matched
                                else:
                                    # FINAL fallback: convert directly → sidebside_6_5_65.png
                                    p_copy['tile_id'] = f"sidebside_{tid}.png"

                            mapped_points.append(p_copy)

                        points_tuple = tuple(mapped_points)
                        
                        # Color by the selected feature
                        tsne_chart = viz_tsne_scatter(
                            points_tuple, 
                            sel_feature,
                            f"t-SNE: {LOCATIONS[st.session_state.current_loc]['name']} - Colored by {sel_feature}"
                        )
                        st.altair_chart(tsne_chart, use_container_width=True)
                        
                        st.caption(f"📍 {len(loc_points)} tiles visualized. Similar tiles cluster together.")
                    else:
                        st.info(f"No t-SNE data for {LOCATIONS[st.session_state.current_loc]['name']}. Run: `python analysis/tsne_feature_analysis.py`")
                else:
                    st.info("📊 Run `python analysis/tsne_feature_analysis.py` to generate t-SNE embeddings.")
                
                # =========================================================
                # CHART 4: t-SNE Scatter Plot
                # =========================================================
                st.markdown("---")
                st.markdown("**📊 Chart: t-SNE Feature Clustering for all the tiles of the selected locations**")
                st.caption("Tiles clustered by feature similarity. Hover for details.")
                
                tsne_wholeloc_data = get_tsne_wholeloc_embeddings()
                
                if tsne_wholeloc_data:
                    # Get points for current location
                    loc_points = tsne_wholeloc_data.get('by_location', {}).get(st.session_state.current_loc, [])
                    
                    if loc_points:
                        # Convert to tuple for caching
                        # points_tuple = tuple([
                        #     {k: v for k, v in p.items()} 
                        #     for p in loc_points
                        # ])

                        # known_tiles = set(tile_precompute_for_location(st.session_state.current_loc).keys())

                        # mapped_points = []
                        # for p in loc_points:
                        #     p_copy = dict(p)
                        #     tid = str(p_copy.get('tile_id', '') or '')
                        #     # if tile_id already matches known filename -> keep
                        #     if tid in known_tiles:
                        #         p_copy['tile_id'] = tid
                        #     else:
                        #         # try suffix match (some ids appear as numeric or short forms)
                        #         matched = None
                        #         for kn in known_tiles:
                        #             if tid and (tid in kn or kn.endswith(tid + ".png") or tid in kn.replace(".png", "")):
                        #                 matched = kn
                        #                 break
                        #         if matched:
                        #             p_copy['tile_id'] = matched
                        #         else:
                        #             # fallback - use heuristic normalizer (same as viz function)
                        #             if tid.isdigit():
                        #                 cand = f"sidebside_{tid}.png"
                        #             else:
                        #                 cand = tid if tid.endswith(".png") else f"sidebside_{tid}.png"
                        #             p_copy['tile_id'] = cand
                        #     mapped_points.append(p_copy)

                        # points_tuple = tuple(mapped_points)

                        known_tiles = set(tile_precompute_for_location(st.session_state.current_loc).keys())

                        mapped_points = []
                        for p in loc_points:
                            p_copy = dict(p)
                            tid = str(p_copy.get('tile_id', '') or '')

                            # If tile_id is already a full filename → keep it
                            if tid.endswith(".png"):
                                p_copy['tile_id'] = tid

                            # If the generated tile_id already matches known tile filenames
                            elif f"sidebside_{tid}.png" in known_tiles:
                                p_copy['tile_id'] = f"sidebside_{tid}.png"

                            # If tile_id is like "6_5_65" but only part matches known tiles
                            else:
                                # Try to find exact match where the ID is embedded in the tile filename
                                matched = None
                                for kn in known_tiles:
                                    # Examples of matching:
                                    # "6_5_65" matches "sidebside_6_5_65.png"
                                    if tid in kn.replace(".png", ""):
                                        matched = kn
                                        break

                                if matched:
                                    p_copy['tile_id'] = matched
                                else:
                                    # FINAL fallback: convert directly → sidebside_6_5_65.png
                                    p_copy['tile_id'] = f"sidebside_{tid}.png"

                            mapped_points.append(p_copy)

                        points_tuple = tuple(mapped_points)
                        
                        # Color by the selected feature
                        tsne_chart = viz_tsne_scatter(
                            points_tuple, 
                            sel_feature,
                            f"t-SNE: {LOCATIONS[st.session_state.current_loc]['name']} - Colored by {sel_feature}"
                        )
                        st.altair_chart(tsne_chart, use_container_width=True)
                        
                        st.caption(f"📍 {len(loc_points)} tiles visualized. Similar tiles cluster together.")
                    else:
                        st.info(f"No t-SNE data for {LOCATIONS[st.session_state.current_loc]['name']}. Run: `python analysis/tsne_analysis.py`")
                else:
                    st.info("📊 Run `python analysis/tsne_analysis.py` to generate t-SNE embeddings.")

    else:
        st.info("📊 Run `python analysis/feature_impact_analysis.py` to generate factor analysis data.")

    st.markdown("---")
    with st.expander("🔬 Research Questions Addressed in Tab A", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **RQ1: Perception & Patterns** ✅
            **Perception & Patterns:** When does the model detect crosswalks correctly, and what causes it to miss or misidentify them?
            - Linked view strip shows detection patterns
            - Error overlay visualizes TP/FP/FN
            
            **RQ2: Confidence & Choice** ✅
            **Confidence & Choice:** How do small changes like threshold or clean-up, affect what the model marks as a crosswalk?
            - Threshold slider adjusts sensitivity
            - Threshold chart shows metric tradeoffs
            - Morphology controls refine predictions
            """)
        with c2:
            st.markdown("""
            **RQ5: Evidence to Trust** ✅
            **From Evidence to Trust:** What visual cues help users understand and trust the model’s predictions?
            - Side-by-side comparisons
            - Radar chart shows overall quality
            - Quality gauge provides intuitive score
            
            **RQ7: Edge Cases** ✅
            **Edge Cases & Novel Situations:** Are there unusual intersections or road layouts that consistently confuse the model, and what do these reveal about the model’s limits?
            - Factor analysis identifies failure modes
            - Two bar charts show tile and location impact
            - Low-IoU tile navigation
            """)

# Small helper functions used in Tab A threshold code (kept simple & local)
def compute_metrics_simulated(pred_mask: np.ndarray, seg: np.ndarray) -> Dict:
    """Quick approximate metrics given predicted mask and seg array (used for threshold chart)."""
    gt = get_mask_from_seg_array(seg, 'crosswalk').astype(bool)
    gt_exp = binary_dilation(gt, iterations=2)
    p = pred_mask.astype(bool)
    tp = int(np.sum(p & gt))
    fp = int(np.sum(p & ~gt_exp))
    fn = int(np.sum(gt_exp & ~p))
    tn = int(np.sum(~p & ~gt_exp))
    eps = 1e-8
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': round(prec, 4), 'recall': round(rec, 4),
            'f1': round(f1, 4), 'iou': round(iou, 4)}

# NOTE: we intentionally did NOT wrap this in @st.cache_data because it's a tiny helper
def compute_metrics_from_masks_cached(tmp_mask: np.ndarray, seg: np.ndarray) -> Dict:
    return compute_metrics_simulated(tmp_mask, seg)

# =============================================================================
# Tab B - Network Inspector (kept same behavior but map & results are cached)
# =============================================================================
def render_tab_b():
    st.header("🗺️ Network Inspector")
    st.caption("Geographic analysis and cross-location comparison")
    available = get_locations()
    results = get_results()
    if not available:
        st.error("❌ No data")
        return
    if 'tab_b_loc' not in st.session_state or st.session_state.tab_b_loc not in available:
        st.session_state.tab_b_loc = available[0]
    st.markdown("### 📍 Select Location")
    cols = st.columns(len(available))
    for i, loc in enumerate(available):
        with cols[i]:
            is_sel = (loc == st.session_state.tab_b_loc)
            if st.button(LOCATIONS[loc]['name'], key=f"b_{loc}",
                        type="primary" if is_sel else "secondary",
                        use_container_width=True):
                st.session_state.tab_b_loc = loc
                st.rerun()
    sel = st.session_state.tab_b_loc
    sb = results.get('stage_b', {}).get(sel, {})
    rq4 = sb.get('rq_answers', {}).get('RQ4', {})
    rq3 = sb.get('rq_answers', {}).get('RQ3', {})
    st.markdown("---")
    st.subheader(f"📊 Metrics: {LOCATIONS[sel]['name']}")
    st.caption("Ground truth comparison using OSM data")
    if rq4 and 'precision' in rq4:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            a, b = st.columns([5, 1])
            prec = rq4['precision']
            with a:
                st.metric("Precision", f"{prec:.1%}",
                         delta="Good" if prec > 0.8 else "Low",
                         delta_color="normal" if prec > 0.5 else "inverse")
            with b:
                with st.popover("?"): st.markdown(TIPS["precision"])
        with m2:
            a, b = st.columns([5, 1])
            rec = rq4['recall']
            with a:
                st.metric("Recall", f"{rec:.1%}",
                         delta="Good" if rec > 0.5 else "Low",
                         delta_color="normal" if rec > 0.3 else "inverse")
            with b:
                with st.popover("?"): st.markdown(TIPS["recall"])
        with m3:
            a, b = st.columns([5, 1])
            with a: st.metric("F1 Score", f"{rq4['f1']:.3f}")
            with b:
                with st.popover("?"): st.markdown(TIPS["f1"])
        with m4:
            a, b = st.columns([5, 1])
            gt = rq4.get('ground_truth', rq4.get('predictions', 0))
            with a: st.metric("Ground Truth", f"{gt:,}" if isinstance(gt, int) else str(gt))
            with b:
                with st.popover("?"): st.markdown(TIPS["gt"])
        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Error Breakdown:**")
            a, b = st.columns([5, 1])
            with a: st.metric("✅ True Positives", f"{rq4.get('tp', 0):,}")
            with b:
                with st.popover("?"): st.markdown(TIPS["tp"])
            a, b = st.columns([5, 1])
            with a: st.metric("🟠 False Positives", f"{rq4.get('fp', 0):,}")
            with b:
                with st.popover("?"): st.markdown(TIPS["fp"])
            a, b = st.columns([5, 1])
            with a: st.metric("🔵 False Negatives", f"{rq4.get('fn', 0):,}")
            with b:
                with st.popover("?"): st.markdown(TIPS["fn"])
        with c2:
            st.markdown("**Error Distribution**")
            err_data = {'TP': rq4.get('tp', 0), 'FP': rq4.get('fp', 0), 'FN': rq4.get('fn', 0)}
            fig = viz_donut_cached(tuple(sorted(err_data.items())), "Error Distribution")
            st.pyplot(fig, use_container_width=False, clear_figure=True)
    else:
        st.info("Run `python analysis/complete_stage_analysis.py` for metrics.")
    st.markdown("---")
    st.subheader("🗺️ Interactive Map")
    if HAS_FOLIUM:
        center = LOCATIONS[sel]['center']
        # cache map generation by selected location + available tiles names
        @st.cache_data
        def build_map_for_location(selected_loc: str, tile_names: Tuple[str, ...]):
            m = folium.Map(location=center, zoom_start=15, tiles='CartoDB positron')
            for loc in list(get_locations()):
                c = LOCATIONS[loc]['center']
                folium.Marker(
                    c, popup=LOCATIONS[loc]['name'], tooltip=LOCATIONS[loc]['name'],
                    icon=folium.Icon(color='blue' if loc == selected_loc else 'gray',
                                    icon='star' if loc == selected_loc else 'info-sign')
                ).add_to(m)
            tiles = get_tiles(selected_loc)
            # limit tiles to first 80 for performance (same as before)
            for i, t in enumerate(tiles[:80]):
                try:
                    _, seg = split_image_cached(t['path'])
                    det = np.any(get_mask_from_seg_array(seg, 'crosswalk'))
                    lat = center[0] - 0.004 + (i // 8) * 0.001
                    lon = center[1] - 0.004 + (i % 8) * 0.001
                    folium.CircleMarker(
                        [lat, lon], radius=6, color='green' if det else 'red',
                        fill=True, fillColor='green' if det else 'red', fillOpacity=0.7,
                        popup=t['name'], tooltip=f"{'✅' if det else '❌'} Tile {i+1}"
                    ).add_to(m)
                except:
                    pass
            return m
        tiles_list = tuple(sorted(list(tile_precompute_for_location(sel).keys())))
        m = build_map_for_location(sel, tiles_list)
        st.markdown("🟢 Detected | 🔴 No Detection | ⭐ Selected")
        st_folium(m, width=None, height=400)
    else:
        st.warning("Install folium for map")
    st.markdown("---")
    st.subheader("📍 Placement Analysis (RQ3)")
    if rq3 and 'near_sidewalk_pct' in rq3:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Near Sidewalk", f"{rq3['near_sidewalk_pct']:.1f}%")
        with c2: st.metric("Near Road", f"{rq3['near_road_pct']:.1f}%")
        with c3: st.metric("Total Detected", f"{rq3.get('total', 0):,}")
        g1, g2 = st.columns(2)
        with g1:
            fig = viz_gauge_cached(rq3['near_sidewalk_pct'] / 100, "Near Sidewalk")
            st.pyplot(fig)
            plt.close(fig)
        # with g2:
        #     fig = viz_gauge_cached(rq3['near_road_pct'] / 100, "Near Road")
        #     st.pyplot(fig)
        #     plt.close(fig)
    st.markdown("---")
    st.subheader("📊 All Locations Comparison")
    summary = []
    stack_data = []
    for loc in available:
        sb_loc = results.get('stage_b', {}).get(loc, {})
        rq = sb_loc.get('rq_answers', {}).get('RQ4', {})
        summary.append({
            'location': LOCATIONS[loc]['name'],
            'precision': rq.get('precision', 0),
            'recall': rq.get('recall', 0),
            'f1': rq.get('f1', 0),
            'tp': rq.get('tp', 0),
            'fp': rq.get('fp', 0),
            'fn': rq.get('fn', 0)
        })
        for et, val in [('TP', rq.get('tp', 0)), ('FP', rq.get('fp', 0)), ('FN', rq.get('fn', 0))]:
            stack_data.append({'location': LOCATIONS[loc]['name'], 'error_type': et, 'value': val})
    df = pd.DataFrame(summary)
    st.dataframe(
        df.style.format({
            'precision': '{:.1%}', 'recall': '{:.1%}', 'f1': '{:.3f}',
            'tp': '{:,}', 'fp': '{:,}', 'fn': '{:,}'
        }).background_gradient(subset=['precision', 'recall', 'f1'], cmap='RdYlGn'),
        use_container_width=True, hide_index=True
    )
    st.markdown("**Visual Comparisons**")
    v1, v2 = st.columns(2)
    with v1:
        if len(summary) > 1:
            summary_tuple = tuple((s['location'], s['precision'], s['recall'], s['f1']) for s in summary)
            st.altair_chart(viz_scatter_locations_cached(summary_tuple), use_container_width=True)
    with v2:
        if stack_data:
            stack_tuple = tuple((s['location'], s['error_type'], s['value']) for s in stack_data)
            st.altair_chart(viz_stacked_errors_cached(stack_tuple), use_container_width=True)
    st.markdown("---")
    with st.expander("🔬 Research Questions Addressed in Tab B", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **RQ3: Placement & Plausibility** ✅
            **Placement & Plausibility:** Do detected crosswalks appear in logical places, such as near intersections or sidewalks?
            - Placement analysis with gauge charts
            - Near sidewalk/road percentages
            - Visual validation on interactive map
            
            **RQ4: Agreement & Disagreement** ✅
            **Agreement & Disagreement:** How well do model results match official or open data, and where do they differ?
            - Precision/Recall/F1 vs OSM ground truth
            - TP/FP/FN breakdown per location
            - Cross-location comparison table
            """)
        with c2:
            st.markdown("""
            **RQ6: Usability for Decision-Making** ✅
            **Usability for Decision-Making:** When a user inspects the interactive visualizations, which cues help them trust or question the AI’s outputs?
            - Interactive map for geographic exploration
            - Color-coded tile markers (green/red)
            - Summary table for quick comparison
            - Scatter plot shows precision-recall tradeoffs
            - Stacked bar shows error distribution
            """)

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.title("🚶 CrossCheck-NYC")
    st.caption("Visual Analytics Tool for Evaluating AI-Mapped Crosswalk Data | ZebraSense Lads")
    with st.expander("📖 About This Tool", expanded=False):
        st.markdown("""
        **CrossCheck-NYC** is a two-stage visual analytics system for evaluating 
        AI-derived crosswalk maps from [Tile2Net](https://github.com/VIDA-NYU/tile2net).
        
        ---
        
        **🗺️ Locations Covered:**
        - Financial District (Manhattan)
        - East Village (Manhattan)
        - Bay Ridge (Brooklyn)
        - Downtown Brooklyn
        - Kew Gardens (Queens)
        
        ---
        
        **Stage A - Segmentation Detective:**
        Pixel-level analysis of AI predictions including error overlays,
        threshold tuning, morphology controls, and environmental factor analysis.
        
        **Stage B - Network Inspector:**
        Geographic visualization with OSM ground truth comparison,
        cross-location metrics, and placement analysis.
        
        ---        
        **How to Use:**
        1. Select a location from the sidebar
        2. Navigate tiles using buttons or dropdown
        3. Adjust threshold/morphology to see effects
        4. Use bookmarks to save interesting tiles
        5. Check Factors panel for failure analysis
        6. Use Tab B for map view and cross-location comparison
        """)

    tab_a, tab_b = st.tabs([
        "🔍 Tab A: Segmentation Detective",
        "🗺️ Tab B: Network Inspector"
    ])
    with tab_a:
        render_tab_a()
    with tab_b:
        render_tab_b()
    st.markdown("---")
    st.caption("CrossCheck-NYC | CS-GY 9223 Visual Analytics Fall 2025 | ZebraSense Lads | Powered by Tile2Net")

if __name__ == "__main__":
    main()
