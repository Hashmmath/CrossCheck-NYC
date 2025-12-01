"""
Visualization Utilities for Brooklyn Crosswalk QA
================================================

Helpers for creating visualizations:
- Error overlay images
- Reliability diagrams
- Maps with folium/pydeck
- Metric plots
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationUtils:
    """Visualization utility functions."""
    
    # Default color scheme for error overlays
    DEFAULT_COLORS = {
        'tp': (0, 255, 0, 128),    # Green - True Positive
        'fp': (255, 0, 0, 128),    # Red - False Positive
        'fn': (0, 0, 255, 128),    # Blue - False Negative
        'tn': (0, 0, 0, 0),        # Transparent - True Negative
        'prediction': (255, 255, 0, 128)  # Yellow - Prediction
    }
    
    @staticmethod
    def create_error_overlay(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5,
        colors: Optional[Dict] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create RGBA error overlay image.
        
        Args:
            prediction: Prediction probabilities or binary mask
            ground_truth: Ground truth binary mask
            threshold: Threshold for binarization
            colors: Custom color dict
            alpha: Overall alpha multiplier
            
        Returns:
            RGBA array (H, W, 4)
        """
        colors = colors or VisualizationUtils.DEFAULT_COLORS
        
        # Binarize prediction
        if prediction.dtype in [np.float32, np.float64]:
            pred_binary = prediction >= threshold
        else:
            pred_binary = prediction.astype(bool)
        
        gt_binary = ground_truth.astype(bool)
        
        # Create masks
        tp = pred_binary & gt_binary
        fp = pred_binary & ~gt_binary
        fn = ~pred_binary & gt_binary
        
        # Create overlay
        height, width = prediction.shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        
        overlay[tp] = colors['tp']
        overlay[fp] = colors['fp']
        overlay[fn] = colors['fn']
        
        # Apply alpha multiplier
        overlay[:, :, 3] = (overlay[:, :, 3] * alpha).astype(np.uint8)
        
        return overlay
    
    @staticmethod
    def blend_with_image(
        base_image: np.ndarray,
        overlay: np.ndarray
    ) -> np.ndarray:
        """
        Blend overlay onto base image.
        
        Args:
            base_image: RGB or RGBA base image
            overlay: RGBA overlay
            
        Returns:
            Blended RGB image
        """
        # Ensure base is RGB
        if len(base_image.shape) == 2:
            base_image = np.stack([base_image] * 3, axis=-1)
        elif base_image.shape[-1] == 4:
            base_image = base_image[:, :, :3]
        
        # Normalize
        if base_image.dtype == np.uint8:
            base_float = base_image.astype(np.float32) / 255.0
        else:
            base_float = base_image.astype(np.float32)
        
        if overlay.dtype == np.uint8:
            overlay_float = overlay.astype(np.float32) / 255.0
        else:
            overlay_float = overlay.astype(np.float32)
        
        # Extract alpha
        alpha = overlay_float[:, :, 3:4]
        
        # Blend
        blended = base_float * (1 - alpha) + overlay_float[:, :, :3] * alpha
        
        # Convert back to uint8
        return (blended * 255).astype(np.uint8)
    
    @staticmethod
    def create_confidence_heatmap(
        probabilities: np.ndarray,
        cmap: str = 'viridis'
    ) -> np.ndarray:
        """
        Create heatmap visualization of prediction confidence.
        
        Args:
            probabilities: Prediction probabilities
            cmap: Matplotlib colormap name
            
        Returns:
            RGB heatmap array
        """
        # Normalize to 0-1
        prob_norm = np.clip(probabilities, 0, 1)
        
        # Apply colormap
        colormap = plt.cm.get_cmap(cmap)
        heatmap = colormap(prob_norm)
        
        # Convert to RGB uint8
        return (heatmap[:, :, :3] * 255).astype(np.uint8)
    
    @staticmethod
    def create_side_by_side(
        images: List[np.ndarray],
        titles: Optional[List[str]] = None,
        padding: int = 10
    ) -> np.ndarray:
        """
        Create side-by-side image comparison.
        
        Args:
            images: List of images to combine
            titles: Optional titles for each image
            padding: Padding between images
            
        Returns:
            Combined image
        """
        # Ensure all images are RGB
        processed = []
        max_height = 0
        
        for img in images:
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 4:
                img = img[:, :, :3]
            processed.append(img)
            max_height = max(max_height, img.shape[0])
        
        # Pad to same height
        padded = []
        for img in processed:
            if img.shape[0] < max_height:
                pad_height = max_height - img.shape[0]
                img = np.pad(img, ((0, pad_height), (0, 0), (0, 0)), mode='constant')
            padded.append(img)
        
        # Combine horizontally with padding
        separator = np.ones((max_height, padding, 3), dtype=np.uint8) * 255
        
        combined = padded[0]
        for img in padded[1:]:
            combined = np.concatenate([combined, separator, img], axis=1)
        
        return combined
    
    @staticmethod
    def save_comparison_figure(
        original: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        error_overlay: np.ndarray,
        output_path: Path,
        title: str = "Crosswalk Detection Comparison"
    ):
        """
        Save a comparison figure with multiple panels.
        
        Args:
            original: Original image
            ground_truth: Ground truth mask
            prediction: Prediction probabilities
            error_overlay: Error overlay
            output_path: Output file path
            title: Figure title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original
        axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Ground Truth
        axes[0, 1].imshow(ground_truth, cmap='gray')
        axes[0, 1].set_title("Ground Truth")
        axes[0, 1].axis('off')
        
        # Prediction
        im = axes[1, 0].imshow(prediction, cmap='viridis', vmin=0, vmax=1)
        axes[1, 0].set_title("Prediction Probabilities")
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Error overlay
        axes[1, 1].imshow(original)
        axes[1, 1].imshow(error_overlay)
        axes[1, 1].set_title("Error Overlay (Green=TP, Red=FP, Blue=FN)")
        axes[1, 1].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison figure to {output_path}")
    
    @staticmethod
    def plot_metrics_comparison(
        results: List[Dict],
        labels: List[str],
        metrics: List[str] = ['iou', 'f1', 'precision', 'recall'],
        output_path: Optional[Path] = None
    ):
        """
        Plot bar chart comparing metrics across different configurations.
        
        Args:
            results: List of result dicts
            labels: Labels for each result
            metrics: Metrics to plot
            output_path: Optional output path
        """
        n_metrics = len(metrics)
        n_configs = len(results)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(n_metrics)
        width = 0.8 / n_configs
        
        for i, (result, label) in enumerate(zip(results, labels)):
            values = [result.get(m, 0) for m in metrics]
            offset = (i - n_configs / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=label)
        
        ax.set_ylabel('Score')
        ax.set_title('Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_threshold_sweep(
        thresholds: List[float],
        metrics_dict: Dict[str, List[float]],
        output_path: Optional[Path] = None
    ):
        """
        Plot metrics across threshold sweep.
        
        Args:
            thresholds: List of threshold values
            metrics_dict: Dict mapping metric names to value lists
            output_path: Optional output path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric_name, values in metrics_dict.items():
            ax.plot(thresholds, values, '-o', label=metric_name.upper(), markersize=4)
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Threshold Sweep Analysis')
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def create_folium_map(
        center: Tuple[float, float] = (40.6920, -73.9825),
        zoom: int = 15
    ):
        """
        Create base Folium map for Brooklyn.
        
        Args:
            center: Map center (lat, lon)
            zoom: Initial zoom level
            
        Returns:
            Folium Map object
        """
        try:
            import folium
            
            m = folium.Map(
                location=center,
                zoom_start=zoom,
                tiles='OpenStreetMap'
            )
            
            # Add satellite layer
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite',
                overlay=False
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            return m
            
        except ImportError:
            logger.warning("Folium not available")
            return None
    
    @staticmethod
    def add_geojson_to_map(
        map_obj,
        geojson_path: Path,
        name: str = "Layer",
        style: Optional[Dict] = None
    ):
        """
        Add GeoJSON layer to Folium map.
        
        Args:
            map_obj: Folium Map object
            geojson_path: Path to GeoJSON file
            name: Layer name
            style: Style dict
        """
        try:
            import folium
            import json
            
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
            
            style = style or {
                'fillColor': '#ff7800',
                'color': '#000000',
                'weight': 1,
                'fillOpacity': 0.5
            }
            
            folium.GeoJson(
                geojson_data,
                name=name,
                style_function=lambda x: style
            ).add_to(map_obj)
            
        except ImportError:
            logger.warning("Folium not available")
        except Exception as e:
            logger.error(f"Failed to add GeoJSON: {e}")


def generate_tile_preview(
    tile_path: Path,
    pred_path: Path,
    gt_path: Path,
    output_path: Path,
    threshold: float = 0.5
):
    """
    Generate a preview image for a single tile.
    
    Args:
        tile_path: Path to original tile image
        pred_path: Path to prediction file
        gt_path: Path to ground truth file
        output_path: Output path for preview
    """
    import rasterio
    
    # Load original
    if tile_path.suffix.lower() in ['.tif', '.tiff']:
        with rasterio.open(tile_path) as src:
            original = src.read()
            if original.shape[0] == 3:
                original = np.transpose(original, (1, 2, 0))
    else:
        original = np.array(Image.open(tile_path).convert('RGB'))
    
    # Load prediction
    with rasterio.open(pred_path) as src:
        prediction = src.read(1)
    
    # Load ground truth
    with rasterio.open(gt_path) as src:
        ground_truth = src.read(1)
    
    # Create error overlay
    overlay = VisualizationUtils.create_error_overlay(
        prediction, ground_truth, threshold
    )
    
    # Save comparison
    VisualizationUtils.save_comparison_figure(
        original, ground_truth, prediction, overlay,
        output_path,
        title=f"Tile: {tile_path.stem}"
    )