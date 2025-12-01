"""
Enhanced Visualization Module for Crosswalk QA
==============================================

Creates visualizations matching the Tile2Net paper style:
1. Multi-panel comparison views (Original, GT, Prediction, Error Overlay)
2. Class-colored segmentation maps (Sidewalk=Blue, Crosswalk=Orange, Road=Gray)
3. Calibration plots (reliability diagrams, ECE curves)
4. Threshold sweep visualizations

Usage:
    python enhanced_viz.py --input data/processed --output outputs/figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from PIL import Image
import rasterio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Color scheme matching Tile2Net paper
TILE2NET_COLORS = {
    'background': '#000000',  # Black
    'road': '#808080',        # Gray
    'sidewalk': '#4169E1',    # Royal Blue
    'crosswalk': '#FF6B35',   # Orange
    'footpath': '#32CD32',    # Lime Green
}

# Error overlay colors
ERROR_COLORS = {
    'TP': '#00FF00',  # Green - True Positive
    'FP': '#FF0000',  # Red - False Positive  
    'FN': '#0000FF',  # Blue - False Negative
    'TN': None,       # Transparent - True Negative
}


class EnhancedVisualizer:
    """
    Create publication-quality visualizations for crosswalk detection QA.
    """
    
    def __init__(self, output_dir: str = "outputs/figures"):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup colormaps
        self._setup_colormaps()
    
    def _setup_colormaps(self):
        """Create custom colormaps for segmentation visualization."""
        
        # 5-class segmentation colormap (Tile2Net style)
        colors = [
            (0, 0, 0, 1),           # 0: Background - Black
            (0.5, 0.5, 0.5, 1),     # 1: Road - Gray
            (0.25, 0.41, 0.88, 1),  # 2: Sidewalk - Blue
            (1.0, 0.42, 0.21, 1),   # 3: Crosswalk - Orange
            (0.2, 0.8, 0.2, 1),     # 4: Footpath - Green
        ]
        self.seg_cmap = ListedColormap(colors, name='tile2net_seg')
        
        # Error overlay colormap
        error_colors = [
            (0, 0, 0, 0),           # 0: TN - Transparent
            (0, 1, 0, 0.7),         # 1: TP - Green
            (1, 0, 0, 0.7),         # 2: FP - Red
            (0, 0, 1, 0.7),         # 3: FN - Blue
        ]
        self.error_cmap = ListedColormap(error_colors, name='error_overlay')
        
        # Probability colormap (viridis-like but with more contrast)
        self.prob_cmap = 'viridis'
    
    def create_tile2net_comparison(
        self,
        original: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        title: str = "Tile Comparison",
        save_path: Optional[str] = None,
        show_legend: bool = True
    ) -> plt.Figure:
        """
        Create Tile2Net paper-style comparison figure.
        
        Layout:
        +-------------------+-------------------+
        | Original Image    | Ground Truth      |
        +-------------------+-------------------+
        | Prediction        | Error Overlay     |
        +-------------------+-------------------+
        
        Args:
            original: RGB image (H, W, 3)
            ground_truth: Binary or multi-class mask (H, W)
            prediction: Binary or multi-class mask (H, W)
            probabilities: Optional probability map (H, W)
            title: Figure title
            save_path: Path to save figure
            show_legend: Whether to show class legend
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Panel a: Original Image
        ax = axes[0, 0]
        ax.imshow(original)
        ax.set_title("(a) Orthorectified Tile", fontsize=11)
        ax.axis('off')
        
        # Panel b: Ground Truth
        ax = axes[0, 1]
        self._plot_segmentation(ax, ground_truth, is_binary=True)
        ax.set_title("(b) Ground Truth", fontsize=11)
        ax.axis('off')
        
        # Panel c: Prediction or Probabilities
        ax = axes[1, 0]
        if probabilities is not None:
            im = ax.imshow(probabilities, cmap=self.prob_cmap, vmin=0, vmax=1)
            ax.set_title("(c) Prediction Probabilities", fontsize=11)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('P(crosswalk)', fontsize=9)
        else:
            self._plot_segmentation(ax, prediction, is_binary=True)
            ax.set_title("(c) Prediction", fontsize=11)
        ax.axis('off')
        
        # Panel d: Error Overlay
        ax = axes[1, 1]
        self._plot_error_overlay(ax, original, ground_truth, prediction)
        ax.set_title("(d) Error Overlay", fontsize=11)
        ax.axis('off')
        
        # Add legend
        if show_legend:
            self._add_error_legend(fig)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved comparison figure to {save_path}")
        
        return fig
    
    def create_multiclass_comparison(
        self,
        original: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        title: str = "Multi-class Segmentation",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create multi-class segmentation comparison (Tile2Net paper Fig 3 style).
        
        Layout:
        +-------------------+-------------------+-------------------+
        | Original Tile     | Ground Truth      | Prediction        |
        +-------------------+-------------------+-------------------+
        
        With legend: Sidewalk=Blue, Crosswalk=Orange, Road=Gray, Background=Black
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Original
        axes[0].imshow(original)
        axes[0].set_title("Orthorectified Tile", fontsize=11)
        axes[0].axis('off')
        
        # Ground Truth (multi-class)
        self._plot_segmentation(axes[1], ground_truth, is_binary=False)
        axes[1].set_title("Annotation (Ground Truth)", fontsize=11)
        axes[1].axis('off')
        
        # Prediction (multi-class)
        self._plot_segmentation(axes[2], prediction, is_binary=False)
        axes[2].set_title("Model Prediction", fontsize=11)
        axes[2].axis('off')
        
        # Add class legend
        self._add_class_legend(fig)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def _plot_segmentation(
        self, 
        ax: plt.Axes, 
        mask: np.ndarray,
        is_binary: bool = True
    ):
        """Plot segmentation mask with appropriate coloring."""
        
        if is_binary:
            # Binary crosswalk mask - use orange on black
            colored = np.zeros((*mask.shape, 4))
            colored[mask > 0] = [1.0, 0.42, 0.21, 1.0]  # Orange for crosswalk
            colored[mask == 0] = [0, 0, 0, 1.0]          # Black for background
            ax.imshow(colored)
        else:
            # Multi-class - use full colormap
            ax.imshow(mask, cmap=self.seg_cmap, vmin=0, vmax=4, interpolation='nearest')
    
    def _plot_error_overlay(
        self,
        ax: plt.Axes,
        original: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        alpha: float = 0.5
    ):
        """Plot original image with error overlay."""
        
        # Ensure binary
        gt_binary = (ground_truth > 0).astype(np.uint8)
        pred_binary = (prediction > 0).astype(np.uint8)
        
        # Compute error categories
        # 0: TN (neither), 1: TP (both), 2: FP (pred only), 3: FN (gt only)
        error_mask = np.zeros_like(gt_binary, dtype=np.uint8)
        error_mask[(gt_binary == 1) & (pred_binary == 1)] = 1  # TP
        error_mask[(gt_binary == 0) & (pred_binary == 1)] = 2  # FP
        error_mask[(gt_binary == 1) & (pred_binary == 0)] = 3  # FN
        
        # Show original
        ax.imshow(original)
        
        # Create RGBA overlay
        overlay = np.zeros((*error_mask.shape, 4))
        overlay[error_mask == 1] = [0, 1, 0, alpha]      # TP: Green
        overlay[error_mask == 2] = [1, 0, 0, alpha]      # FP: Red
        overlay[error_mask == 3] = [0, 0, 1, alpha]      # FN: Blue
        
        ax.imshow(overlay)
    
    def _add_error_legend(self, fig: plt.Figure):
        """Add error overlay legend to figure."""
        legend_elements = [
            mpatches.Patch(facecolor='green', alpha=0.7, label='True Positive (TP)'),
            mpatches.Patch(facecolor='red', alpha=0.7, label='False Positive (FP)'),
            mpatches.Patch(facecolor='blue', alpha=0.7, label='False Negative (FN)'),
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=3,
            fontsize=10,
            frameon=True,
            fancybox=True
        )
    
    def _add_class_legend(self, fig: plt.Figure):
        """Add segmentation class legend."""
        legend_elements = [
            mpatches.Patch(facecolor=TILE2NET_COLORS['sidewalk'], label='Sidewalk'),
            mpatches.Patch(facecolor=TILE2NET_COLORS['crosswalk'], label='Crosswalk'),
            mpatches.Patch(facecolor=TILE2NET_COLORS['road'], label='Road'),
            mpatches.Patch(facecolor=TILE2NET_COLORS['background'], label='Background'),
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=4,
            fontsize=10,
            frameon=True,
            fancybox=True
        )
    
    def create_calibration_figure(
        self,
        reliability_data: Dict,
        ece: float,
        mce: float,
        title: str = "Model Calibration Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create calibration analysis figure with reliability diagram.
        
        Args:
            reliability_data: Dict with 'bin_centers', 'bin_accuracies', 'bin_counts'
            ece: Expected Calibration Error
            mce: Maximum Calibration Error
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Reliability Diagram
        ax = axes[0]
        bin_centers = reliability_data['bin_centers']
        bin_accuracies = reliability_data['bin_accuracies']
        bin_counts = reliability_data['bin_counts']
        
        # Bar chart for reliability
        width = 0.08
        bars = ax.bar(bin_centers, bin_accuracies, width=width, 
                      color='steelblue', edgecolor='black', alpha=0.7,
                      label='Observed Accuracy')
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        
        # Gap visualization (difference between observed and expected)
        for i, (center, acc) in enumerate(zip(bin_centers, bin_accuracies)):
            if not np.isnan(acc):
                ax.plot([center, center], [center, acc], 'r-', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Fraction of Positives', fontsize=11)
        ax.set_title(f'Reliability Diagram (ECE={ece:.4f})', fontsize=11)
        ax.legend(loc='upper left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Confidence histogram
        ax = axes[1]
        # Normalize bin counts
        total = sum(bin_counts)
        if total > 0:
            bin_fractions = [c / total for c in bin_counts]
        else:
            bin_fractions = bin_counts
        
        ax.bar(bin_centers, bin_fractions, width=width,
               color='gray', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Fraction of Samples', fontsize=11)
        ax.set_title('Confidence Distribution', fontsize=11)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add text box with metrics
        textstr = f'ECE: {ece:.4f}\nMCE: {mce:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved calibration figure to {save_path}")
        
        return fig
    
    def create_threshold_sweep_figure(
        self,
        thresholds: List[float],
        metrics: Dict[str, List[float]],
        optimal_threshold: Optional[float] = None,
        title: str = "Threshold Sensitivity Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create threshold sweep analysis figure.
        
        Args:
            thresholds: List of threshold values
            metrics: Dict with lists for 'iou', 'f1', 'precision', 'recall'
            optimal_threshold: Optimal threshold to highlight
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot metrics
        colors = {'iou': 'blue', 'f1': 'green', 'precision': 'orange', 'recall': 'purple'}
        labels = {'iou': 'IoU', 'f1': 'F1 Score', 'precision': 'Precision', 'recall': 'Recall'}
        
        for metric_name, values in metrics.items():
            if metric_name in colors:
                ax.plot(thresholds, values, '-o', color=colors[metric_name],
                       label=labels[metric_name], linewidth=2, markersize=4)
        
        # Mark optimal threshold
        if optimal_threshold is not None:
            ax.axvline(x=optimal_threshold, color='red', linestyle='--', 
                      linewidth=2, label=f'Optimal τ = {optimal_threshold:.2f}')
        
        ax.set_xlabel('Threshold (τ)', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(thresholds), max(thresholds))
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved threshold sweep figure to {save_path}")
        
        return fig
    
    def create_risk_coverage_curve(
        self,
        coverage: List[float],
        accuracy: List[float],
        iou: Optional[List[float]] = None,
        title: str = "Risk-Coverage Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create risk-coverage curve showing accuracy vs fraction of pixels kept.
        
        This shows how accuracy improves as we become more selective
        (keep only high-confidence pixels).
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(coverage, accuracy, 'b-o', linewidth=2, markersize=4, label='Accuracy')
        
        if iou is not None:
            ax.plot(coverage, iou, 'g-s', linewidth=2, markersize=4, label='IoU')
        
        ax.set_xlabel('Coverage (Fraction of Pixels Kept)', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add diagonal reference (random baseline)
        ax.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.5, label='Random')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def create_morphology_comparison(
        self,
        original_pred: np.ndarray,
        processed_pred: np.ndarray,
        ground_truth: np.ndarray,
        original_image: np.ndarray,
        operation: str = "opening",
        kernel_size: int = 5,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare before/after morphological post-processing.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Morphology Effect: {operation} (kernel={kernel_size}×{kernel_size})", 
                    fontsize=14, fontweight='bold')
        
        # Row 1: Before
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(original_pred, cmap='gray')
        axes[0, 1].set_title("Before Post-processing")
        axes[0, 1].axis('off')
        
        self._plot_error_overlay(axes[0, 2], original_image, ground_truth, original_pred)
        axes[0, 2].set_title("Errors Before")
        axes[0, 2].axis('off')
        
        # Row 2: After
        axes[1, 0].imshow(original_image)
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(processed_pred, cmap='gray')
        axes[1, 1].set_title("After Post-processing")
        axes[1, 1].axis('off')
        
        self._plot_error_overlay(axes[1, 2], original_image, ground_truth, processed_pred)
        axes[1, 2].set_title("Errors After")
        axes[1, 2].axis('off')
        
        # Add legend
        self._add_error_legend(fig)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        
        return fig


def process_tile_for_visualization(
    original_path: str,
    gt_path: str,
    prob_path: str,
    threshold: float = 0.5,
    output_dir: str = "outputs/figures"
) -> str:
    """
    Process a single tile and create visualization.
    
    Args:
        original_path: Path to original ortho image
        gt_path: Path to ground truth mask
        prob_path: Path to probability map
        threshold: Threshold for binary prediction
        output_dir: Output directory for figures
    """
    # Load data
    original = np.array(Image.open(original_path))
    
    with rasterio.open(gt_path) as src:
        gt = src.read(1)
    
    with rasterio.open(prob_path) as src:
        probs = src.read(1)
    
    # Create binary prediction
    pred = (probs >= threshold).astype(np.uint8)
    
    # Create visualizer
    viz = EnhancedVisualizer(output_dir)
    
    # Create comparison figure
    tile_name = Path(original_path).stem
    save_path = Path(output_dir) / f"{tile_name}_comparison.png"
    
    fig = viz.create_tile2net_comparison(
        original=original,
        ground_truth=gt,
        prediction=pred,
        probabilities=probs,
        title=f"Tile: {tile_name}",
        save_path=str(save_path)
    )
    
    plt.close(fig)
    
    return str(save_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create enhanced visualizations")
    parser.add_argument("--original", required=True, help="Original image path")
    parser.add_argument("--gt", required=True, help="Ground truth mask path")
    parser.add_argument("--prob", required=True, help="Probability map path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    parser.add_argument("--output", default="outputs/figures", help="Output directory")
    
    args = parser.parse_args()
    
    result = process_tile_for_visualization(
        original_path=args.original,
        gt_path=args.gt,
        prob_path=args.prob,
        threshold=args.threshold,
        output_dir=args.output
    )
    
    print(f"Created visualization: {result}")