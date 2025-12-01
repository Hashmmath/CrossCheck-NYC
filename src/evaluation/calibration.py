"""
Calibration Analysis for Brooklyn Crosswalk QA
=============================================

Analyzes model calibration - how well predicted probabilities
match actual outcomes.

Key outputs:
- Reliability diagram (calibration curve)
- Expected Calibration Error (ECE)
- Risk-coverage curves

A well-calibrated model should have predicted probabilities that
match observed frequencies. E.g., for all pixels where the model
predicts 0.7 probability, about 70% should actually be crosswalks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalibrationBin:
    """Container for a single calibration bin."""
    bin_idx: int
    bin_lower: float
    bin_upper: float
    bin_center: float
    avg_confidence: float  # Mean predicted probability in bin
    avg_accuracy: float    # Fraction of positive ground truth in bin
    count: int             # Number of pixels in bin
    
    @property
    def calibration_error(self) -> float:
        """Absolute difference between confidence and accuracy."""
        return abs(self.avg_confidence - self.avg_accuracy)


@dataclass
class CalibrationResult:
    """Container for calibration analysis results."""
    bins: List[CalibrationBin]
    ece: float                    # Expected Calibration Error
    mce: float                    # Maximum Calibration Error
    reliability_diagram: Dict     # Data for plotting
    n_samples: int
    
    def to_dict(self) -> Dict:
        return {
            'ece': self.ece,
            'mce': self.mce,
            'n_bins': len(self.bins),
            'n_samples': self.n_samples,
            'bins': [
                {
                    'bin_idx': b.bin_idx,
                    'bin_center': b.bin_center,
                    'avg_confidence': b.avg_confidence,
                    'avg_accuracy': b.avg_accuracy,
                    'count': b.count,
                    'calibration_error': b.calibration_error
                }
                for b in self.bins
            ]
        }


class CalibrationAnalyzer:
    """
    Analyze calibration of segmentation model predictions.
    
    Calibration measures how well predicted probabilities match
    actual frequencies. This is critical for:
    - Understanding model confidence
    - Setting appropriate thresholds
    - Identifying overconfident/underconfident predictions
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        strategy: str = 'uniform'
    ):
        """
        Initialize calibration analyzer.
        
        Args:
            n_bins: Number of bins for calibration analysis
            strategy: 'uniform' (equal width) or 'quantile' (equal count)
        """
        self.n_bins = n_bins
        self.strategy = strategy
    
    def compute_calibration(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> CalibrationResult:
        """
        Compute calibration metrics.
        
        Args:
            predictions: Predicted probabilities (H, W) or flattened
            ground_truth: Ground truth binary mask
            mask: Optional mask to exclude certain pixels (e.g., edges)
            
        Returns:
            CalibrationResult with bins, ECE, MCE
        """
        # Flatten arrays
        pred_flat = predictions.flatten().astype(np.float64)
        gt_flat = ground_truth.flatten().astype(np.float64)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.flatten().astype(bool)
            pred_flat = pred_flat[mask_flat]
            gt_flat = gt_flat[mask_flat]
        
        n_samples = len(pred_flat)
        
        if n_samples == 0:
            logger.warning("No samples for calibration analysis")
            return CalibrationResult(
                bins=[], ece=0.0, mce=0.0,
                reliability_diagram={}, n_samples=0
            )
        
        # Determine bin edges
        if self.strategy == 'uniform':
            bin_edges = np.linspace(0, 1, self.n_bins + 1)
        else:  # quantile
            # Use percentiles to create equal-count bins
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.percentile(pred_flat, percentiles)
            bin_edges[0] = 0.0
            bin_edges[-1] = 1.0
        
        # Compute bin statistics
        bins = []
        
        for i in range(self.n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            
            # Find samples in this bin
            if i == self.n_bins - 1:
                # Include right edge for last bin
                in_bin = (pred_flat >= lower) & (pred_flat <= upper)
            else:
                in_bin = (pred_flat >= lower) & (pred_flat < upper)
            
            count = np.sum(in_bin)
            
            if count > 0:
                avg_confidence = np.mean(pred_flat[in_bin])
                avg_accuracy = np.mean(gt_flat[in_bin])
            else:
                avg_confidence = (lower + upper) / 2
                avg_accuracy = 0.0
            
            bins.append(CalibrationBin(
                bin_idx=i,
                bin_lower=lower,
                bin_upper=upper,
                bin_center=(lower + upper) / 2,
                avg_confidence=avg_confidence,
                avg_accuracy=avg_accuracy,
                count=int(count)
            ))
        
        # Compute ECE (Expected Calibration Error)
        # ECE = sum(|accuracy - confidence| * count) / total
        ece = sum(
            b.calibration_error * b.count for b in bins
        ) / n_samples
        
        # Compute MCE (Maximum Calibration Error)
        mce = max(b.calibration_error for b in bins) if bins else 0.0
        
        # Prepare reliability diagram data
        reliability_diagram = {
            'confidence': [b.avg_confidence for b in bins],
            'accuracy': [b.avg_accuracy for b in bins],
            'counts': [b.count for b in bins],
            'bin_centers': [b.bin_center for b in bins]
        }
        
        return CalibrationResult(
            bins=bins,
            ece=ece,
            mce=mce,
            reliability_diagram=reliability_diagram,
            n_samples=n_samples
        )
    
    def compute_risk_coverage(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        metric: str = 'accuracy',
        n_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute risk-coverage curve.
        
        Shows how accuracy/IoU changes as we reject low-confidence predictions.
        
        Args:
            predictions: Predicted probabilities
            ground_truth: Ground truth binary mask
            metric: 'accuracy', 'iou', or 'f1'
            n_points: Number of coverage points
            
        Returns:
            Dict with 'coverage', 'risk' (or chosen metric), 'thresholds'
        """
        pred_flat = predictions.flatten()
        gt_flat = ground_truth.flatten().astype(bool)
        
        # Confidence thresholds
        thresholds = np.linspace(0, 1, n_points)
        
        coverages = []
        risks = []
        
        for t in thresholds:
            # Keep only pixels with confidence >= t (in either direction)
            # For binary segmentation, confidence = max(p, 1-p)
            confidence = np.maximum(pred_flat, 1 - pred_flat)
            mask = confidence >= t
            
            coverage = np.mean(mask)
            coverages.append(coverage)
            
            if coverage > 0:
                # Compute metric on kept pixels
                pred_kept = (pred_flat[mask] >= 0.5).astype(bool)
                gt_kept = gt_flat[mask]
                
                if metric == 'accuracy':
                    risk_value = np.mean(pred_kept == gt_kept)
                elif metric == 'iou':
                    tp = np.sum(pred_kept & gt_kept)
                    fp = np.sum(pred_kept & ~gt_kept)
                    fn = np.sum(~pred_kept & gt_kept)
                    risk_value = tp / (tp + fp + fn + 1e-7)
                elif metric == 'f1':
                    tp = np.sum(pred_kept & gt_kept)
                    fp = np.sum(pred_kept & ~gt_kept)
                    fn = np.sum(~pred_kept & gt_kept)
                    risk_value = 2 * tp / (2 * tp + fp + fn + 1e-7)
                else:
                    risk_value = np.mean(pred_kept == gt_kept)
                
                risks.append(risk_value)
            else:
                risks.append(1.0)  # Perfect if we reject everything
        
        return {
            'coverage': np.array(coverages),
            metric: np.array(risks),
            'thresholds': thresholds
        }
    
    def compute_confidence_histogram(
        self,
        predictions: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        n_bins: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Compute histogram of predicted confidences.
        
        Optionally separate by ground truth class for analysis.
        
        Args:
            predictions: Predicted probabilities
            ground_truth: Optional ground truth for class separation
            n_bins: Number of histogram bins
            
        Returns:
            Dict with histogram data
        """
        pred_flat = predictions.flatten()
        
        result = {
            'bins': np.linspace(0, 1, n_bins + 1),
            'all_counts': np.histogram(pred_flat, bins=n_bins, range=(0, 1))[0]
        }
        
        if ground_truth is not None:
            gt_flat = ground_truth.flatten().astype(bool)
            
            # Separate positive and negative ground truth
            result['positive_counts'] = np.histogram(
                pred_flat[gt_flat], bins=n_bins, range=(0, 1)
            )[0]
            result['negative_counts'] = np.histogram(
                pred_flat[~gt_flat], bins=n_bins, range=(0, 1)
            )[0]
        
        return result
    
    def analyze_overconfidence(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        confidence_threshold: float = 0.9
    ) -> Dict:
        """
        Analyze overconfident predictions.
        
        Finds pixels where model is very confident but wrong.
        
        Args:
            predictions: Predicted probabilities
            ground_truth: Ground truth binary mask
            confidence_threshold: Threshold for "high confidence"
            
        Returns:
            Dict with overconfidence analysis
        """
        pred_flat = predictions.flatten()
        gt_flat = ground_truth.flatten().astype(bool)
        
        # High confidence predictions (either class)
        high_conf_pos = pred_flat >= confidence_threshold
        high_conf_neg = pred_flat <= (1 - confidence_threshold)
        high_conf = high_conf_pos | high_conf_neg
        
        # Correct predictions
        pred_class = pred_flat >= 0.5
        correct = pred_class == gt_flat
        
        # Overconfident = high confidence but wrong
        overconfident = high_conf & ~correct
        
        return {
            'overconfident_count': int(np.sum(overconfident)),
            'overconfident_rate': float(np.mean(overconfident)),
            'high_confidence_count': int(np.sum(high_conf)),
            'high_confidence_accuracy': float(np.mean(correct[high_conf])) if np.sum(high_conf) > 0 else 0.0,
            'confidence_threshold': confidence_threshold,
            
            # Breakdown by prediction type
            'overconfident_false_positives': int(np.sum(
                high_conf_pos & ~gt_flat
            )),
            'overconfident_false_negatives': int(np.sum(
                high_conf_neg & gt_flat
            ))
        }
    
    def aggregate_calibration(
        self,
        results_list: List[CalibrationResult]
    ) -> CalibrationResult:
        """
        Aggregate calibration results from multiple tiles.
        
        Args:
            results_list: List of CalibrationResult from multiple tiles
            
        Returns:
            Aggregated CalibrationResult
        """
        if not results_list:
            return CalibrationResult(
                bins=[], ece=0.0, mce=0.0,
                reliability_diagram={}, n_samples=0
            )
        
        # Aggregate bin statistics
        total_samples = sum(r.n_samples for r in results_list)
        
        # Weight ECE by sample count
        weighted_ece = sum(
            r.ece * r.n_samples for r in results_list
        ) / total_samples
        
        # MCE is max across all results
        mce = max(r.mce for r in results_list)
        
        # Aggregate bin data (weighted average)
        n_bins = len(results_list[0].bins)
        aggregated_bins = []
        
        for i in range(n_bins):
            total_count = sum(r.bins[i].count for r in results_list)
            
            if total_count > 0:
                avg_confidence = sum(
                    r.bins[i].avg_confidence * r.bins[i].count for r in results_list
                ) / total_count
                avg_accuracy = sum(
                    r.bins[i].avg_accuracy * r.bins[i].count for r in results_list
                ) / total_count
            else:
                avg_confidence = results_list[0].bins[i].bin_center
                avg_accuracy = 0.0
            
            aggregated_bins.append(CalibrationBin(
                bin_idx=i,
                bin_lower=results_list[0].bins[i].bin_lower,
                bin_upper=results_list[0].bins[i].bin_upper,
                bin_center=results_list[0].bins[i].bin_center,
                avg_confidence=avg_confidence,
                avg_accuracy=avg_accuracy,
                count=total_count
            ))
        
        reliability_diagram = {
            'confidence': [b.avg_confidence for b in aggregated_bins],
            'accuracy': [b.avg_accuracy for b in aggregated_bins],
            'counts': [b.count for b in aggregated_bins],
            'bin_centers': [b.bin_center for b in aggregated_bins]
        }
        
        return CalibrationResult(
            bins=aggregated_bins,
            ece=weighted_ece,
            mce=mce,
            reliability_diagram=reliability_diagram,
            n_samples=total_samples
        )


def plot_reliability_diagram(
    calibration_result: CalibrationResult,
    ax=None,
    show_histogram: bool = True,
    title: str = "Reliability Diagram"
):
    """
    Plot reliability diagram.
    
    Args:
        calibration_result: CalibrationResult to plot
        ax: Matplotlib axis (creates new if None)
        show_histogram: Whether to show bin count histogram
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    rd = calibration_result.reliability_diagram
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7)
    
    # Calibration curve
    ax.plot(
        rd['confidence'], rd['accuracy'],
        'b-o', label=f'Model (ECE={calibration_result.ece:.3f})',
        markersize=8
    )
    
    # Fill gap to show calibration error
    for i in range(len(rd['confidence'])):
        ax.fill_between(
            [rd['confidence'][i] - 0.02, rd['confidence'][i] + 0.02],
            [rd['confidence'][i], rd['confidence'][i]],
            [rd['accuracy'][i], rd['accuracy'][i]],
            alpha=0.3, color='red'
        )
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Add histogram as twin axis
    if show_histogram:
        ax2 = ax.twinx()
        bin_width = rd['bin_centers'][1] - rd['bin_centers'][0] if len(rd['bin_centers']) > 1 else 0.1
        ax2.bar(
            rd['bin_centers'], rd['counts'],
            width=bin_width * 0.8, alpha=0.2, color='gray',
            label='Sample count'
        )
        ax2.set_ylabel('Count', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
    
    return ax


def plot_risk_coverage(
    risk_coverage: Dict[str, np.ndarray],
    metric_name: str = 'accuracy',
    ax=None,
    title: str = "Risk-Coverage Curve"
):
    """
    Plot risk-coverage curve.
    
    Args:
        risk_coverage: Output from compute_risk_coverage
        metric_name: Name of metric to plot
        ax: Matplotlib axis
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(
        risk_coverage['coverage'],
        risk_coverage[metric_name],
        'b-', linewidth=2
    )
    
    ax.set_xlabel('Coverage (fraction of pixels used)')
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    return ax