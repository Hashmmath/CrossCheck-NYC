"""
Post-Processing for Brooklyn Crosswalk QA
=========================================

Light post-processing operations to improve predictions:
1. Threshold sweep
2. Morphological operations (open/close)
3. Connected component filtering

Records deltas in IoU/F1 to understand impact of each operation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from scipy import ndimage
from skimage import morphology
from skimage.measure import label, regionprops

from .metrics import SegmentationMetrics, MetricResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PostProcessingResult:
    """Container for post-processing results."""
    original_metrics: MetricResults
    processed_metrics: MetricResults
    processing_params: Dict
    delta_iou: float
    delta_f1: float
    processed_mask: np.ndarray


class PostProcessor:
    """
    Post-processing operations for segmentation predictions.
    
    Operations:
    - Threshold adjustment
    - Morphological opening (remove small FP)
    - Morphological closing (fill small FN)
    - Connected component filtering
    - Minimum area filtering
    """
    
    def __init__(self):
        """Initialize post-processor."""
        self.metrics = SegmentationMetrics()
    
    def apply_threshold(
        self,
        probabilities: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Apply threshold to probability map.
        
        Args:
            probabilities: Predicted probabilities
            threshold: Threshold value
            
        Returns:
            Binary mask
        """
        return (probabilities >= threshold).astype(np.uint8)
    
    def morphological_open(
        self,
        mask: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Apply morphological opening (erosion followed by dilation).
        
        Removes small isolated positive regions (potential false positives).
        
        Args:
            mask: Binary mask
            kernel_size: Size of structuring element
            
        Returns:
            Opened mask
        """
        kernel = morphology.disk(kernel_size // 2)
        opened = morphology.binary_opening(mask.astype(bool), kernel)
        return opened.astype(np.uint8)
    
    def morphological_close(
        self,
        mask: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Apply morphological closing (dilation followed by erosion).
        
        Fills small holes in positive regions (potential false negatives).
        
        Args:
            mask: Binary mask
            kernel_size: Size of structuring element
            
        Returns:
            Closed mask
        """
        kernel = morphology.disk(kernel_size // 2)
        closed = morphology.binary_closing(mask.astype(bool), kernel)
        return closed.astype(np.uint8)
    
    def filter_small_components(
        self,
        mask: np.ndarray,
        min_area: int = 50
    ) -> np.ndarray:
        """
        Remove connected components smaller than minimum area.
        
        Args:
            mask: Binary mask
            min_area: Minimum component area in pixels
            
        Returns:
            Filtered mask
        """
        labeled = label(mask)
        regions = regionprops(labeled)
        
        filtered = np.zeros_like(mask)
        for region in regions:
            if region.area >= min_area:
                filtered[labeled == region.label] = 1
        
        return filtered.astype(np.uint8)
    
    def filter_by_aspect_ratio(
        self,
        mask: np.ndarray,
        min_ratio: float = 0.1,
        max_ratio: float = 10.0
    ) -> np.ndarray:
        """
        Filter components by aspect ratio.
        
        Crosswalks typically have moderate aspect ratios.
        Very elongated or very square shapes may be noise.
        
        Args:
            mask: Binary mask
            min_ratio: Minimum aspect ratio
            max_ratio: Maximum aspect ratio
            
        Returns:
            Filtered mask
        """
        labeled = label(mask)
        regions = regionprops(labeled)
        
        filtered = np.zeros_like(mask)
        for region in regions:
            # Compute aspect ratio from bounding box
            minr, minc, maxr, maxc = region.bbox
            height = maxr - minr
            width = maxc - minc
            
            if height > 0 and width > 0:
                ratio = max(height, width) / min(height, width)
                if min_ratio <= ratio <= max_ratio:
                    filtered[labeled == region.label] = 1
        
        return filtered.astype(np.uint8)
    
    def process_with_params(
        self,
        probabilities: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5,
        open_kernel: Optional[int] = None,
        close_kernel: Optional[int] = None,
        min_area: Optional[int] = None
    ) -> PostProcessingResult:
        """
        Apply post-processing with given parameters and compute metrics.
        
        Args:
            probabilities: Predicted probabilities
            ground_truth: Ground truth mask
            threshold: Threshold value
            open_kernel: Morphological opening kernel size (None to skip)
            close_kernel: Morphological closing kernel size (None to skip)
            min_area: Minimum component area (None to skip)
            
        Returns:
            PostProcessingResult with before/after metrics
        """
        # Original metrics (at threshold)
        original_metrics = self.metrics.compute_all_metrics(
            probabilities, ground_truth, threshold
        )
        
        # Apply post-processing
        mask = self.apply_threshold(probabilities, threshold)
        
        processing_params = {'threshold': threshold}
        
        if open_kernel is not None:
            mask = self.morphological_open(mask, open_kernel)
            processing_params['open_kernel'] = open_kernel
        
        if close_kernel is not None:
            mask = self.morphological_close(mask, close_kernel)
            processing_params['close_kernel'] = close_kernel
        
        if min_area is not None:
            mask = self.filter_small_components(mask, min_area)
            processing_params['min_area'] = min_area
        
        # Processed metrics
        processed_metrics = self.metrics.compute_all_metrics(
            mask, ground_truth, threshold=0.5  # Already binary
        )
        
        return PostProcessingResult(
            original_metrics=original_metrics,
            processed_metrics=processed_metrics,
            processing_params=processing_params,
            delta_iou=processed_metrics.iou - original_metrics.iou,
            delta_f1=processed_metrics.f1 - original_metrics.f1,
            processed_mask=mask
        )
    
    def sweep_parameters(
        self,
        probabilities: np.ndarray,
        ground_truth: np.ndarray,
        thresholds: Optional[List[float]] = None,
        open_kernels: Optional[List[int]] = None,
        close_kernels: Optional[List[int]] = None,
        min_areas: Optional[List[int]] = None
    ) -> List[PostProcessingResult]:
        """
        Sweep through parameter combinations to find optimal settings.
        
        Args:
            probabilities: Predicted probabilities
            ground_truth: Ground truth mask
            thresholds: Thresholds to try
            open_kernels: Opening kernel sizes to try
            close_kernels: Closing kernel sizes to try
            min_areas: Minimum areas to try
            
        Returns:
            List of PostProcessingResult for each combination
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        if open_kernels is None:
            open_kernels = [None, 3, 5]
        
        if close_kernels is None:
            close_kernels = [None, 3, 5]
        
        if min_areas is None:
            min_areas = [None, 25, 50, 100]
        
        results = []
        
        for t in thresholds:
            for ok in open_kernels:
                for ck in close_kernels:
                    for ma in min_areas:
                        result = self.process_with_params(
                            probabilities, ground_truth,
                            threshold=t,
                            open_kernel=ok,
                            close_kernel=ck,
                            min_area=ma
                        )
                        results.append(result)
        
        return results
    
    def find_optimal_params(
        self,
        probabilities: np.ndarray,
        ground_truth: np.ndarray,
        metric: str = 'f1',
        **sweep_kwargs
    ) -> Tuple[Dict, PostProcessingResult]:
        """
        Find optimal post-processing parameters.
        
        Args:
            probabilities: Predicted probabilities
            ground_truth: Ground truth mask
            metric: Metric to optimize ('f1', 'iou')
            **sweep_kwargs: Arguments for sweep_parameters
            
        Returns:
            Tuple of (optimal params dict, best result)
        """
        results = self.sweep_parameters(
            probabilities, ground_truth, **sweep_kwargs
        )
        
        # Find best by metric
        metric_values = [getattr(r.processed_metrics, metric) for r in results]
        best_idx = np.argmax(metric_values)
        best_result = results[best_idx]
        
        return best_result.processing_params, best_result
    
    def analyze_morphology_effect(
        self,
        probabilities: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Analyze the effect of morphological operations.
        
        Returns detailed analysis of how open/close affect FP/FN.
        
        Args:
            probabilities: Predicted probabilities
            ground_truth: Ground truth mask
            threshold: Threshold value
            
        Returns:
            Dict with analysis results
        """
        base_mask = self.apply_threshold(probabilities, threshold)
        base_metrics = self.metrics.compute_all_metrics(base_mask, ground_truth)
        
        results = {
            'baseline': {
                'metrics': base_metrics.to_dict(),
                'fp_count': base_metrics.confusion.fp,
                'fn_count': base_metrics.confusion.fn
            },
            'operations': []
        }
        
        # Test various operations
        operations = [
            ('open_3', lambda m: self.morphological_open(m, 3)),
            ('open_5', lambda m: self.morphological_open(m, 5)),
            ('close_3', lambda m: self.morphological_close(m, 3)),
            ('close_5', lambda m: self.morphological_close(m, 5)),
            ('open_3_close_3', lambda m: self.morphological_close(
                self.morphological_open(m, 3), 3
            )),
        ]
        
        for name, op in operations:
            processed = op(base_mask)
            metrics = self.metrics.compute_all_metrics(processed, ground_truth)
            
            results['operations'].append({
                'name': name,
                'metrics': metrics.to_dict(),
                'delta_fp': metrics.confusion.fp - base_metrics.confusion.fp,
                'delta_fn': metrics.confusion.fn - base_metrics.confusion.fn,
                'delta_iou': metrics.iou - base_metrics.iou,
                'delta_f1': metrics.f1 - base_metrics.f1
            })
        
        return results


class OperatingPointSelector:
    """
    Select optimal operating point based on calibration and metrics.
    
    Combines threshold selection, morphology choice, and calibration
    analysis to recommend the best operating point.
    """
    
    def __init__(self):
        """Initialize selector."""
        self.post_processor = PostProcessor()
        self.metrics = SegmentationMetrics()
    
    def select_operating_point(
        self,
        probabilities: np.ndarray,
        ground_truth: np.ndarray,
        calibration_result=None,
        objective: str = 'balanced'
    ) -> Dict:
        """
        Select optimal operating point.
        
        Args:
            probabilities: Predicted probabilities
            ground_truth: Ground truth mask
            calibration_result: Optional calibration analysis
            objective: 'balanced', 'high_precision', or 'high_recall'
            
        Returns:
            Dict with recommended parameters and justification
        """
        # Threshold sweep
        threshold_results = self.metrics.threshold_sweep(
            probabilities, ground_truth
        )
        
        # Find optimal threshold based on objective
        if objective == 'balanced':
            # Maximize F1
            f1_values = [r.f1 for r in threshold_results]
            best_t_idx = np.argmax(f1_values)
        elif objective == 'high_precision':
            # Find threshold giving precision >= 0.8 with max recall
            valid = [r for r in threshold_results if r.precision >= 0.8]
            if valid:
                best_t_idx = np.argmax([r.recall for r in valid])
                best_t_idx = threshold_results.index(valid[best_t_idx])
            else:
                best_t_idx = np.argmax([r.precision for r in threshold_results])
        else:  # high_recall
            # Find threshold giving recall >= 0.8 with max precision
            valid = [r for r in threshold_results if r.recall >= 0.8]
            if valid:
                best_t_idx = np.argmax([r.precision for r in valid])
                best_t_idx = threshold_results.index(valid[best_t_idx])
            else:
                best_t_idx = np.argmax([r.recall for r in threshold_results])
        
        optimal_threshold = threshold_results[best_t_idx].threshold
        
        # Test morphology options at optimal threshold
        morph_params, morph_result = self.post_processor.find_optimal_params(
            probabilities, ground_truth,
            thresholds=[optimal_threshold],
            metric='f1'
        )
        
        # Build recommendation
        recommendation = {
            'threshold': optimal_threshold,
            'morphology': {
                'open_kernel': morph_params.get('open_kernel'),
                'close_kernel': morph_params.get('close_kernel'),
                'min_area': morph_params.get('min_area')
            },
            'objective': objective,
            'expected_metrics': morph_result.processed_metrics.to_dict(),
            'justification': self._generate_justification(
                threshold_results, morph_result, calibration_result, objective
            )
        }
        
        return recommendation
    
    def _generate_justification(
        self,
        threshold_results: List[MetricResults],
        morph_result: PostProcessingResult,
        calibration_result,
        objective: str
    ) -> str:
        """Generate human-readable justification for recommendations."""
        
        lines = []
        
        # Threshold justification
        best_metrics = morph_result.processed_metrics
        lines.append(
            f"Selected threshold {morph_result.processing_params['threshold']:.2f} "
            f"optimizing for {objective}."
        )
        lines.append(
            f"Expected F1: {best_metrics.f1:.3f}, "
            f"IoU: {best_metrics.iou:.3f}, "
            f"Precision: {best_metrics.precision:.3f}, "
            f"Recall: {best_metrics.recall:.3f}"
        )
        
        # Morphology justification
        if morph_result.delta_f1 > 0.01:
            lines.append(
                f"Morphological post-processing improves F1 by "
                f"{morph_result.delta_f1:.3f}."
            )
        elif morph_result.delta_f1 < -0.01:
            lines.append(
                "Morphological post-processing slightly reduces metrics; "
                "consider skipping."
            )
        else:
            lines.append("Morphological post-processing has minimal effect.")
        
        # Calibration note
        if calibration_result:
            if calibration_result.ece < 0.05:
                lines.append("Model is well-calibrated (ECE < 0.05).")
            elif calibration_result.ece < 0.1:
                lines.append(
                    f"Model is moderately calibrated (ECE = {calibration_result.ece:.3f})."
                )
            else:
                lines.append(
                    f"Model is poorly calibrated (ECE = {calibration_result.ece:.3f}). "
                    "Consider recalibration."
                )
        
        return " ".join(lines)