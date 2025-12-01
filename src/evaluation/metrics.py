"""
Segmentation Metrics for Brooklyn Crosswalk QA
==============================================

Computes pixel-level evaluation metrics:
- IoU (Intersection over Union)
- F1 Score (Dice coefficient)
- Precision and Recall
- Confusion matrices (TP/FP/FN/TN)

All metrics can be computed at different thresholds for threshold sweep analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConfusionMatrix:
    """Container for confusion matrix values."""
    tp: int  # True Positives
    fp: int  # False Positives
    fn: int  # False Negatives
    tn: int  # True Negatives
    
    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn
    
    @property
    def positives(self) -> int:
        """Total actual positives (TP + FN)."""
        return self.tp + self.fn
    
    @property
    def negatives(self) -> int:
        """Total actual negatives (TN + FP)."""
        return self.tn + self.fp
    
    @property
    def predicted_positives(self) -> int:
        """Total predicted positives (TP + FP)."""
        return self.tp + self.fp
    
    @property
    def predicted_negatives(self) -> int:
        """Total predicted negatives (TN + FN)."""
        return self.tn + self.fn
    
    def to_dict(self) -> Dict:
        return {
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'tn': self.tn,
            'total': self.total,
            'positives': self.positives,
            'negatives': self.negatives
        }


@dataclass
class MetricResults:
    """Container for all computed metrics."""
    iou: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    confusion: ConfusionMatrix
    threshold: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            'iou': self.iou,
            'f1': self.f1,
            'precision': self.precision,
            'recall': self.recall,
            'accuracy': self.accuracy,
            'threshold': self.threshold,
            'confusion': self.confusion.to_dict()
        }


class SegmentationMetrics:
    """
    Compute segmentation evaluation metrics.
    
    Supports:
    - Single threshold evaluation
    - Threshold sweep
    - Per-tile and aggregate metrics
    - Confusion mask generation for visualization
    """
    
    def __init__(self, eps: float = 1e-7):
        """
        Initialize metrics calculator.
        
        Args:
            eps: Small value to prevent division by zero
        """
        self.eps = eps
    
    def compute_confusion_matrix(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5
    ) -> ConfusionMatrix:
        """
        Compute confusion matrix.
        
        Args:
            prediction: Predicted probabilities or binary mask
            ground_truth: Ground truth binary mask
            threshold: Threshold for converting probabilities to binary
            
        Returns:
            ConfusionMatrix with TP, FP, FN, TN counts
        """
        # Binarize prediction if needed
        if prediction.dtype in [np.float32, np.float64]:
            pred_binary = (prediction >= threshold).astype(bool)
        else:
            pred_binary = prediction.astype(bool)
        
        gt_binary = ground_truth.astype(bool)
        
        # Ensure same shape
        if pred_binary.shape != gt_binary.shape:
            raise ValueError(
                f"Shape mismatch: prediction {pred_binary.shape} vs "
                f"ground truth {gt_binary.shape}"
            )
        
        # Compute confusion matrix
        tp = np.sum(pred_binary & gt_binary)
        fp = np.sum(pred_binary & ~gt_binary)
        fn = np.sum(~pred_binary & gt_binary)
        tn = np.sum(~pred_binary & ~gt_binary)
        
        return ConfusionMatrix(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    
    def compute_iou(self, confusion: ConfusionMatrix) -> float:
        """
        Compute Intersection over Union (Jaccard Index).
        
        IoU = TP / (TP + FP + FN)
        """
        intersection = confusion.tp
        union = confusion.tp + confusion.fp + confusion.fn
        return intersection / (union + self.eps)
    
    def compute_f1(self, confusion: ConfusionMatrix) -> float:
        """
        Compute F1 Score (Dice Coefficient).
        
        F1 = 2 * TP / (2 * TP + FP + FN)
        """
        return (2 * confusion.tp) / (2 * confusion.tp + confusion.fp + confusion.fn + self.eps)
    
    def compute_precision(self, confusion: ConfusionMatrix) -> float:
        """
        Compute Precision.
        
        Precision = TP / (TP + FP)
        """
        return confusion.tp / (confusion.tp + confusion.fp + self.eps)
    
    def compute_recall(self, confusion: ConfusionMatrix) -> float:
        """
        Compute Recall (Sensitivity).
        
        Recall = TP / (TP + FN)
        """
        return confusion.tp / (confusion.tp + confusion.fn + self.eps)
    
    def compute_accuracy(self, confusion: ConfusionMatrix) -> float:
        """
        Compute Pixel Accuracy.
        
        Accuracy = (TP + TN) / Total
        """
        return (confusion.tp + confusion.tn) / (confusion.total + self.eps)
    
    def compute_all_metrics(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5
    ) -> MetricResults:
        """
        Compute all metrics at given threshold.
        
        Args:
            prediction: Predicted probabilities or binary mask
            ground_truth: Ground truth binary mask
            threshold: Threshold for binarization
            
        Returns:
            MetricResults with all metrics
        """
        confusion = self.compute_confusion_matrix(prediction, ground_truth, threshold)
        
        return MetricResults(
            iou=self.compute_iou(confusion),
            f1=self.compute_f1(confusion),
            precision=self.compute_precision(confusion),
            recall=self.compute_recall(confusion),
            accuracy=self.compute_accuracy(confusion),
            confusion=confusion,
            threshold=threshold
        )
    
    def threshold_sweep(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> List[MetricResults]:
        """
        Compute metrics across multiple thresholds.
        
        Useful for:
        - Finding optimal threshold
        - Plotting precision-recall curves
        - Understanding sensitivity to threshold choice
        
        Args:
            prediction: Predicted probabilities
            ground_truth: Ground truth binary mask
            thresholds: List of thresholds (default: 0.1 to 0.9 by 0.05)
            
        Returns:
            List of MetricResults, one per threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05).tolist()
        
        results = []
        for t in thresholds:
            results.append(self.compute_all_metrics(prediction, ground_truth, t))
        
        return results
    
    def find_optimal_threshold(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        metric: str = 'f1',
        thresholds: Optional[List[float]] = None
    ) -> Tuple[float, MetricResults]:
        """
        Find threshold that maximizes a given metric.
        
        Args:
            prediction: Predicted probabilities
            ground_truth: Ground truth binary mask
            metric: Metric to optimize ('f1', 'iou', 'precision', 'recall')
            thresholds: Thresholds to search
            
        Returns:
            Tuple of (optimal threshold, metrics at that threshold)
        """
        results = self.threshold_sweep(prediction, ground_truth, thresholds)
        
        metric_values = [getattr(r, metric) for r in results]
        best_idx = np.argmax(metric_values)
        
        return results[best_idx].threshold, results[best_idx]
    
    def generate_confusion_mask(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Generate pixel-level confusion masks for visualization.
        
        Returns separate binary masks for TP, FP, FN, TN pixels.
        These can be overlaid on the original image with different colors.
        
        Args:
            prediction: Predicted probabilities or binary mask
            ground_truth: Ground truth binary mask
            threshold: Threshold for binarization
            
        Returns:
            Dict with 'tp', 'fp', 'fn', 'tn' boolean arrays
        """
        # Binarize
        if prediction.dtype in [np.float32, np.float64]:
            pred_binary = prediction >= threshold
        else:
            pred_binary = prediction.astype(bool)
        
        gt_binary = ground_truth.astype(bool)
        
        return {
            'tp': pred_binary & gt_binary,
            'fp': pred_binary & ~gt_binary,
            'fn': ~pred_binary & gt_binary,
            'tn': ~pred_binary & ~gt_binary
        }
    
    def generate_error_overlay(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.5,
        colors: Optional[Dict[str, Tuple]] = None
    ) -> np.ndarray:
        """
        Generate RGBA error overlay image.
        
        Args:
            prediction: Predicted probabilities or binary mask
            ground_truth: Ground truth binary mask
            threshold: Threshold for binarization
            colors: Dict mapping 'tp', 'fp', 'fn' to RGBA tuples
            
        Returns:
            RGBA array (H, W, 4) with colored error overlay
        """
        if colors is None:
            colors = {
                'tp': (0, 255, 0, 128),    # Green
                'fp': (255, 0, 0, 128),    # Red
                'fn': (0, 0, 255, 128),    # Blue
                'tn': (0, 0, 0, 0)         # Transparent
            }
        
        masks = self.generate_confusion_mask(prediction, ground_truth, threshold)
        
        height, width = prediction.shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        
        for key, mask in masks.items():
            if key in colors:
                overlay[mask] = colors[key]
        
        return overlay
    
    def aggregate_metrics(
        self,
        results_list: List[MetricResults]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across multiple tiles.
        
        Computes mean, std, min, max for each metric.
        
        Args:
            results_list: List of MetricResults from multiple tiles
            
        Returns:
            Dict with aggregated statistics
        """
        metrics = ['iou', 'f1', 'precision', 'recall', 'accuracy']
        
        aggregated = {}
        for metric in metrics:
            values = [getattr(r, metric) for r in results_list]
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'n': len(values)
            }
        
        # Also aggregate confusion matrix totals
        total_tp = sum(r.confusion.tp for r in results_list)
        total_fp = sum(r.confusion.fp for r in results_list)
        total_fn = sum(r.confusion.fn for r in results_list)
        total_tn = sum(r.confusion.tn for r in results_list)
        
        total_confusion = ConfusionMatrix(
            tp=total_tp, fp=total_fp, fn=total_fn, tn=total_tn
        )
        
        # Compute micro-averaged metrics (based on total counts)
        aggregated['micro'] = {
            'iou': self.compute_iou(total_confusion),
            'f1': self.compute_f1(total_confusion),
            'precision': self.compute_precision(total_confusion),
            'recall': self.compute_recall(total_confusion),
            'accuracy': self.compute_accuracy(total_confusion),
            'confusion': total_confusion.to_dict()
        }
        
        return aggregated


def compute_precision_recall_curve(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    n_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve.
    
    Args:
        prediction: Predicted probabilities
        ground_truth: Ground truth binary mask
        n_thresholds: Number of threshold points
        
    Returns:
        Tuple of (precision array, recall array, threshold array)
    """
    metrics = SegmentationMetrics()
    thresholds = np.linspace(0, 1, n_thresholds)
    
    precisions = []
    recalls = []
    
    for t in thresholds:
        result = metrics.compute_all_metrics(prediction, ground_truth, t)
        precisions.append(result.precision)
        recalls.append(result.recall)
    
    return np.array(precisions), np.array(recalls), thresholds


def compute_roc_curve(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    n_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    
    Args:
        prediction: Predicted probabilities
        ground_truth: Ground truth binary mask
        n_thresholds: Number of threshold points
        
    Returns:
        Tuple of (FPR array, TPR array, threshold array)
    """
    metrics = SegmentationMetrics()
    thresholds = np.linspace(0, 1, n_thresholds)
    
    fprs = []  # False Positive Rate
    tprs = []  # True Positive Rate (Recall)
    
    for t in thresholds:
        confusion = metrics.compute_confusion_matrix(prediction, ground_truth, t)
        
        # TPR = TP / (TP + FN) = Recall
        tpr = confusion.tp / (confusion.tp + confusion.fn + 1e-7)
        # FPR = FP / (FP + TN)
        fpr = confusion.fp / (confusion.fp + confusion.tn + 1e-7)
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    return np.array(fprs), np.array(tprs), thresholds