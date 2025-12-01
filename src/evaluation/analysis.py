"""
Evaluation Analysis Pipeline for Brooklyn Crosswalk QA
=====================================================

Main pipeline that orchestrates:
1. Grid alignment
2. Metric computation (IoU, F1, precision, recall)
3. Calibration analysis (reliability diagram, ECE)
4. Post-processing sweep
5. Operating point selection

This is the entry point for Stage A evaluation.

Usage:
    python analysis.py --config config/config.yaml
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

import numpy as np
import yaml
from tqdm import tqdm

from .alignment import GridAligner
from .metrics import SegmentationMetrics, MetricResults, compute_precision_recall_curve
from .calibration import CalibrationAnalyzer, CalibrationResult, plot_reliability_diagram
from .postprocessing import PostProcessor, OperatingPointSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Complete evaluation pipeline for crosswalk detection QA.
    
    Implements the full Stage A evaluation:
    1. Alignment verification
    2. Per-tile metrics
    3. Threshold sweep
    4. Calibration analysis
    5. Post-processing optimization
    6. Operating point selection
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize evaluation pipeline."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.paths = self.config['paths']
        
        # Initialize components
        self.aligner = GridAligner()
        self.metrics = SegmentationMetrics()
        self.calibrator = CalibrationAnalyzer(
            n_bins=self.config['evaluation']['calibration']['n_bins'],
            strategy=self.config['evaluation']['calibration']['strategy']
        )
        self.post_processor = PostProcessor()
        self.op_selector = OperatingPointSelector()
        
        # Results storage
        self.results = {}
    
    def evaluate_single_tile(
        self,
        prediction_path: Path,
        ground_truth_path: Path,
        tile_id: str = None
    ) -> Dict:
        """
        Evaluate a single tile.
        
        Args:
            prediction_path: Path to prediction probability map
            ground_truth_path: Path to ground truth mask
            tile_id: Optional tile identifier
            
        Returns:
            Dict with all evaluation results for this tile
        """
        tile_id = tile_id or prediction_path.stem
        
        logger.info(f"Evaluating tile: {tile_id}")
        
        # Load and align
        pred, gt, metadata = self.aligner.load_aligned_pair(
            prediction_path, ground_truth_path
        )
        
        # Ensure prediction is probability (0-1 range)
        if pred.max() > 1:
            pred = pred.astype(np.float32) / 255.0
        
        # Ensure ground truth is binary
        gt = (gt > 0).astype(np.uint8)
        
        results = {
            'tile_id': tile_id,
            'metadata': metadata,
            'prediction_path': str(prediction_path),
            'ground_truth_path': str(ground_truth_path)
        }
        
        # 1. Metrics at default threshold
        default_threshold = 0.5
        default_metrics = self.metrics.compute_all_metrics(pred, gt, default_threshold)
        results['metrics_default'] = default_metrics.to_dict()
        
        # 2. Threshold sweep
        thresholds = np.arange(
            self.config['evaluation']['thresholds']['start'],
            self.config['evaluation']['thresholds']['end'],
            self.config['evaluation']['thresholds']['step']
        ).tolist()
        
        threshold_results = self.metrics.threshold_sweep(pred, gt, thresholds)
        results['threshold_sweep'] = [r.to_dict() for r in threshold_results]
        
        # Find optimal threshold
        optimal_t, optimal_metrics = self.metrics.find_optimal_threshold(
            pred, gt, metric='f1', thresholds=thresholds
        )
        results['optimal_threshold'] = {
            'threshold': optimal_t,
            'metrics': optimal_metrics.to_dict()
        }
        
        # 3. Calibration analysis
        calibration = self.calibrator.compute_calibration(pred, gt)
        results['calibration'] = calibration.to_dict()
        
        # 4. Risk-coverage
        risk_coverage = self.calibrator.compute_risk_coverage(pred, gt, metric='f1')
        results['risk_coverage'] = {
            'coverage': risk_coverage['coverage'].tolist(),
            'f1': risk_coverage['f1'].tolist(),
            'thresholds': risk_coverage['thresholds'].tolist()
        }
        
        # 5. Post-processing analysis
        morph_analysis = self.post_processor.analyze_morphology_effect(
            pred, gt, threshold=optimal_t
        )
        results['morphology_analysis'] = morph_analysis
        
        # 6. Generate confusion mask for visualization
        confusion_mask = self.metrics.generate_confusion_mask(pred, gt, optimal_t)
        results['confusion_counts'] = {
            'tp_pixels': int(np.sum(confusion_mask['tp'])),
            'fp_pixels': int(np.sum(confusion_mask['fp'])),
            'fn_pixels': int(np.sum(confusion_mask['fn'])),
            'tn_pixels': int(np.sum(confusion_mask['tn']))
        }
        
        return results
    
    def evaluate_all_tiles(
        self,
        predictions_dir: Path,
        ground_truth_dir: Path,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Evaluate all tiles in directories.
        
        Args:
            predictions_dir: Directory containing prediction files
            ground_truth_dir: Directory containing ground truth files
            output_dir: Directory for output results
            
        Returns:
            Dict with aggregated results
        """
        predictions_dir = Path(predictions_dir)
        ground_truth_dir = Path(ground_truth_dir)
        output_dir = Path(output_dir or self.paths['outputs']['metrics'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find prediction files
        pred_files = list(predictions_dir.glob("*_prob.tif"))
        if not pred_files:
            pred_files = list(predictions_dir.glob("*.tif"))
        
        logger.info(f"Found {len(pred_files)} prediction files")
        
        all_tile_results = []
        all_metric_results = []
        all_calibration_results = []
        
        for pred_path in tqdm(pred_files, desc="Evaluating tiles"):
            # Find matching ground truth
            # Try different naming conventions
            tile_stem = pred_path.stem.replace('_prob', '').replace('_mask', '')
            
            gt_candidates = [
                ground_truth_dir / f"{tile_stem}_gt.tif",
                ground_truth_dir / f"{tile_stem}.tif",
                ground_truth_dir / f"{pred_path.stem}_gt.tif"
            ]
            
            gt_path = None
            for candidate in gt_candidates:
                if candidate.exists():
                    gt_path = candidate
                    break
            
            if gt_path is None:
                logger.warning(f"No ground truth found for {pred_path.name}")
                continue
            
            try:
                tile_result = self.evaluate_single_tile(pred_path, gt_path, tile_stem)
                all_tile_results.append(tile_result)
                
                # Collect for aggregation
                all_metric_results.append(MetricResults(
                    iou=tile_result['metrics_default']['iou'],
                    f1=tile_result['metrics_default']['f1'],
                    precision=tile_result['metrics_default']['precision'],
                    recall=tile_result['metrics_default']['recall'],
                    accuracy=tile_result['metrics_default']['accuracy'],
                    confusion=None,  # We don't need this for aggregation
                    threshold=0.5
                ))
                
                all_calibration_results.append(CalibrationResult(
                    bins=[],  # Simplified
                    ece=tile_result['calibration']['ece'],
                    mce=tile_result['calibration']['mce'],
                    reliability_diagram=tile_result['calibration'],
                    n_samples=tile_result['calibration']['n_samples']
                ))
                
            except Exception as e:
                logger.error(f"Failed to evaluate {pred_path.name}: {e}")
        
        # Aggregate results
        aggregated = self._aggregate_results(
            all_tile_results, all_metric_results, all_calibration_results
        )
        
        # Save results
        self._save_results(all_tile_results, aggregated, output_dir)
        
        return aggregated
    
    def _aggregate_results(
        self,
        tile_results: List[Dict],
        metric_results: List[MetricResults],
        calibration_results: List[CalibrationResult]
    ) -> Dict:
        """Aggregate results across all tiles."""
        
        if not tile_results:
            return {"error": "No tiles evaluated"}
        
        # Aggregate metrics
        aggregated_metrics = self.metrics.aggregate_metrics(
            [MetricResults(
                iou=r.iou,
                f1=r.f1,
                precision=r.precision,
                recall=r.recall,
                accuracy=r.accuracy,
                confusion=type('obj', (object,), {
                    'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                    'to_dict': lambda s: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
                })(),
                threshold=r.threshold
            ) for r in metric_results]
        )
        
        # Aggregate calibration
        avg_ece = np.mean([c.ece for c in calibration_results])
        avg_mce = np.mean([c.mce for c in calibration_results])
        
        # Find overall optimal threshold
        all_thresholds = []
        for tile in tile_results:
            all_thresholds.append(tile['optimal_threshold']['threshold'])
        
        recommended_threshold = np.median(all_thresholds)
        
        return {
            'summary': {
                'n_tiles': len(tile_results),
                'evaluation_date': datetime.now().isoformat()
            },
            'metrics': aggregated_metrics,
            'calibration': {
                'mean_ece': float(avg_ece),
                'mean_mce': float(avg_mce),
                'ece_std': float(np.std([c.ece for c in calibration_results]))
            },
            'recommendations': {
                'optimal_threshold': float(recommended_threshold),
                'threshold_std': float(np.std(all_thresholds))
            }
        }
    
    def _save_results(
        self,
        tile_results: List[Dict],
        aggregated: Dict,
        output_dir: Path
    ):
        """Save evaluation results to files."""
        
        # Save per-tile results
        tiles_path = output_dir / "tile_results.json"
        with open(tiles_path, 'w') as f:
            json.dump(tile_results, f, indent=2, default=str)
        
        # Save aggregated results
        agg_path = output_dir / "aggregated_results.json"
        with open(agg_path, 'w') as f:
            json.dump(aggregated, f, indent=2, default=str)
        
        # Save summary CSV
        import pandas as pd
        
        summary_data = []
        for tile in tile_results:
            summary_data.append({
                'tile_id': tile['tile_id'],
                'iou': tile['metrics_default']['iou'],
                'f1': tile['metrics_default']['f1'],
                'precision': tile['metrics_default']['precision'],
                'recall': tile['metrics_default']['recall'],
                'ece': tile['calibration']['ece'],
                'optimal_threshold': tile['optimal_threshold']['threshold'],
                'tp_pixels': tile['confusion_counts']['tp_pixels'],
                'fp_pixels': tile['confusion_counts']['fp_pixels'],
                'fn_pixels': tile['confusion_counts']['fn_pixels']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_dir / "tile_summary.csv", index=False)
        
        logger.info(f"Results saved to {output_dir}")
    
    def select_operating_point(
        self,
        predictions_dir: Path,
        ground_truth_dir: Path,
        objective: str = 'balanced'
    ) -> Dict:
        """
        Select optimal operating point across all tiles.
        
        Args:
            predictions_dir: Directory containing predictions
            ground_truth_dir: Directory containing ground truth
            objective: 'balanced', 'high_precision', or 'high_recall'
            
        Returns:
            Dict with recommended operating point and justification
        """
        # Run full evaluation first
        aggregated = self.evaluate_all_tiles(predictions_dir, ground_truth_dir)
        
        # Get operating point recommendation
        recommendation = {
            'threshold': aggregated['recommendations']['optimal_threshold'],
            'objective': objective,
            'expected_metrics': aggregated['metrics'],
            'calibration': aggregated['calibration']
        }
        
        # Add morphology recommendation based on majority voting
        # (Would require storing and aggregating morphology results)
        
        return recommendation
    
    def generate_evaluation_report(
        self,
        results_dir: Path,
        output_path: Path
    ) -> Path:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results_dir: Directory with evaluation results
            output_path: Path for output report
            
        Returns:
            Path to generated report
        """
        import pandas as pd
        
        results_dir = Path(results_dir)
        output_path = Path(output_path)
        
        # Load results
        with open(results_dir / "aggregated_results.json", 'r') as f:
            aggregated = json.load(f)
        
        tile_df = pd.read_csv(results_dir / "tile_summary.csv")
        
        # Generate report
        report_lines = [
            "# Brooklyn Crosswalk Detection - Evaluation Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- **Tiles evaluated**: {aggregated['summary']['n_tiles']}",
            f"- **Mean IoU**: {aggregated['metrics']['iou']['mean']:.4f} (±{aggregated['metrics']['iou']['std']:.4f})",
            f"- **Mean F1**: {aggregated['metrics']['f1']['mean']:.4f} (±{aggregated['metrics']['f1']['std']:.4f})",
            f"- **Mean Precision**: {aggregated['metrics']['precision']['mean']:.4f}",
            f"- **Mean Recall**: {aggregated['metrics']['recall']['mean']:.4f}",
            "",
            "## Calibration",
            f"- **Mean ECE**: {aggregated['calibration']['mean_ece']:.4f}",
            f"- **Mean MCE**: {aggregated['calibration']['mean_mce']:.4f}",
            "",
            "## Recommendations",
            f"- **Optimal Threshold**: {aggregated['recommendations']['optimal_threshold']:.2f}",
            "",
            "## Per-Tile Statistics",
            "",
            tile_df.describe().to_markdown(),
            "",
            "## Notes",
            "- IoU (Intersection over Union) measures overlap between prediction and ground truth",
            "- F1 Score balances precision and recall",
            "- ECE (Expected Calibration Error) measures how well probabilities match actual frequencies",
            "- Lower ECE indicates better calibration"
        ]
        
        report_text = "\n".join(report_lines)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {output_path}")
        
        return output_path


def main():
    """CLI for evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file')
    parser.add_argument('--predictions', '-p', required=True, help='Predictions directory')
    parser.add_argument('--ground-truth', '-g', required=True, help='Ground truth directory')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--objective', default='balanced',
                       choices=['balanced', 'high_precision', 'high_recall'],
                       help='Optimization objective')
    
    args = parser.parse_args()
    
    # Run evaluation
    pipeline = EvaluationPipeline(args.config)
    
    output_dir = Path(args.output) if args.output else Path(pipeline.paths['outputs']['metrics'])
    
    results = pipeline.evaluate_all_tiles(
        Path(args.predictions),
        Path(args.ground_truth),
        output_dir
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Tiles evaluated: {results['summary']['n_tiles']}")
    print(f"Mean IoU: {results['metrics']['iou']['mean']:.4f}")
    print(f"Mean F1: {results['metrics']['f1']['mean']:.4f}")
    print(f"Mean ECE: {results['calibration']['mean_ece']:.4f}")
    print(f"Recommended threshold: {results['recommendations']['optimal_threshold']:.2f}")
    
    if args.report:
        report_path = output_dir / "evaluation_report.md"
        pipeline.generate_evaluation_report(output_dir, report_path)
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()