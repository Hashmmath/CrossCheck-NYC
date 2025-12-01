#!/usr/bin/env python
"""
Brooklyn Crosswalk Detection QA - Main Runner
=============================================

Complete pipeline runner for Stage A evaluation.

Steps:
1. Download data (orthoimagery, LION, OSM)
2. Preprocess tiles
3. Run inference (or use simulated for testing)
4. Generate ground truth
5. Run evaluation
6. Generate reports

Usage:
    python run_pipeline.py --all                    # Run complete pipeline
    python run_pipeline.py --step download          # Run specific step
    python run_pipeline.py --step evaluate          # Just run evaluation
    python run_pipeline.py --test                   # Test mode with simulated data
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def step_download(config: dict, test_mode: bool = False):
    """Step 1: Download all required data."""
    logger.info("=" * 60)
    logger.info("STEP 1: DOWNLOADING DATA")
    logger.info("=" * 60)
    
    from src.data.download import DataDownloader
    
    downloader = DataDownloader()
    
    # Download with test area for faster iteration
    results = downloader.download_all(
        use_test_area=True,
        skip_ortho=test_mode  # Skip large ortho download in test mode
    )
    
    return results


def step_preprocess(config: dict):
    """Step 2: Preprocess tiles."""
    logger.info("=" * 60)
    logger.info("STEP 2: PREPROCESSING TILES")
    logger.info("=" * 60)
    
    from src.data.preprocessing import TilePreprocessor
    
    preprocessor = TilePreprocessor()
    
    input_dir = Path(config['paths']['raw']['ortho'])
    output_dir = Path(config['paths']['processed']['tiles'])
    
    if not input_dir.exists() or not list(input_dir.glob("*.png")):
        logger.warning("No tiles found to preprocess. Run download step first.")
        return None
    
    results = preprocessor.process_directory(
        input_dir,
        output_dir,
        stitch=True,
        tile_step=2
    )
    
    return results


def step_inference(config: dict, simulate: bool = False):
    """Step 3: Run inference."""
    logger.info("=" * 60)
    logger.info("STEP 3: RUNNING INFERENCE")
    logger.info("=" * 60)
    
    if simulate:
        from src.inference.tile2net_wrapper import SimulatedTile2NetInference
        inference = SimulatedTile2NetInference()
        logger.info("Using SIMULATED inference (no GPU required)")
    else:
        from src.inference.tile2net_wrapper import Tile2NetInference
        inference = Tile2NetInference()
    
    input_dir = Path(config['paths']['processed']['tiles'])
    output_dir = Path(config['paths']['processed']['predictions'])
    
    # Check for stitched tiles first
    stitched_dir = input_dir / "stitched"
    if stitched_dir.exists():
        input_dir = stitched_dir
    
    if not input_dir.exists():
        logger.warning("No preprocessed tiles found. Run preprocess step first.")
        return None
    
    results = inference.process_tile_directory(
        input_dir,
        output_dir,
        save_probabilities=True,
        save_masks=True,
        threshold=0.5
    )
    
    return results


def step_ground_truth(config: dict):
    """Step 4: Generate ground truth masks."""
    logger.info("=" * 60)
    logger.info("STEP 4: GENERATING GROUND TRUTH")
    logger.info("=" * 60)
    
    from src.data.ground_truth import GroundTruthGenerator
    
    generator = GroundTruthGenerator()
    
    # Find tile manifest
    tiles_dir = Path(config['paths']['processed']['tiles'])
    manifest_path = tiles_dir / "processing_manifest.json"
    
    if not manifest_path.exists():
        # Try stitched directory
        manifest_path = tiles_dir / "stitched" / "processing_manifest.json"
    
    if not manifest_path.exists():
        logger.warning("No tile manifest found. Run preprocess step first.")
        return None
    
    output_dir = Path(config['paths']['processed']['ground_truth'])
    
    masks = generator.generate_ground_truth_for_tiles(
        manifest_path,
        output_dir
    )
    
    return masks


def step_evaluate(config: dict):
    """Step 5: Run evaluation."""
    logger.info("=" * 60)
    logger.info("STEP 5: RUNNING EVALUATION")
    logger.info("=" * 60)
    
    from src.evaluation.analysis import EvaluationPipeline
    
    pipeline = EvaluationPipeline()
    
    predictions_dir = Path(config['paths']['processed']['predictions']) / "probabilities"
    ground_truth_dir = Path(config['paths']['processed']['ground_truth'])
    output_dir = Path(config['paths']['outputs']['metrics'])
    
    if not predictions_dir.exists():
        # Try direct predictions directory
        predictions_dir = Path(config['paths']['processed']['predictions'])
    
    if not predictions_dir.exists():
        logger.warning("No predictions found. Run inference step first.")
        return None
    
    if not ground_truth_dir.exists():
        logger.warning("No ground truth found. Run ground_truth step first.")
        return None
    
    results = pipeline.evaluate_all_tiles(
        predictions_dir,
        ground_truth_dir,
        output_dir
    )
    
    # Generate report
    report_path = output_dir / "evaluation_report.md"
    pipeline.generate_evaluation_report(output_dir, report_path)
    
    return results


def step_visualize(config: dict):
    """Step 6: Generate visualizations."""
    logger.info("=" * 60)
    logger.info("STEP 6: GENERATING VISUALIZATIONS")
    logger.info("=" * 60)
    
    from src.utils.viz import generate_tile_preview, VisualizationUtils
    
    predictions_dir = Path(config['paths']['processed']['predictions'])
    ground_truth_dir = Path(config['paths']['processed']['ground_truth'])
    tiles_dir = Path(config['paths']['processed']['tiles'])
    output_dir = Path(config['paths']['outputs']['figures'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    prob_files = list((predictions_dir / "probabilities").glob("*_prob.tif"))
    if not prob_files:
        prob_files = list(predictions_dir.glob("*_prob.tif"))
    
    if not prob_files:
        logger.warning("No prediction files found for visualization.")
        return None
    
    # Generate previews for first few tiles
    for pred_path in prob_files[:5]:  # Limit to 5 for speed
        tile_stem = pred_path.stem.replace('_prob', '')
        
        # Find original tile
        tile_candidates = [
            tiles_dir / "stitched" / f"{tile_stem}.tif",
            tiles_dir / f"{tile_stem}.tif",
            tiles_dir / f"{tile_stem}.png"
        ]
        
        tile_path = None
        for candidate in tile_candidates:
            if candidate.exists():
                tile_path = candidate
                break
        
        # Find ground truth
        gt_candidates = [
            ground_truth_dir / f"{tile_stem}_gt.tif",
            ground_truth_dir / f"{tile_stem}.tif"
        ]
        
        gt_path = None
        for candidate in gt_candidates:
            if candidate.exists():
                gt_path = candidate
                break
        
        if tile_path and gt_path:
            output_path = output_dir / f"{tile_stem}_preview.png"
            try:
                generate_tile_preview(
                    tile_path, pred_path, gt_path, output_path
                )
            except Exception as e:
                logger.error(f"Failed to generate preview for {tile_stem}: {e}")
    
    logger.info(f"Visualizations saved to {output_dir}")
    return output_dir


def run_full_pipeline(config: dict, test_mode: bool = False):
    """Run the complete pipeline."""
    logger.info("=" * 60)
    logger.info("BROOKLYN CROSSWALK DETECTION QA - FULL PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    results = {}
    
    # Step 1: Download
    try:
        results['download'] = step_download(config, test_mode)
    except Exception as e:
        logger.error(f"Download step failed: {e}")
        results['download'] = {'error': str(e)}
    
    # Step 2: Preprocess
    try:
        results['preprocess'] = step_preprocess(config)
    except Exception as e:
        logger.error(f"Preprocess step failed: {e}")
        results['preprocess'] = {'error': str(e)}
    
    # Step 3: Inference
    try:
        results['inference'] = step_inference(config, simulate=test_mode)
    except Exception as e:
        logger.error(f"Inference step failed: {e}")
        results['inference'] = {'error': str(e)}
    
    # Step 4: Ground Truth
    try:
        results['ground_truth'] = step_ground_truth(config)
    except Exception as e:
        logger.error(f"Ground truth step failed: {e}")
        results['ground_truth'] = {'error': str(e)}
    
    # Step 5: Evaluate
    try:
        results['evaluate'] = step_evaluate(config)
    except Exception as e:
        logger.error(f"Evaluation step failed: {e}")
        results['evaluate'] = {'error': str(e)}
    
    # Step 6: Visualize
    try:
        results['visualize'] = step_visualize(config)
    except Exception as e:
        logger.error(f"Visualization step failed: {e}")
        results['visualize'] = {'error': str(e)}
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Brooklyn Crosswalk Detection QA Pipeline"
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run complete pipeline'
    )
    
    parser.add_argument(
        '--step', '-s',
        choices=['download', 'preprocess', 'inference', 'ground_truth', 'evaluate', 'visualize'],
        help='Run specific step only'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test mode (simulated inference, skip large downloads)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.all:
        run_full_pipeline(config, test_mode=args.test)
    elif args.step:
        step_functions = {
            'download': step_download,
            'preprocess': step_preprocess,
            'inference': lambda c: step_inference(c, simulate=args.test),
            'ground_truth': step_ground_truth,
            'evaluate': step_evaluate,
            'visualize': step_visualize
        }
        
        step_fn = step_functions[args.step]
        if args.step == 'download':
            step_fn(config, test_mode=args.test)
        else:
            step_fn(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()