"""
Grid Alignment for Brooklyn Crosswalk QA
========================================

Ensures ground truth and predictions are perfectly aligned
for accurate pixel-level evaluation.

Handles:
- CRS reprojection
- Resolution matching
- Extent alignment
- Validation/verification
"""

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds, Affine
from rasterio.warp import reproject, Resampling, calculate_default_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GridAligner:
    """
    Align ground truth and prediction grids for evaluation.
    
    Ensures both rasters have identical:
    - CRS (coordinate reference system)
    - Extent (bounding box)
    - Resolution (pixel size)
    - Grid registration (pixel alignment)
    """
    
    def __init__(self):
        """Initialize grid aligner."""
        pass
    
    def get_raster_info(self, path: Path) -> Dict:
        """
        Get raster metadata.
        
        Args:
            path: Path to raster file
            
        Returns:
            Dict with raster properties
        """
        with rasterio.open(path) as src:
            return {
                'path': str(path),
                'crs': str(src.crs),
                'bounds': dict(src.bounds._asdict()),
                'transform': list(src.transform),
                'width': src.width,
                'height': src.height,
                'resolution': src.res,
                'dtype': str(src.dtypes[0]),
                'count': src.count
            }
    
    def check_alignment(
        self,
        raster1_path: Path,
        raster2_path: Path,
        tolerance: float = 1e-6
    ) -> Dict:
        """
        Check if two rasters are aligned.
        
        Args:
            raster1_path: Path to first raster
            raster2_path: Path to second raster
            tolerance: Tolerance for floating point comparisons
            
        Returns:
            Dict with alignment check results
        """
        info1 = self.get_raster_info(raster1_path)
        info2 = self.get_raster_info(raster2_path)
        
        # Check CRS
        crs_match = info1['crs'] == info2['crs']
        
        # Check bounds
        bounds_match = all(
            abs(info1['bounds'][k] - info2['bounds'][k]) < tolerance
            for k in ['left', 'bottom', 'right', 'top']
        )
        
        # Check resolution
        res_match = all(
            abs(info1['resolution'][i] - info2['resolution'][i]) < tolerance
            for i in range(2)
        )
        
        # Check dimensions
        dims_match = (
            info1['width'] == info2['width'] and
            info1['height'] == info2['height']
        )
        
        # Check transform (pixel registration)
        transform_match = all(
            abs(info1['transform'][i] - info2['transform'][i]) < tolerance
            for i in range(6)
        )
        
        aligned = crs_match and bounds_match and res_match and dims_match and transform_match
        
        return {
            'aligned': aligned,
            'crs_match': crs_match,
            'bounds_match': bounds_match,
            'resolution_match': res_match,
            'dimensions_match': dims_match,
            'transform_match': transform_match,
            'raster1': info1,
            'raster2': info2
        }
    
    def align_to_reference(
        self,
        source_path: Path,
        reference_path: Path,
        output_path: Path,
        resampling: str = 'nearest'
    ) -> Path:
        """
        Align source raster to match reference raster grid.
        
        Args:
            source_path: Path to source raster (to be aligned)
            reference_path: Path to reference raster (target grid)
            output_path: Path for aligned output
            resampling: Resampling method ('nearest', 'bilinear', 'cubic')
            
        Returns:
            Path to aligned raster
        """
        resampling_methods = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic
        }
        
        # Get reference properties
        with rasterio.open(reference_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
        
        # Read source and reproject
        with rasterio.open(source_path) as src:
            source_data = src.read(1)
            source_crs = src.crs
            source_transform = src.transform
            
            # Create output array
            aligned_data = np.zeros((ref_height, ref_width), dtype=source_data.dtype)
            
            # Reproject
            reproject(
                source=source_data,
                destination=aligned_data,
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling_methods.get(resampling, Resampling.nearest)
            )
        
        # Save aligned raster
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=ref_height,
            width=ref_width,
            count=1,
            dtype=aligned_data.dtype,
            crs=ref_crs,
            transform=ref_transform,
            compress='lzw'
        ) as dst:
            dst.write(aligned_data, 1)
        
        logger.info(f"Aligned raster saved to {output_path}")
        
        return output_path
    
    def load_aligned_pair(
        self,
        prediction_path: Path,
        ground_truth_path: Path,
        align_gt_to_pred: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load prediction and ground truth as aligned arrays.
        
        Automatically aligns if needed.
        
        Args:
            prediction_path: Path to prediction raster
            ground_truth_path: Path to ground truth raster
            align_gt_to_pred: If True, align GT to prediction grid
            
        Returns:
            Tuple of (prediction array, ground truth array, metadata dict)
        """
        # Check alignment
        alignment = self.check_alignment(prediction_path, ground_truth_path)
        
        if alignment['aligned']:
            # Already aligned, just load
            with rasterio.open(prediction_path) as src:
                pred = src.read(1)
                metadata = {
                    'crs': str(src.crs),
                    'transform': list(src.transform),
                    'bounds': dict(src.bounds._asdict())
                }
            
            with rasterio.open(ground_truth_path) as src:
                gt = src.read(1)
            
            return pred, gt, metadata
        
        # Need to align
        logger.info("Rasters not aligned, performing alignment...")
        
        if align_gt_to_pred:
            # Align GT to prediction grid
            aligned_gt_path = ground_truth_path.parent / f"{ground_truth_path.stem}_aligned.tif"
            self.align_to_reference(
                ground_truth_path, prediction_path, aligned_gt_path
            )
            
            with rasterio.open(prediction_path) as src:
                pred = src.read(1)
                metadata = {
                    'crs': str(src.crs),
                    'transform': list(src.transform),
                    'bounds': dict(src.bounds._asdict()),
                    'aligned_from': str(ground_truth_path)
                }
            
            with rasterio.open(aligned_gt_path) as src:
                gt = src.read(1)
        else:
            # Align prediction to GT grid
            aligned_pred_path = prediction_path.parent / f"{prediction_path.stem}_aligned.tif"
            self.align_to_reference(
                prediction_path, ground_truth_path, aligned_pred_path
            )
            
            with rasterio.open(aligned_pred_path) as src:
                pred = src.read(1)
            
            with rasterio.open(ground_truth_path) as src:
                gt = src.read(1)
                metadata = {
                    'crs': str(src.crs),
                    'transform': list(src.transform),
                    'bounds': dict(src.bounds._asdict()),
                    'aligned_from': str(prediction_path)
                }
        
        return pred, gt, metadata
    
    def create_common_grid(
        self,
        rasters: list,
        output_crs: Optional[str] = None,
        output_resolution: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        Create a common grid specification for multiple rasters.
        
        Args:
            rasters: List of raster paths
            output_crs: Target CRS (default: use first raster's CRS)
            output_resolution: Target resolution (default: finest resolution)
            
        Returns:
            Dict with common grid specification
        """
        # Collect info from all rasters
        infos = [self.get_raster_info(Path(r)) for r in rasters]
        
        # Determine output CRS
        if output_crs is None:
            output_crs = infos[0]['crs']
        
        # Determine output resolution (use finest)
        if output_resolution is None:
            resolutions = [info['resolution'] for info in infos]
            output_resolution = (
                min(r[0] for r in resolutions),
                min(r[1] for r in resolutions)
            )
        
        # Determine combined bounds
        all_bounds = [info['bounds'] for info in infos]
        combined_bounds = {
            'left': min(b['left'] for b in all_bounds),
            'bottom': min(b['bottom'] for b in all_bounds),
            'right': max(b['right'] for b in all_bounds),
            'top': max(b['top'] for b in all_bounds)
        }
        
        # Calculate dimensions
        width = int(np.ceil(
            (combined_bounds['right'] - combined_bounds['left']) / output_resolution[0]
        ))
        height = int(np.ceil(
            (combined_bounds['top'] - combined_bounds['bottom']) / output_resolution[1]
        ))
        
        # Create transform
        transform = from_bounds(
            combined_bounds['left'],
            combined_bounds['bottom'],
            combined_bounds['right'],
            combined_bounds['top'],
            width,
            height
        )
        
        return {
            'crs': output_crs,
            'resolution': output_resolution,
            'bounds': combined_bounds,
            'width': width,
            'height': height,
            'transform': list(transform)
        }
    
    def resample_to_grid(
        self,
        source_path: Path,
        grid_spec: Dict,
        output_path: Path,
        resampling: str = 'nearest'
    ) -> Path:
        """
        Resample raster to specified grid.
        
        Args:
            source_path: Path to source raster
            grid_spec: Grid specification dict
            output_path: Output path
            resampling: Resampling method
            
        Returns:
            Path to resampled raster
        """
        resampling_methods = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic
        }
        
        target_crs = CRS.from_string(grid_spec['crs'])
        target_transform = Affine(*grid_spec['transform'][:6])
        target_width = grid_spec['width']
        target_height = grid_spec['height']
        
        with rasterio.open(source_path) as src:
            source_data = src.read(1)
            
            resampled = np.zeros((target_height, target_width), dtype=source_data.dtype)
            
            reproject(
                source=source_data,
                destination=resampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=resampling_methods.get(resampling, Resampling.nearest)
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=target_height,
            width=target_width,
            count=1,
            dtype=resampled.dtype,
            crs=target_crs,
            transform=target_transform,
            compress='lzw'
        ) as dst:
            dst.write(resampled, 1)
        
        return output_path


def visualize_alignment(
    pred_path: Path,
    gt_path: Path,
    output_path: Path,
    block_size: int = 32
):
    """
    Create checkerboard visualization to verify alignment.
    
    Alternates blocks between prediction and ground truth edges
    to visually confirm alignment.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for alignment visualization")
        return
    
    from PIL import Image
    
    # Read rasters
    with rasterio.open(pred_path) as src:
        pred = src.read(1)
    
    with rasterio.open(gt_path) as src:
        gt = src.read(1)
    
    if pred.shape != gt.shape:
        logger.error("Cannot visualize: shapes don't match")
        return
    
    height, width = pred.shape
    
    # Create edge images
    pred_norm = ((pred - pred.min()) / (pred.max() - pred.min() + 1e-8) * 255).astype(np.uint8)
    gt_norm = ((gt - gt.min()) / (gt.max() - gt.min() + 1e-8) * 255).astype(np.uint8)
    
    pred_edges = cv2.Canny(pred_norm, 50, 150)
    gt_edges = cv2.Canny(gt_norm, 50, 150)
    
    # Create checkerboard
    checkerboard = np.zeros((height, width), dtype=bool)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                checkerboard[i:i+block_size, j:j+block_size] = True
    
    # Combine
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[checkerboard, 0] = gt_edges[checkerboard]  # Red for GT
    overlay[~checkerboard, 2] = pred_edges[~checkerboard]  # Blue for pred
    
    Image.fromarray(overlay).save(output_path)
    logger.info(f"Alignment visualization saved to {output_path}")