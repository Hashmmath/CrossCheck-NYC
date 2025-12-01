"""
Tile2Net Inference Wrapper for Brooklyn Crosswalk QA
====================================================

Wraps Tile2Net model for inference with emphasis on:
1. Getting per-pixel PROBABILITIES (not just binary masks)
2. Extracting crosswalk class specifically
3. Batch processing of tiles
4. Saving probability maps for calibration analysis

Usage:
    python tile2net_wrapper.py --config config/config.yaml --input data/processed/tiles
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tile2NetInference:
    """
    Wrapper for Tile2Net semantic segmentation model.
    
    Key features:
    - Returns per-pixel probabilities for calibration analysis
    - Extracts crosswalk class specifically
    - Preserves georeferencing information
    - Supports batch processing
    """
    
    # Tile2Net class indices (based on model documentation)
    CLASSES = {
        0: 'background',
        1: 'road',
        2: 'sidewalk',
        3: 'crosswalk',
        4: 'footpath'
    }
    
    CROSSWALK_CLASS_IDX = 3
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        device: Optional[str] = None
    ):
        """
        Initialize Tile2Net inference wrapper.
        
        Args:
            config_path: Path to configuration file
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Model settings
        self.tile_size = self.config['tile2net']['inference']['tile_size']
        self.batch_size = self.config['tile2net']['inference']['batch_size']
        self.target_class = self.config['tile2net']['target_class']
        
        # Model will be loaded lazily
        self.model = None
        self._model_loaded = False
    
    def load_model(self, weights_path: Optional[str] = None):
        """
        Load Tile2Net model weights.
        
        Args:
            weights_path: Path to model weights (or directory containing them)
        """
        if self._model_loaded:
            logger.info("Model already loaded")
            return
        
        weights_path = weights_path or self.config['tile2net']['model']['weights_path']
        weights_path = Path(weights_path)
        
        # Find weight file
        if weights_path.is_dir():
            weight_files = list(weights_path.glob("*.pth")) + list(weights_path.glob("*.pt"))
            if not weight_files:
                raise FileNotFoundError(f"No weight files found in {weights_path}")
            weights_path = weight_files[0]
        
        logger.info(f"Loading model from {weights_path}")
        
        try:
            # Try to import tile2net and use its model loading
            from tile2net.tileseg.inference.inference import Inference
            
            # Create a temporary inference object to get the model
            self._tile2net_inference = Inference.__new__(Inference)
            self._tile2net_inference.device = self.device
            
            # Load model architecture and weights
            # This depends on tile2net's internal structure
            self._load_tile2net_model(weights_path)
            
        except ImportError:
            logger.warning("tile2net not installed, using standalone model loader")
            self._load_standalone_model(weights_path)
        
        self._model_loaded = True
        logger.info("Model loaded successfully")
    
    def _load_tile2net_model(self, weights_path: Path):
        """Load model using tile2net's built-in loader."""
        try:
            # Import tile2net components
            from tile2net.tileseg.model import model_factory
            
            # Create model (DeepLabV3+ with ResNet backbone)
            self.model = model_factory.get_model(
                'deeplabv3plus_resnet101',
                num_classes=5,  # background, road, sidewalk, crosswalk, footpath
                pretrained=False
            )
            
            # Load weights
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.warning(f"Failed to load via tile2net: {e}")
            self._load_standalone_model(weights_path)
    
    def _load_standalone_model(self, weights_path: Path):
        """Load model without tile2net dependency."""
        try:
            import torchvision.models.segmentation as seg_models
            
            # Create DeepLabV3+ with ResNet101 backbone
            self.model = seg_models.deeplabv3_resnet101(
                weights=None,
                num_classes=5
            )
            
            # Load weights
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Try to load (may need to adjust keys)
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                # Try without strict matching
                self.model.load_state_dict(state_dict, strict=False)
                logger.warning("Loaded weights with strict=False")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array (H, W, C) in [0, 255] or [0, 1]
            
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Ensure float
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor (B, C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        
        return tensor.to(self.device)
    
    def predict_probabilities(
        self,
        image: Union[np.ndarray, Path],
        return_all_classes: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get per-pixel probabilities for input image.
        
        This is the key method for calibration analysis - we need
        probabilities, not just argmax predictions.
        
        Args:
            image: Input image (array or path)
            return_all_classes: Whether to return all class probabilities
            
        Returns:
            If return_all_classes: Dict mapping class names to probability arrays
            Otherwise: Crosswalk probability array (H, W) in [0, 1]
        """
        if not self._model_loaded:
            self.load_model()
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Get original size
        orig_h, orig_w = image.shape[:2]
        
        # Resize for model if needed
        if orig_h != self.tile_size or orig_w != self.tile_size:
            input_tensor = F.interpolate(
                input_tensor,
                size=(self.tile_size, self.tile_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Handle different output formats
            if isinstance(output, dict):
                logits = output['out']
            else:
                logits = output
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
        
        # Resize back to original size
        if orig_h != self.tile_size or orig_w != self.tile_size:
            probs = F.interpolate(
                probs,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )
        
        # Convert to numpy
        probs_np = probs.squeeze(0).cpu().numpy()
        
        if return_all_classes:
            return {
                name: probs_np[idx]
                for idx, name in self.CLASSES.items()
            }
        else:
            # Return only crosswalk probabilities
            return probs_np[self.CROSSWALK_CLASS_IDX]
    
    def predict_mask(
        self,
        image: Union[np.ndarray, Path],
        threshold: float = 0.5,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Get binary prediction mask at given threshold.
        
        Args:
            image: Input image
            threshold: Probability threshold for positive class
            target_class: Class index (default: crosswalk)
            
        Returns:
            Binary mask (H, W) with 1 for positive predictions
        """
        target_class = target_class or self.target_class
        
        if target_class == self.CROSSWALK_CLASS_IDX:
            probs = self.predict_probabilities(image)
        else:
            all_probs = self.predict_probabilities(image, return_all_classes=True)
            probs = all_probs[self.CLASSES[target_class]]
        
        return (probs >= threshold).astype(np.uint8)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from file."""
        path = Path(path)
        
        if path.suffix.lower() in ['.tif', '.tiff']:
            with rasterio.open(path) as src:
                img = src.read()
                if img.shape[0] in [3, 4]:
                    img = np.transpose(img, (1, 2, 0))
                if img.shape[-1] == 4:
                    img = img[:, :, :3]
        else:
            img = np.array(Image.open(path).convert('RGB'))
        
        return img
    
    def process_tile_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        save_probabilities: bool = True,
        save_masks: bool = True,
        threshold: float = 0.5
    ) -> Dict:
        """
        Process all tiles in a directory.
        
        Args:
            input_dir: Directory containing image tiles
            output_dir: Output directory
            save_probabilities: Save probability maps (for calibration)
            save_masks: Save binary masks
            threshold: Threshold for binary masks
            
        Returns:
            Processing summary dict
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        prob_dir = output_dir / "probabilities"
        mask_dir = output_dir / "masks"
        
        if save_probabilities:
            prob_dir.mkdir(parents=True, exist_ok=True)
        if save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Find tile images
        tile_paths = (
            list(input_dir.glob("*.tif")) +
            list(input_dir.glob("*.tiff")) +
            list(input_dir.glob("*.png"))
        )
        
        # Check for stitched subdirectory
        stitched_dir = input_dir / "stitched"
        if stitched_dir.exists():
            tile_paths += (
                list(stitched_dir.glob("*.tif")) +
                list(stitched_dir.glob("*.tiff"))
            )
        
        logger.info(f"Found {len(tile_paths)} tiles to process")
        
        if not self._model_loaded:
            self.load_model()
        
        results = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "threshold": threshold,
            "tiles_processed": 0,
            "probabilities": [],
            "masks": []
        }
        
        for tile_path in tqdm(tile_paths, desc="Processing tiles"):
            try:
                # Get probabilities
                probs = self.predict_probabilities(tile_path)
                
                # Get georeference info
                bounds, crs = self._get_georef_info(tile_path)
                
                # Save probability map
                if save_probabilities:
                    prob_path = prob_dir / f"{tile_path.stem}_prob.tif"
                    self._save_raster(probs, prob_path, bounds, crs)
                    results["probabilities"].append(str(prob_path))
                
                # Save binary mask
                if save_masks:
                    mask = (probs >= threshold).astype(np.uint8)
                    mask_path = mask_dir / f"{tile_path.stem}_mask.tif"
                    self._save_raster(mask, mask_path, bounds, crs)
                    results["masks"].append(str(mask_path))
                
                results["tiles_processed"] += 1
                
            except Exception as e:
                logger.error(f"Failed to process {tile_path}: {e}")
        
        # Save manifest
        manifest_path = output_dir / "inference_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Processed {results['tiles_processed']} tiles")
        
        return results
    
    def _get_georef_info(
        self,
        path: Path
    ) -> Tuple[Optional[rasterio.coords.BoundingBox], Optional[CRS]]:
        """Get georeferencing info from file."""
        if path.suffix.lower() in ['.tif', '.tiff']:
            try:
                with rasterio.open(path) as src:
                    return src.bounds, src.crs
            except Exception:
                pass
        return None, None
    
    def _save_raster(
        self,
        array: np.ndarray,
        path: Path,
        bounds: Optional[rasterio.coords.BoundingBox],
        crs: Optional[CRS]
    ):
        """Save array as GeoTIFF."""
        height, width = array.shape[:2]
        
        if bounds is not None:
            transform = from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top,
                width, height
            )
        else:
            transform = from_bounds(0, 0, width, height, width, height)
        
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=array.dtype if array.dtype in [np.uint8, np.float32] else np.float32,
            crs=crs or CRS.from_epsg(4326),
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(array.astype(dst.meta['dtype']), 1)


# =============================================================================
# Simulated Inference (for testing without GPU/model)
# =============================================================================

class SimulatedTile2NetInference(Tile2NetInference):
    """
    Simulated inference for testing pipeline without actual model.
    
    Generates synthetic probability maps that mimic model behavior
    for testing the evaluation pipeline.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path)
        self._model_loaded = True  # Skip model loading
        logger.info("Using SIMULATED inference (no actual model)")
    
    def load_model(self, weights_path: Optional[str] = None):
        """No-op for simulated inference."""
        self._model_loaded = True
    
    def predict_probabilities(
        self,
        image: Union[np.ndarray, Path],
        return_all_classes: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate simulated probability map.
        
        Creates synthetic probabilities based on image intensity patterns
        to mimic where crosswalks might appear.
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        height, width = image.shape[:2]
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Normalize
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        
        # Simulate crosswalk detection based on high-contrast linear features
        # This is a rough approximation - real model would be more sophisticated
        
        # Edge detection to find linear features
        from scipy import ndimage
        
        sobel_h = ndimage.sobel(gray, axis=0)
        sobel_v = ndimage.sobel(gray, axis=1)
        edges = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # High intensity + edges = potential crosswalk
        crosswalk_prob = 0.3 * gray + 0.7 * edges
        crosswalk_prob = np.clip(crosswalk_prob, 0, 1)
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.1, crosswalk_prob.shape)
        crosswalk_prob = np.clip(crosswalk_prob + noise, 0, 1).astype(np.float32)
        
        if return_all_classes:
            # Generate other class probabilities
            return {
                'background': 1 - crosswalk_prob * 0.5,
                'road': np.clip(0.3 + np.random.normal(0, 0.1, crosswalk_prob.shape), 0, 1).astype(np.float32),
                'sidewalk': np.clip(0.2 + np.random.normal(0, 0.1, crosswalk_prob.shape), 0, 1).astype(np.float32),
                'crosswalk': crosswalk_prob,
                'footpath': np.clip(0.1 + np.random.normal(0, 0.05, crosswalk_prob.shape), 0, 1).astype(np.float32)
            }
        
        return crosswalk_prob


def main():
    """CLI for Tile2Net inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Tile2Net inference")
    
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Config file')
    parser.add_argument('--input', '-i', required=True, help='Input tile directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--weights', '-w', help='Model weights path')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Threshold for masks')
    parser.add_argument('--no-probs', action='store_true', help='Skip probability maps')
    parser.add_argument('--no-masks', action='store_true', help='Skip binary masks')
    parser.add_argument('--simulate', action='store_true', help='Use simulated inference (testing)')
    
    args = parser.parse_args()
    
    # Create inference object
    if args.simulate:
        inference = SimulatedTile2NetInference(args.config)
    else:
        inference = Tile2NetInference(args.config)
        if args.weights:
            inference.load_model(args.weights)
    
    # Process tiles
    results = inference.process_tile_directory(
        Path(args.input),
        Path(args.output),
        save_probabilities=not args.no_probs,
        save_masks=not args.no_masks,
        threshold=args.threshold
    )
    
    print(f"\nInference complete:")
    print(f"  Tiles processed: {results['tiles_processed']}")
    print(f"  Probability maps: {len(results['probabilities'])}")
    print(f"  Binary masks: {len(results['masks'])}")


if __name__ == "__main__":
    main()