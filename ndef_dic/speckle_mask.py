"""
Speckle mask generator for NDeF-DIC.

Automatically segments the speckle-textured specimen region from the
background using texture features (local variance, gradient magnitude,
local entropy), Otsu thresholding, and morphological post-processing.

The resulting masks are used to:
  1. Mask background before COLMAP (so SfM only sees the specimen).
  2. Mask background pixels during photometric loss computation.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpeckleMaskConfig:
    """Configuration for speckle mask generation."""
    window_size: int = 15
    close_kernel_size: int = 15
    open_kernel_size: int = 7
    vote_threshold: int = 2  # need ≥2 of 3 features to agree
    keep_largest_only: bool = True


class SpeckleMaskGenerator:
    """
    Generate binary masks that isolate the speckle-textured specimen
    from the background in each camera view.

    Algorithm:
      1. Compute 3 texture feature maps per image:
         - Local variance (σ²)
         - Local gradient magnitude (|∇I|)
         - Local entropy (H)
      2. Otsu threshold each feature map independently.
      3. Majority vote: pixel is "speckle" if ≥2 of 3 features agree.
      4. Morphological post-processing (close holes, remove noise).
      5. Optionally keep only the largest connected component.

    Usage:
        generator = SpeckleMaskGenerator(config)
        masks, masked_images = generator.generate(images)
    """

    def __init__(self, config: SpeckleMaskConfig | None = None):
        self.config = config or SpeckleMaskConfig()

    def generate(
        self,
        images: List[np.ndarray],
        return_features: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]] | Tuple[
        List[np.ndarray], List[np.ndarray], List[dict]
    ]:
        """
        Generate speckle masks for a list of images.

        Args:
            images: List of (H, W) grayscale images, np.float32 or np.uint8.
            return_features: If True, also return feature maps for inspection.

        Returns:
            masks: List of (H, W) binary masks (np.uint8, 255 = speckle).
            masked_images: List of (H, W) images with background set to 0.
            features_list (optional): List of dicts with feature maps.
        """
        masks = []
        masked_images = []
        features_list = [] if return_features else None

        for img in images:
            # Ensure float [0, 1]
            if img.dtype == np.uint8:
                img_f = img.astype(np.float32) / 255.0
            else:
                img_f = img.astype(np.float32)

            # Step 1: Compute texture features
            features = self._compute_features(img_f)
            var_map, grad_map, entropy_map = features["variance"], \
                                              features["gradient"], \
                                              features["entropy"]

            # Step 2: Otsu thresholding + majority vote
            mask = self._threshold_and_vote(var_map, grad_map, entropy_map)

            # Step 3: Morphological post-processing
            mask = self._postprocess(mask)

            masks.append(mask)

            # Apply mask
            masked = img_f.copy()
            masked[mask == 0] = 0.0
            masked_images.append(masked)

            if return_features:
                features_list.append(features)

        if return_features:
            return masks, masked_images, features_list
        return masks, masked_images

    def _compute_features(self, img: np.ndarray) -> dict:
        """
        Compute texture feature maps for a single image.

        Args:
            img: (H, W) float32, range [0, 1].

        Returns:
            dict with keys 'variance', 'gradient', 'entropy'.
        """
        ws = self.config.window_size

        # Must be uint8 for OpenCV operations
        img_u8 = (img * 255).astype(np.uint8)

        # --- Local variance ---
        mean = cv2.blur(img, (ws, ws))
        mean_sq = cv2.blur(img ** 2, (ws, ws))
        var_map = mean_sq - mean ** 2  # (H, W), float
        var_map = np.maximum(var_map, 0)  # numerical stability

        # --- Local gradient magnitude ---
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        grad_map = cv2.blur(grad, (ws, ws))

        # --- Local entropy ---
        entropy_map = self._local_entropy(img_u8, ws)

        return {
            "variance": var_map,
            "gradient": grad_map,
            "entropy": entropy_map,
        }

    def _local_entropy(self, img_u8: np.ndarray, window_size: int) -> np.ndarray:
        """
        Compute local Shannon entropy in sliding windows.

        H = -Σ p_i * log2(p_i) over the intensity histogram in the window.

        Efficient implementation using integral histograms or sliding
        window + histogram per pixel.

        Args:
            img_u8: (H, W) uint8 image [0, 255].
            window_size: window size.

        Returns:
            entropy_map: (H, W) float32, entropy values.
        """
        H, W = img_u8.shape
        half = window_size // 2

        # Pad image
        padded = np.pad(img_u8, half, mode="reflect")
        entropy_map = np.zeros((H, W), dtype=np.float32)

        # Process each pixel (can be slow for large images — optimize later)
        # For now, use a vectorized approach with stride tricks
        for i in range(H):
            for j in range(W):
                patch = padded[i:i + window_size, j:j + window_size]
                hist, _ = np.histogram(patch, bins=32, range=(0, 255))
                hist = hist / hist.sum()
                # Avoid log(0)
                hist = hist[hist > 0]
                entropy_map[i, j] = -np.sum(hist * np.log2(hist))

        return entropy_map

    def _threshold_and_vote(
        self,
        var_map: np.ndarray,
        grad_map: np.ndarray,
        entropy_map: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Otsu threshold to each feature map and take majority vote.

        The combination of 3 features makes the segmentation robust:
        - A true speckle region is high in all 3 features.
        - Background is low in all 3 (or at most 1 by chance).
        """
        # Normalize each map to [0, 255] for Otsu
        var_norm = self._normalize_uint8(var_map)
        grad_norm = self._normalize_uint8(grad_map)
        entropy_norm = self._normalize_uint8(entropy_map)

        # Otsu thresholding
        _, var_mask = cv2.threshold(
            var_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, grad_mask = cv2.threshold(
            grad_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, entropy_mask = cv2.threshold(
            entropy_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Majority vote
        vote = (
            (var_mask > 0).astype(np.int32)
            + (grad_mask > 0).astype(np.int32)
            + (entropy_mask > 0).astype(np.int32)
        )
        mask = (vote >= self.config.vote_threshold).astype(np.uint8) * 255

        return mask

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        """
        Morphological post-processing:
          1. Close: fill small holes inside the speckle region.
          2. Open: remove isolated noise outside.
          3. Optionally keep only the largest connected component.
          4. Edge smoothing.
        """
        # Closing — fill holes
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.close_kernel_size, self.config.close_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # Opening — remove noise
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.open_kernel_size, self.config.open_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # Keep largest connected component
        if self.config.keep_largest_only:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            if num_labels > 1:
                # Label 0 is background
                largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask = (labels == largest_label).astype(np.uint8) * 255

        # Edge smoothing
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_smooth)

        return mask

    @staticmethod
    def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 255] uint8."""
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max - arr_min < 1e-8:
            return np.zeros_like(arr, dtype=np.uint8)
        normalized = (arr - arr_min) / (arr_max - arr_min) * 255.0
        return normalized.astype(np.uint8)

    @staticmethod
    def save_mask(mask: np.ndarray, filepath: str):
        """Save a mask to disk."""
        cv2.imwrite(filepath, mask)

    @staticmethod
    def load_mask(filepath: str) -> np.ndarray:
        """Load a mask from disk."""
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
