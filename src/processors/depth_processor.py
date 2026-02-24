import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class DepthProcessor:
    """Process depth images to extract 3D coordinates"""
    
    def __init__(self, color_intrinsic: Dict):
        """
        Initialize depth processor
        
        Args:
            color_intrinsic: Camera intrinsic parameters
        """
        self.fx = color_intrinsic["fx"]
        self.fy = color_intrinsic["fy"]
        self.cx = color_intrinsic["cx"]
        self.cy = color_intrinsic["cy"]
    
    def get_center_depth(self, package: Dict, depth_img: np.ndarray) -> np.ndarray:
        """
        Get robust 3D coordinate of package center
        
        Args:
            package: Package dict with center_point
            depth_img: Depth image in meters
            
        Returns:
            3D coordinate [X, Y, Z]
        """
        cx, cy = map(int, package["center_point"])
        h, w = depth_img.shape[:2]
        
        # Try center pixel first
        z_center = depth_img[cy, cx].item()
        
        if np.isfinite(z_center) and z_center > 0:
            depth_verify, conf = self._get_robust_depth(depth_img, cx, cy, 1)
            if depth_verify is not None and abs(z_center - depth_verify) / depth_verify < 0.15:
                return self._pixel_to_3d(cx, cy, z_center)
        
        # Search with increasing radius
        best_result = None
        best_confidence = 0
        
        for radius in [2, 3, 5, 7, 10, 15, 20, 30, 50]:
            depth_estimate, confidence = self._get_robust_depth(depth_img, cx, cy, radius)
            
            if depth_estimate is not None:
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = (depth_estimate, radius)
                
                if confidence > 0.3:
                    radius_1cm = int((0.01 * self.fx) / depth_estimate)
                    radius_1cm = max(2, min(radius_1cm, 15))
                    
                    final_depth, _ = self._get_robust_depth(depth_img, cx, cy, radius_1cm)
                    
                    if final_depth is not None:
                        return self._pixel_to_3d(cx, cy, final_depth)
        
        # Use best result
        if best_result is not None:
            depth_estimate, _ = best_result
            return self._pixel_to_3d(cx, cy, depth_estimate)
        
        # Fallback: box mean
        return self._get_box_fallback(package, depth_img)
    
    def _get_robust_depth(self, depth_img: np.ndarray, cx: int, cy: int, 
                         radius: int) -> Tuple[Optional[float], float]:
        """
        Get robust depth estimate in a region
        
        Returns:
            (depth_estimate, confidence)
        """
        h, w = depth_img.shape[:2]
        
        y_min = max(0, cy - radius)
        y_max = min(h - 1, cy + radius)
        x_min = max(0, cx - radius)
        x_max = min(w - 1, cx + radius)
        
        region = depth_img[y_min:y_max+1, x_min:x_max+1]
        valid_depths = region[(np.isfinite(region)) & (region > 0)]
        
        if len(valid_depths) == 0:
            return None, 0
        
        if len(valid_depths) < 3:
            return np.median(valid_depths), len(valid_depths) / ((radius * 2 + 1)**2)
        
        median_depth = np.median(valid_depths)
        
        # Remove outliers
        lower_bound = median_depth * 0.8
        upper_bound = median_depth * 1.2
        filtered = valid_depths[(valid_depths >= lower_bound) & 
                               (valid_depths <= upper_bound)]
        
        if len(filtered) < 3:
            return median_depth, len(valid_depths) / ((radius * 2 + 1)**2)
        
        mean_depth = np.mean(filtered)
        confidence = len(filtered) / (radius * 2 + 1)**2
        
        return mean_depth, confidence
    
    def _pixel_to_3d(self, u: int, v: int, z: float) -> np.ndarray:
        """Convert pixel + depth to 3D coordinate"""
        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy
        return np.array([float(X), float(Y), float(z)])
    
    def _get_box_fallback(self, package: Dict, depth_img: np.ndarray) -> np.ndarray:
        """Fallback: use mean depth of entire box"""
        h, w = depth_img.shape[:2]
        cx, cy = map(int, package["center_point"])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        box_int = np.intp(package["box_points"])
        cv2.fillPoly(mask, [box_int], 1)
        
        masked_depth = depth_img[mask == 1]
        valid_depths = masked_depth[(np.isfinite(masked_depth)) & (masked_depth > 0)]
        
        if len(valid_depths) > 0:
            fallback_depth = float(np.median(valid_depths))
            return self._pixel_to_3d(cx, cy, fallback_depth)
        
        raise ValueError(f"Cannot find valid depth for package at ({cx}, {cy})")
    
    def get_mean_depth(self, package: Dict, depth_img: np.ndarray) -> Tuple[float, float]:
        """
        Get mean depth of box_points and box_points_scaled
        
        Returns:
            (mean_depth_orig, mean_depth_scaled)
        """
        h, w = depth_img.shape[:2]
        
        def compute_mean(box_points):
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.intp(box_points)], 1)
            
            masked_depth = depth_img[mask == 1]
            valid_depths = masked_depth[(np.isfinite(masked_depth)) & (masked_depth > 0)]
            
            if len(valid_depths) == 0:
                return None
            
            q1, q3 = np.percentile(valid_depths, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            filtered = valid_depths[(valid_depths >= lower) & (valid_depths <= upper)]
            
            if len(filtered) == 0:
                return float(np.median(valid_depths))
            
            return float(np.mean(filtered))
        
        mean_orig = compute_mean(package["box_points"])
        mean_scaled = compute_mean(package["box_points_scaled"])
        
        return mean_orig, mean_scaled