import numpy as np
from typing import List, Dict, Tuple


class GeometryUtils:
    """Utility functions for geometric operations"""
    
    @staticmethod
    def select_topmost_package(packages_3D: List[Dict], 
                              strategy: str,
                              roi_top_right: Tuple[float, float]) -> Dict:
        """
        Select the topmost package based on strategy
        
        Args:
            packages_3D: List of packages with 3D properties
            strategy: "center", "mean_orig", or "mean_scaled"
            roi_top_right: (x, y) of ROI top-right corner
            
        Returns:
            Selected package
        """
        if len(packages_3D) == 0:
            raise ValueError("No packages to select from")
        
        if len(packages_3D) == 1:
            return packages_3D[0]
        
        # Choose primary key
        if strategy == "center":
            primary_key = "center_depth"
            secondary_key = "mean_depth_scaled"
        elif strategy == "mean_orig":
            primary_key = "mean_depth_orig"
            secondary_key = None
        else:
            primary_key = "mean_depth_scaled"
            secondary_key = None
        
        # Sort by primary key
        packages_3D.sort(key=lambda p: p[primary_key])
        best_primary = packages_3D[0][primary_key]
        
        # Filter candidates within 5mm
        candidates = [p for p in packages_3D 
                     if abs(p[primary_key] - best_primary) < 0.005]
        
        # Sort by secondary key if needed
        if secondary_key and len(candidates) > 1:
            candidates.sort(key=lambda p: p[secondary_key])
            best_secondary = candidates[0][secondary_key]
            candidates = [p for p in candidates 
                         if abs(p[secondary_key] - best_secondary) < 0.005]
        
        # Sort by distance to top-right (farther is better)
        if len(candidates) > 1:
            top_right_x, top_right_y = roi_top_right
            candidates.sort(
                key=lambda p: -np.sqrt(
                    (np.mean(p['box_points'][:, 0]) - top_right_x)**2 +
                    (np.mean(p['box_points'][:, 1]) - top_right_y)**2
                )
            )
        
        return candidates[0]