import numpy as np
from shapely.geometry import Polygon
from typing import List, Dict


class Postprocessor:
    """Post-process detected packages to remove overlaps"""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize postprocessor
        
        Args:
            iou_threshold: IoU threshold for overlap removal
        """
        self.iou_threshold = iou_threshold
    
    def process(self, packages: List[Dict]) -> List[Dict]:
        """
        Remove overlapping packages based on IoU and containment
        
        Args:
            packages: List of detected packages
            
        Returns:
            Filtered list of packages
        """
        if len(packages) <= 1:
            return packages
        
        polygons = [Polygon(pkg["box_points"]) for pkg in packages]
        keep = [True] * len(packages)
        
        for i in range(len(packages)):
            if not keep[i]:
                continue
                
            for j in range(i+1, len(packages)):
                if not keep[j]:
                    continue
                
                poly_i = polygons[i]
                poly_j = polygons[j]
                
                if not poly_i.is_valid or not poly_j.is_valid:
                    continue
                
                inter_area = poly_i.intersection(poly_j).area
                if inter_area == 0:
                    continue
                
                smaller_area = min(poly_i.area, poly_j.area)
                overlap_ratio = inter_area / smaller_area
                
                # Check containment
                if poly_i.contains(poly_j):
                    keep[i] = False
                    continue
                elif poly_j.contains(poly_i):
                    keep[j] = False
                    continue
                
                # Check overlap
                if overlap_ratio > self.iou_threshold:
                    if poly_i.area >= poly_j.area:
                        keep[i] = False
                    else:
                        keep[j] = False
        
        filtered = [pkg for k, pkg in zip(keep, packages) if k]
        return filtered