import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from typing import List, Dict, Tuple


class YOLODetector:
    """YOLO-based oriented bounding box detector for parcels"""
    
    def __init__(self, model_path: str, roi: Tuple[int, int, int, int], 
                 confidence: float = 0.8):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO weights
            roi: Region of interest (x_min, y_min, x_max, y_max)
            confidence: Detection confidence threshold
        """
        self.model = YOLO(model_path, task='detect')
        self.confidence = confidence
        
        x1, y1, x2, y2 = roi
        self.roi = Polygon([
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2)
        ])
    
    def detect(self, rgb_img_path: str, scale: float = 1.0) -> List[Dict]:
        """
        Detect parcels in RGB image
        
        Args:
            rgb_img_path: Path to RGB image
            scale: Scale factor for box expansion
            
        Returns:
            List of detected packages with box_points and center_point
        """
        rgb_img = cv2.imread(rgb_img_path, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_img, conf=self.confidence, verbose=False)
        
        packages = []
        
        for res in results:
            if not hasattr(res, 'obb') or res.obb is None:
                continue
                
            for xyxyxyxy in res.obb.xyxyxyxy.cpu().numpy():
                box_points = xyxyxyxy
                
                cx = np.mean(box_points[:, 0])
                cy = np.mean(box_points[:, 1])
                center_point = Point(cx, cy)
                
                # Filter by ROI
                if not self.roi.contains(center_point):
                    continue
                
                # Scale box if needed
                if scale != 1.0:
                    center = np.array([cx, cy])
                    scaled_points = center + (box_points - center) * scale
                else:
                    scaled_points = box_points.copy()
                
                packages.append({
                    "box_points": box_points,
                    "box_points_scaled": scaled_points,
                    "center_point": np.array([cx, cy])
                })
        
        return packages