import cv2
import numpy as np
import open3d as o3d
from typing import List, Dict


class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def draw_packages(img: np.ndarray, packages: List[Dict], 
                     selected_idx: int = -1) -> np.ndarray:
        """
        Draw detected packages on image
        
        Args:
            img: Input image
            packages: List of packages to draw
            selected_idx: Index of selected package (-1 for all)
            
        Returns:
            Image with drawn packages
        """
        img_copy = img.copy()
        
        for i, pkg in enumerate(packages):
            # Color selection
            if selected_idx >= 0 and i == selected_idx:
                color_orig = (0, 255, 0)     # Green for selected
                color_scaled = (255, 255, 0) # Yellow for selected
                thickness = 3
            else:
                color_orig = (128, 128, 128)   # Gray for others
                color_scaled = (200, 200, 200)
                thickness = 1
            
            # Draw boxes
            box_orig = np.intp(pkg["box_points"])
            box_scaled = np.intp(pkg["box_points_scaled"])
            
            cv2.drawContours(img_copy, [box_orig], 0, color_orig, thickness)
            cv2.drawContours(img_copy, [box_scaled], 0, color_scaled, thickness)
            
            # Draw center
            cx, cy = map(int, pkg["center_point"])
            cv2.circle(img_copy, (cx, cy), 5, (0, 0, 255), -1)
        
        return img_copy
    
    @staticmethod
    def draw_depth_info(img: np.ndarray, packages_3D: List[Dict]) -> np.ndarray:
        """
        Draw depth information on image
        
        Args:
            img: Input image
            packages_3D: List of packages with 3D properties
            
        Returns:
            Image with depth info
        """
        img_copy = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 255, 0)
        
        for pkg in packages_3D:
            cx, cy = map(int, pkg["center_point"])
            
            # Draw center point
            cv2.circle(img_copy, (cx, cy), 3, (0, 0, 255), -1)
            
            # Draw depth info
            center_z = pkg["3D_coordinate"][-1]
            mean_orig = pkg.get("mean_depth_orig", 0)
            mean_scaled = pkg.get("mean_depth_scaled", 0)
            
            cv2.putText(img_copy, f"c: {center_z:.3f}", (cx+5, cy-10),
                       font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(img_copy, f"s: {mean_scaled:.3f}", (cx+5, cy+5),
                       font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.putText(img_copy, f"o: {mean_orig:.3f}", (cx+5, cy+20),
                       font, font_scale, color, thickness, cv2.LINE_AA)
        
        return img_copy
    
    @staticmethod
    def save_colored_pointcloud(points: np.ndarray, center: np.ndarray,
                               output_path: str, neighbor_radius: float = 0.02):
        """
        Save colored point cloud to PLY file
        
        Args:
            points: Point cloud (N, 3)
            center: Center point (3,)
            output_path: Output PLY file path
            neighbor_radius: Radius for coloring near center
        """
        if points.shape[0] < 3:
            print(f"Warning: Not enough points to save {output_path}")
            return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Default gray color
        colors = np.ones_like(points) * 0.7
        
        # Red color for points near center
        dists = np.linalg.norm(points - center, axis=1)
        mask = dists <= neighbor_radius
        colors[mask] = np.array([1, 0, 0])  # Red
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved colored point cloud to {output_path}")