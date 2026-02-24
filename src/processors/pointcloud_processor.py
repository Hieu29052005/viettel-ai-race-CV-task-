import cv2
import numpy as np
import open3d as o3d
from typing import Dict, Tuple, Optional


class PointCloudProcessor:
    """Process point clouds for normal vector estimation"""
    
    def __init__(self, color_intrinsic: Dict, knn: int = 30):
        """
        Initialize point cloud processor
        
        Args:
            color_intrinsic: Camera intrinsic parameters
            knn: Number of neighbors for normal estimation
        """
        self.fx = color_intrinsic["fx"]
        self.fy = color_intrinsic["fy"]
        self.cx = color_intrinsic["cx"]
        self.cy = color_intrinsic["cy"]
        self.knn = knn
    
    def extract_pointcloud(self, package: Dict, depth_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract point cloud from box regions
        
        Args:
            package: Package with box_points and box_points_scaled
            depth_img: Depth image in meters
            
        Returns:
            (pc_orig, pc_scaled): Point clouds as (N, 3) arrays
        """
        h, w = depth_img.shape[:2]
        
        def extract_points(box_points):
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.intp(box_points)], 1)
            
            ys, xs = np.where(mask == 1)
            if len(xs) == 0:
                return np.empty((0, 3), dtype=np.float32)
            
            depths = depth_img[ys, xs]
            
            if depths.ndim > 1:
                depths = np.mean(depths, axis=-1)
            depths = np.ravel(depths)
            
            # Filter invalid
            valid_mask = np.isfinite(depths) & (depths > 0)
            if np.sum(valid_mask) < 3:
                return np.empty((0, 3), dtype=np.float32)
            
            xs = xs[valid_mask]
            ys = ys[valid_mask]
            depths = depths[valid_mask]
            
            # Remove outliers
            if len(depths) > 10:
                q1, q3 = np.percentile(depths, [25, 75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                inlier_mask = (depths >= lower) & (depths <= upper)
                xs, ys, depths = xs[inlier_mask], ys[inlier_mask], depths[inlier_mask]
            
            if len(depths) < 3:
                return np.empty((0, 3), dtype=np.float32)
            
            # Convert to 3D
            X = (xs - self.cx) * depths / self.fx
            Y = (ys - self.cy) * depths / self.fy
            Z = depths
            
            return np.stack([X, Y, Z], axis=-1).astype(np.float32)
        
        pc_orig = extract_points(package["box_points"])
        pc_scaled = extract_points(package["box_points_scaled"])
        
        return pc_orig, pc_scaled
    
    def estimate_normal_knn(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate normal using KNN approach
        
        Args:
            points: Point cloud (N, 3)
            
        Returns:
            Normal vector (3,) or None
        """
        if points.shape[0] < 3:
            return None
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.knn)
        )
        
        normals = np.asarray(pcd.normals)
        mean_normal = np.mean(normals, axis=0)
        
        norm = np.linalg.norm(mean_normal)
        if norm == 0:
            return None
        
        mean_normal /= norm
        return mean_normal
    
    def estimate_normal_radius(self, points: np.ndarray, 
                              radius_percentile: int = 65,
                              max_radius: float = 0.1) -> np.ndarray:
        """
        Estimate normal using radius-based PCA
        
        Args:
            points: Point cloud (N, 3)
            radius_percentile: Percentile for radius selection
            max_radius: Maximum radius in meters
            
        Returns:
            Normal vector (3,)
        """
        center_3D = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center_3D, axis=1)
        radius = min(np.percentile(distances, radius_percentile), max_radius)
        
        nearby_points = points[distances <= radius]
        
        centered = nearby_points - center_3D
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        idx = np.argmin(eigenvalues)
        normal = eigenvectors[:, idx].real
        normal = normal / np.linalg.norm(normal)
        
        return normal