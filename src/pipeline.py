import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List

from models import YOLODetector
from processors import DepthProcessor, PointCloudProcessor, Postprocessor
from utils import GeometryUtils, IOUtils, Visualizer


class Pipeline:
    """Main pipeline for 3D parcel detection"""
    
    def __init__(self, config: Dict):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.detector = YOLODetector(
            model_path=config["paths"]["yolo_model"],
            roi=tuple(config["ROI"]),
            confidence=config["detection"]["confidence_threshold"]
        )
        
        self.postprocessor = Postprocessor(
            iou_threshold=config["detection"]["iou_threshold"]
        )
        
        self.depth_processor = DepthProcessor(
            color_intrinsic=config["color_intrinsic"]
        )
        
        self.pointcloud_processor = PointCloudProcessor(
            color_intrinsic=config["color_intrinsic"],
            knn=config["pointcloud"]["knn"]
        )
        
        # Setup paths
        self.root_dir = config["paths"]["root_dir"]
        self.output_file = config["paths"]["output_file"]
        
        # Create debug folder
        self.debug_folder = IOUtils.create_debug_folder(
            config["paths"]["debug_folder"]
        )
        
        # Save config
        IOUtils.save_config(config, 
                           os.path.join(self.debug_folder, "config.yaml"))
        
        print(f"Pipeline initialized")
        print(f"  Debug folder: {self.debug_folder}")
    
    def process_image(self, rgb_path: str, depth_path: str) -> Dict:
        """
        Process a single image pair
        
        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth image
            
        Returns:
            Result dictionary with 3D coordinates and normal vector
        """
        img_name = os.path.basename(rgb_path)
        print(f"\nProcessing {img_name}...")
        
        # 1. Detect packages
        packages = self.detector.detect(
            rgb_path, 
            scale=self.config["detection"]["scale"]
        )
        print(f"  Detected {len(packages)} packages")
        
        # 2. Post-process
        packages = self.postprocessor.process(packages)
        print(f"  After filtering: {len(packages)} packages")
        
        if len(packages) == 0:
            raise ValueError("No packages detected after filtering")
        
        # 3. Load depth image
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_img /= 1000.0  # Convert mm to m
        
        # 4. Get 3D properties for all packages
        packages_3D = []
        for pkg in packages:
            # Get 3D coordinate
            center_3D = self.depth_processor.get_center_depth(pkg, depth_img)
            
            # Get mean depths
            mean_orig, mean_scaled = self.depth_processor.get_mean_depth(pkg, depth_img)
            
            # Get point clouds
            pc_orig, pc_scaled = self.pointcloud_processor.extract_pointcloud(
                pkg, depth_img
            )
            
            packages_3D.append({
                "3D_coordinate": center_3D,
                "center_depth": center_3D[-1],
                "mean_depth_orig": mean_orig,
                "mean_depth_scaled": mean_scaled,
                "center_point": pkg["center_point"],
                "box_points": pkg["box_points"],
                "box_points_scaled": pkg["box_points_scaled"],
                "pc_orig": pc_orig,
                "pc_scaled": pc_scaled
            })
        
        # 5. Select topmost package
        roi_top_right = (self.config["ROI"][2], self.config["ROI"][1])
        selected = GeometryUtils.select_topmost_package(
            packages_3D,
            strategy=self.config["depth"]["strategy"],
            roi_top_right=roi_top_right
        )
        
        print(f"  Selected package at depth: {selected['center_depth']:.3f}m")
        
        # 6. Estimate normal vector
        normal = self.pointcloud_processor.estimate_normal_radius(
            selected["pc_scaled"],
            radius_percentile=self.config["pointcloud"]["normal_radius_percentile"],
            max_radius=self.config["pointcloud"]["max_radius"]
        )
        
        # 7. Save visualizations
        self._save_visualizations(img_name, rgb_path, packages_3D, selected)
        
        # 8. Return result
        x, y, z = selected["3D_coordinate"]
        Rx, Ry, Rz = normal
        
        return {
            'image_filename': 'image_' + img_name,
            'x': float(x),
            'y': float(y),
            'z': float(z),
            'Rx': float(Rx),
            'Ry': float(Ry),
            'Rz': float(Rz)
        }
    
    def _save_visualizations(self, img_name: str, rgb_path: str,
                           packages_3D: List[Dict], selected: Dict):
        """Save debug visualizations"""
        rgb_img = cv2.imread(rgb_path)
        
        # Save detection result
        img_detected = Visualizer.draw_packages(
            rgb_img,
            [{"box_points": p["box_points"],
              "box_points_scaled": p["box_points_scaled"],
              "center_point": p["center_point"]} for p in packages_3D]
        )
        cv2.imwrite(
            os.path.join(self.debug_folder, img_name.replace(".png", "_detected.png")),
            img_detected
        )
        
        # Save depth info
        img_depth = Visualizer.draw_depth_info(rgb_img, packages_3D)
        cv2.imwrite(
            os.path.join(self.debug_folder, img_name.replace(".png", "_depth.png")),
            img_depth
        )
        
        # Save selected package
        img_selected = Visualizer.draw_packages(
            rgb_img,
            [{"box_points": selected["box_points"],
              "box_points_scaled": selected["box_points_scaled"],
              "center_point": selected["center_point"]}],
            selected_idx=0
        )
        cv2.imwrite(
            os.path.join(self.debug_folder, img_name.replace(".png", "_selected.png")),
            img_selected
        )
    
    def run(self):
        """Run pipeline on all images"""
        rgb_folder = os.path.join(self.root_dir, 'rgb')
        depth_folder = os.path.join(self.root_dir, 'depth')
        
        rgb_files = sorted([f for f in os.listdir(rgb_folder) 
                           if f.endswith('.png')])
        
        print(f"\n{'='*80}")
        print(f"Processing {len(rgb_files)} images...")
        print(f"{'='*80}")
        
        results = []
        
        for rgb_file in rgb_files:
            try:
                rgb_path = os.path.join(rgb_folder, rgb_file)
                depth_path = os.path.join(depth_folder, rgb_file)
                
                result = self.process_image(rgb_path, depth_path)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {rgb_file}: {e}")
                # Add fallback result
                results.append({
                    'image_filename': 'image_' + rgb_file,
                    'x': 0.0, 'y': 0.0, 'z': 1.0,
                    'Rx': 0.0, 'Ry': 0.0, 'Rz': 1.0
                })
        
        # Save submission
        IOUtils.save_submission(results, self.output_file)
        
        print(f"\n{'='*80}")
        print(f"Pipeline completed!")
        print(f"  Results saved to: {self.output_file}")
        print(f"  Debug folder: {self.debug_folder}")
        print(f"{'='*80}\n")