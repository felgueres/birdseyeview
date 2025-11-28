import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Dict, Tuple


class DepthEstimator:
    """
    Monocular depth estimation using Depth Anything V2.
    Estimates relative depth from a single RGB image.
    """
    
    def __init__(self, model_size='small', device='cuda'):
        """
        Initialize depth estimation model.
        
        Args:
            model_size: 'small', 'base', or 'large'
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Model variants
        models = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf',
        }
        
        model_name = models.get(model_size, models['small'])
        
        print(f"Loading Depth Anything V2 ({model_size})...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Depth Anything V2 loaded on {self.device}")
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB frame.
        
        Args:
            frame: Input image (H, W, 3) in BGR format
            
        Returns:
            Depth map (H, W) - normalized to [0, 1], closer = higher values
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        inputs = self.processor(images=rgb_frame, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy and normalize
        depth = prediction.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth
    
    def get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int, 
                           radius: int = 5) -> float:
        """
        Get average depth value at a point (useful for objects).
        
        Args:
            depth_map: Depth map from estimate_depth()
            x, y: Point coordinates
            radius: Average over a small region
            
        Returns:
            Average depth value [0, 1]
        """
        h, w = depth_map.shape
        y1 = max(0, y - radius)
        y2 = min(h, y + radius)
        x1 = max(0, x - radius)
        x2 = min(w, x + radius)
        
        region = depth_map[y1:y2, x1:x2]
        return float(np.mean(region))
    
    def get_depth_for_bbox(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Get depth statistics for a bounding box region.
        
        Args:
            depth_map: Depth map
            bbox: (x1, y1, x2, y2)
            
        Returns:
            Dict with 'mean', 'min', 'max' depth values
        """
        x1, y1, x2, y2 = bbox
        region = depth_map[y1:y2, x1:x2]
        
        return {
            'mean': float(np.mean(region)),
            'min': float(np.min(region)),
            'max': float(np.max(region)),
            'median': float(np.median(region))
        }
    
    def colorize_depth(self, depth_map: np.ndarray, colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        Convert depth map to colored visualization.
        
        Args:
            depth_map: Normalized depth map [0, 1]
            colormap: OpenCV colormap
            
        Returns:
            Colored depth map (H, W, 3) in BGR
        """
        # Convert to uint8
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(depth_uint8, colormap)
        
        return colored
    
    def create_depth_overlay(self, frame: np.ndarray, depth_map: np.ndarray, 
                            alpha: float = 0.5) -> np.ndarray:
        """
        Blend depth map with original frame.
        
        Args:
            frame: Original frame
            depth_map: Depth map
            alpha: Blending factor (0 = only frame, 1 = only depth)
            
        Returns:
            Blended image
        """
        colored_depth = self.colorize_depth(depth_map)
        blended = cv2.addWeighted(frame, 1 - alpha, colored_depth, alpha, 0)
        return blended
    
    def estimate_relative_distance(self, depth1: float, depth2: float) -> str:
        """
        Compare relative distances between two objects.
        
        Args:
            depth1, depth2: Depth values [0, 1] from get_depth_at_point()
            
        Returns:
            String like "closer", "farther", "same distance"
        """
        diff = abs(depth1 - depth2)
        
        if diff < 0.05:
            return "same distance"
        elif depth1 > depth2:
            return "closer"
        else:
            return "farther"
    
    def get_depth_statistics(self, depth_map: np.ndarray) -> Dict:
        """Get overall statistics for the depth map"""
        return {
            'mean_depth': float(np.mean(depth_map)),
            'median_depth': float(np.median(depth_map)),
            'min_depth': float(np.min(depth_map)),
            'max_depth': float(np.max(depth_map)),
            'std_depth': float(np.std(depth_map)),
            'near_percentage': float(np.sum(depth_map > 0.7) / depth_map.size * 100),
            'far_percentage': float(np.sum(depth_map < 0.3) / depth_map.size * 100),
        }

