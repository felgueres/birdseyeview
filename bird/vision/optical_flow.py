import cv2
import numpy as np
from typing import Tuple, Optional, Dict

class OpticalFlowTracker:
    def __init__(self, method='lucas_kanade'):
        """
        Args:
            method: 'lucas_kanade' for sparse flow or 'farneback' for dense flow
        """
        self.method = method
        self.prev_frame = None
        self.prev_points = None
        
        # Parameters for Lucas-Kanade sparse optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Parameters for feature detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
    
    def compute_sparse_flow(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute sparse optical flow using Lucas-Kanade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            # First frame - detect features
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_frame = gray
            return None, None
        
        if self.prev_points is None or len(self.prev_points) == 0:
            # Re-detect features if lost
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_frame = gray
            return None, None
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.prev_points, None, **self.lk_params
        )
        
        if new_points is None:
            return None, None
        
        # Select good points
        good_new = new_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        # Update for next iteration
        self.prev_frame = gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        
        return good_old, good_new
    
    def compute_dense_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Compute dense optical flow using Farneback"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return None
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None,
            pyr_scale=0.5,      # Image pyramid scale
            levels=3,           # Number of pyramid layers
            winsize=15,         # Averaging window size
            iterations=3,       # Iterations at each pyramid level
            poly_n=5,           # Pixel neighborhood size
            poly_sigma=1.2,     # Gaussian smoothing
            flags=0
        )
        
        self.prev_frame = gray
        return flow
    
    def draw_sparse_flow(self, frame: np.ndarray, old_points: np.ndarray, new_points: np.ndarray) -> np.ndarray:
        """Visualize sparse optical flow as arrows"""
        vis_frame = frame.copy()
        
        for old, new in zip(old_points, new_points):
            a, b = old.ravel()
            c, d = new.ravel()
            
            # Draw arrow from old to new position
            cv2.arrowedLine(vis_frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2, tipLength=0.3)
            # Draw circles at feature points
            cv2.circle(vis_frame, (int(c), int(d)), 3, (0, 0, 255), -1)
        
        return vis_frame
    
    def draw_dense_flow(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Visualize dense optical flow as color-coded heatmap"""
        # Convert flow to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create HSV image for visualization
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255  # Full saturation
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue represents direction
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude
        
        # Convert HSV to BGR
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Blend with original frame
        vis_frame = cv2.addWeighted(frame, 0.7, flow_vis, 0.3, 0)
        
        return vis_frame
    
    def get_flow_statistics(self, flow: np.ndarray = None, old_points: np.ndarray = None, new_points: np.ndarray = None) -> Dict:
        """Compute statistics about the flow field"""
        if flow is not None:
            # Dense flow statistics
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            return {
                'mean_magnitude': float(np.mean(magnitude)),
                'max_magnitude': float(np.max(magnitude)),
                'mean_angle': float(np.mean(angle)),
                'motion_energy': float(np.sum(magnitude))  # Total motion in frame
            }
        elif old_points is not None and new_points is not None:
            # Sparse flow statistics
            displacements = new_points - old_points
            magnitudes = np.linalg.norm(displacements, axis=1)
            
            return {
                'mean_magnitude': float(np.mean(magnitudes)),
                'max_magnitude': float(np.max(magnitudes)),
                'num_tracked_points': len(old_points),
                'motion_energy': float(np.sum(magnitudes))
            }
        else:
            return {
                'mean_magnitude': 0.0,
                'max_magnitude': 0.0,
                'motion_energy': 0.0
            }
    
    def reset(self):
        """Reset the tracker state"""
        self.prev_frame = None
        self.prev_points = None

