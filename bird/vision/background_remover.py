import cv2
import numpy as np
from typing import List, Dict, Optional


class BackgroundRemover:
    """
    Remove background from video using segmentation masks and/or depth maps.
    """
    
    def __init__(self, mode='mask', depth_threshold=0.6):
        """
        Args:
            mode: 'mask', 'depth', or 'combined'
            depth_threshold: Objects closer than this are kept (0-1, higher=closer)
        """
        self.mode = mode
        self.depth_threshold = depth_threshold
        self.background_color = (0, 255, 0)  # Green screen by default
    
    def set_background_color(self, color):
        """Set background replacement color (B, G, R)"""
        self.background_color = color
    
    def remove_with_masks(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Remove background using object segmentation masks.
        
        Args:
            frame: Original frame
            detections: List of detections with 'mask' key
            
        Returns:
            Frame with background removed
        """
        # Create combined foreground mask
        h, w = frame.shape[:2]
        foreground_mask = np.zeros((h, w), dtype=np.uint8)
        
        for det in detections:
            if 'mask' in det:
                mask = det['mask']
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h))
                
                # Add to foreground
                foreground_mask = np.maximum(foreground_mask, (mask > 0.5).astype(np.uint8))
        
        # Apply mask
        result = self._apply_mask(frame, foreground_mask)
        return result
    
    def remove_with_depth(self, frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Remove background using depth information.
        Objects closer than threshold are kept.
        
        Args:
            frame: Original frame
            depth_map: Depth map [0, 1] where higher = closer
            
        Returns:
            Frame with background removed
        """
        # Create mask from depth
        foreground_mask = (depth_map > self.depth_threshold).astype(np.uint8)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        
        result = self._apply_mask(frame, foreground_mask)
        return result
    
    def remove_combined(self, frame: np.ndarray, detections: List[Dict], 
                       depth_map: np.ndarray) -> np.ndarray:
        """
        Remove background using both masks and depth (best quality).
        
        Args:
            frame: Original frame
            detections: List of detections with masks
            depth_map: Depth map
            
        Returns:
            Frame with background removed
        """
        h, w = frame.shape[:2]
        
        # Get mask-based foreground
        mask_foreground = np.zeros((h, w), dtype=np.uint8)
        for det in detections:
            if 'mask' in det:
                mask = det['mask']
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h))
                mask_foreground = np.maximum(mask_foreground, (mask > 0.5).astype(np.uint8))
        
        # Get depth-based foreground
        depth_foreground = (depth_map > self.depth_threshold).astype(np.uint8)
        
        # Combine: take union (anything detected by either method)
        combined_mask = np.maximum(mask_foreground, depth_foreground)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        result = self._apply_mask(frame, combined_mask)
        return result
    
    def _apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply foreground mask and replace background"""
        # Create background
        background = np.full_like(frame, self.background_color)
        
        # Blur mask edges for smoother transition
        mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        mask_3channel = np.stack([mask_blurred] * 3, axis=-1)
        
        # Blend foreground and background
        result = (frame * mask_3channel + background * (1 - mask_3channel)).astype(np.uint8)
        
        return result
    
    def replace_with_image(self, frame: np.ndarray, mask: np.ndarray, 
                          background_image: np.ndarray) -> np.ndarray:
        """
        Replace background with another image.
        
        Args:
            frame: Original frame
            mask: Foreground mask (0-1)
            background_image: Background image (must match frame size)
        """
        if background_image.shape != frame.shape:
            background_image = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
        
        mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        mask_3channel = np.stack([mask_blurred] * 3, axis=-1)
        
        result = (frame * mask_3channel + background_image * (1 - mask_3channel)).astype(np.uint8)
        return result
    
    def blur_background(self, frame: np.ndarray, mask: np.ndarray, 
                       blur_amount: int = 25) -> np.ndarray:
        """
        Blur background instead of replacing it (like Zoom virtual background).
        
        Args:
            frame: Original frame
            mask: Foreground mask
            blur_amount: Blur kernel size (odd number)
        """
        # Make sure blur_amount is odd
        if blur_amount % 2 == 0:
            blur_amount += 1
        
        # Blur the entire frame
        blurred = cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)
        
        # Blend sharp foreground with blurred background
        mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        mask_3channel = np.stack([mask_blurred] * 3, axis=-1)
        
        result = (frame * mask_3channel + blurred * (1 - mask_3channel)).astype(np.uint8)
        return result

