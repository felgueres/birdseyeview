import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, deque

class SimpleTracker:
    """
    Simple IoU-based tracker for real-time object tracking.
    Tracks objects across frames by matching bounding boxes using Intersection over Union (IoU).
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum consecutive detections before track is confirmed
            iou_threshold: Minimum IoU for matching detection to track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []  # Active tracks
        self.next_id = 0
        self.frame_count = 0
        
        # Store trajectories: track_id -> deque of (x, y, frame) tuples
        self.trajectories = defaultdict(lambda: deque(maxlen=100))
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox', 'class', 'confidence', etc.
            
        Returns:
            List of tracked objects with added 'track_id' field
        """
        self.frame_count += 1
        
        # Extract bounding boxes from detections
        if len(detections) == 0:
            det_bboxes = np.empty((0, 4))
        else:
            det_bboxes = np.array([d['bbox'] for d in detections])
        
        # Match detections to existing tracks
        matched_indices, unmatched_dets, unmatched_trks = self._match_detections_to_tracks(det_bboxes)
        
        # Update matched tracks
        for det_idx, trk_idx in matched_indices:
            self.tracks[trk_idx]['bbox'] = detections[det_idx]['bbox']
            self.tracks[trk_idx]['hits'] += 1
            self.tracks[trk_idx]['age'] = 0
            self.tracks[trk_idx]['class'] = detections[det_idx]['class']
            self.tracks[trk_idx]['confidence'] = detections[det_idx]['confidence']
            
            # Preserve mask and keypoints if present
            if 'mask' in detections[det_idx]:
                self.tracks[trk_idx]['mask'] = detections[det_idx]['mask']
            if 'keypoints' in detections[det_idx]:
                self.tracks[trk_idx]['keypoints'] = detections[det_idx]['keypoints']
            
            # Update trajectory
            center = detections[det_idx]['center']
            self.trajectories[self.tracks[trk_idx]['id']].append(
                (center[0], center[1], self.frame_count)
            )
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = {
                'id': self.next_id,
                'bbox': detections[det_idx]['bbox'],
                'class': detections[det_idx]['class'],
                'confidence': detections[det_idx]['confidence'],
                'hits': 1,
                'age': 0
            }
            # Preserve mask and keypoints if present
            if 'mask' in detections[det_idx]:
                new_track['mask'] = detections[det_idx]['mask']
            if 'keypoints' in detections[det_idx]:
                new_track['keypoints'] = detections[det_idx]['keypoints']
            
            self.tracks.append(new_track)
            
            # Initialize trajectory
            center = detections[det_idx]['center']
            self.trajectories[self.next_id].append(
                (center[0], center[1], self.frame_count)
            )
            
            self.next_id += 1
        
        # Age unmatched tracks
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx]['age'] += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
        
        # Return confirmed tracks (with track_id added to original detection format)
        tracked_objects = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits:
                tracked_obj = {
                    'track_id': track['id'],
                    'bbox': track['bbox'],
                    'center': (
                        int((track['bbox'][0] + track['bbox'][2]) / 2),
                        int((track['bbox'][1] + track['bbox'][3]) / 2)
                    ),
                    'class': track['class'],
                    'confidence': track['confidence'],
                    'trajectory': list(self.trajectories[track['id']])
                }
                # Preserve mask and keypoints if present
                if 'mask' in track:
                    tracked_obj['mask'] = track['mask']
                if 'keypoints' in track:
                    tracked_obj['keypoints'] = track['keypoints']
                    
                tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def _match_detections_to_tracks(self, det_bboxes: np.ndarray) -> Tuple[List, List, List]:
        """
        Match detections to existing tracks using IoU.
        
        Returns:
            matched_indices: List of (detection_idx, track_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if len(self.tracks) == 0:
            return [], list(range(len(det_bboxes))), []
        
        # Get track bounding boxes
        trk_bboxes = np.array([t['bbox'] for t in self.tracks])
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(det_bboxes, trk_bboxes)
        
        # Perform matching using greedy algorithm
        matched_indices = []
        unmatched_detections = list(range(len(det_bboxes)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        # Greedy matching: find best matches above threshold
        while iou_matrix.size > 0:
            if iou_matrix.max() < self.iou_threshold:
                break
            
            # Find best match
            det_idx, trk_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            
            matched_indices.append((unmatched_detections[det_idx], unmatched_tracks[trk_idx]))
            
            # Remove matched detection and track
            iou_matrix = np.delete(iou_matrix, det_idx, axis=0)
            iou_matrix = np.delete(iou_matrix, trk_idx, axis=1)
            unmatched_detections.pop(det_idx)
            unmatched_tracks.pop(trk_idx)
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def _compute_iou_matrix(self, bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        """
        Compute IoU between two sets of bounding boxes.
        
        Args:
            bboxes1: Array of shape (N, 4) with format [x1, y1, x2, y2]
            bboxes2: Array of shape (M, 4) with format [x1, y1, x2, y2]
            
        Returns:
            IoU matrix of shape (N, M)
        """
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.zeros((len(bboxes1), len(bboxes2)))
        
        # Compute intersection
        x1 = np.maximum(bboxes1[:, None, 0], bboxes2[:, 0])
        y1 = np.maximum(bboxes1[:, None, 1], bboxes2[:, 1])
        x2 = np.minimum(bboxes1[:, None, 2], bboxes2[:, 2])
        y2 = np.minimum(bboxes1[:, None, 3], bboxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Compute union
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        union = area1[:, None] + area2 - intersection
        
        # Compute IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def get_active_track_count(self) -> int:
        """Get number of currently active tracks"""
        return len([t for t in self.tracks if t['hits'] >= self.min_hits])
    
    def get_trajectory(self, track_id: int) -> List[Tuple[int, int, int]]:
        """Get trajectory for a specific track ID"""
        return list(self.trajectories[track_id])
    
    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """Get all trajectories"""
        return dict(self.trajectories)
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0
        self.trajectories.clear()

