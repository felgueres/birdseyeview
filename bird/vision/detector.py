import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict
from bird.config import VisionConfig

class ObjectDetector:
    def __init__(self, vision_config: VisionConfig):
        self.model = YOLO(vision_config.model_name)
        self.config = vision_config
        self.class_names = self.model.names
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0)
        ]
        
        # Skeleton connections for pose visualization (COCO format)
        # Connections between keypoint indices [start, end]
        self.skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # legs
            [5, 11], [6, 12], [5, 6],  # torso
            [5, 7], [6, 8], [7, 9], [8, 10],  # arms
            [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]  # face to shoulders
        ]
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, conf=self.config.confidence_threshold, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
            keypoints = result.keypoints if hasattr(result, 'keypoints') and result.keypoints is not None else None
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    detection = {
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (center_x, center_y),
                        'class_id': class_id
                    }

                    # Add mask if available (for segmentation models)
                    if masks is not None and i < len(masks):
                        mask = masks[i].data[0].cpu().numpy()
                        detection['mask'] = mask
                    
                    # Add keypoints if available (for pose models)
                    if keypoints is not None and i < len(keypoints):
                        kpts = keypoints[i].data[0].cpu().numpy()  # Shape: (17, 3) - x, y, confidence
                        detection['keypoints'] = kpts

                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated_frame = frame.copy()
        if not any([self.config.enable_pose, self.config.enable_mask, self.config.enable_box, self.config.enable_classifier]):
            return frame
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            class_id = detection['class_id']
            center = detection['center']
            color = self.colors[class_id % len(self.colors)]
            label = f"{class_name}: {confidence:.2f}"
            if self.config.enable_classifier:
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1] - label_size[1] - 10), (bbox[0] + label_size[0], bbox[1]), color, -1)
                cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Pose point 
            if 'keypoints' in detection and self.config.enable_pose:
                keypoints = detection['keypoints']
                
                # Draw skeleton connections first (so they appear under the keypoints)
                for connection in self.skeleton:
                    start_idx, end_idx = connection[0], connection[1]
                    if start_idx < len(keypoints) and end_idx < len(keypoints):
                        start_point = keypoints[start_idx]
                        end_point = keypoints[end_idx]
                        
                        # Only draw if both keypoints have sufficient confidence
                        if start_point[2] > self.config.keypoint_threshold and end_point[2] > self.config.keypoint_threshold:
                            start_pos = (int(start_point[0]), int(start_point[1]))
                            end_pos = (int(end_point[0]), int(end_point[1]))
                            cv2.line(annotated_frame, start_pos, end_pos, (0, 255, 0), 2)
                
                # Draw keypoints on top
                for kpt in keypoints:
                    if kpt[2] > self.config.keypoint_threshold:  # confidence threshold
                        cv2.circle(annotated_frame, (int(kpt[0]), int(kpt[1])), 4, (0, 0, 255), -1)

            # Segmentation
            if 'mask' in detection and self.config.enable_mask:
                mask = detection['mask']
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_binary = (mask > 0.5).astype(np.uint8)
                colored_mask = np.zeros_like(annotated_frame)
                colored_mask[:] = color
                mask_indices = mask_binary == 1
                annotated_frame[mask_indices] = cv2.addWeighted(
                    annotated_frame[mask_indices], 0.6,
                    colored_mask[mask_indices], 0.4, 0
                )

            # Bounding boxes 
            if self.config.enable_box:
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.circle(annotated_frame, center, 5, color, -1)

        return annotated_frame
    
    def draw_tracks(self, frame: np.ndarray, tracked_objects: List[Dict]) -> np.ndarray:
        """Draw tracking information including IDs and trajectories"""
        annotated_frame = frame.copy()
        
        for obj in tracked_objects:
            track_id = obj['track_id']
            bbox = obj['bbox']
            center = obj['center']
            class_name = obj['class']
            confidence = obj['confidence']
            trajectory = obj.get('trajectory', [])
            
            # Color based on track ID (consistent color per track)
            color = self.colors[track_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw track ID and class
            label = f"ID:{track_id} {class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            cv2.putText(annotated_frame, label, 
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw trajectory
            if self.config.draw_trajectories and len(trajectory) > 1:
                points = np.array([(int(x), int(y)) for x, y, _ in trajectory], dtype=np.int32)
                cv2.polylines(annotated_frame, [points], False, color, 2)
                
                # Draw dots along trajectory
                for x, y, _ in trajectory[-10:]:  # Last 10 points
                    cv2.circle(annotated_frame, (int(x), int(y)), 2, color, -1)
            
            # Draw center point
            cv2.circle(annotated_frame, center, 5, color, -1)
        
        return annotated_frame