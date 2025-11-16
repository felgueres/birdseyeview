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
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, conf=self.config.confidence_threshold, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
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

                    if masks is not None and i < len(masks):
                        mask = masks[i].data[0].cpu().numpy()
                        detection['mask'] = mask

                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated_frame = frame.copy()
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            class_id = detection['class_id']
            center = detection['center']
            color = self.colors[class_id % len(self.colors)]

            if 'mask' in detection:
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

            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.circle(annotated_frame, center, 5, color, -1)

            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)

            cv2.putText(annotated_frame, label, 
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated_frame