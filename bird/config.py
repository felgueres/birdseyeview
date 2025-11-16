class VisionConfig:
    def __init__(self, enable_box=False, enable_mask=True, enable_pose=True):
        self.enable_box = enable_box
        self.enable_mask = enable_mask
        self.enable_pose = enable_pose
        self.model_name = 'yolov8n-pose.pt'  # Options: 'yolov8n.pt', 'yolov8n-seg.pt', 'yolov8n-pose.pt'
        self.confidence_threshold = 0.5
        self.keypoint_threshold = 0.5  # Confidence threshold for pose keypoints
