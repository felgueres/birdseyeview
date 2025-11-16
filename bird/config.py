class VisionConfig:
    def __init__(self, enable_box=False, enable_mask=False, enable_pose=False, enable_optical_flow=True, enable_classifier=False):
        self.enable_box = enable_box
        self.enable_mask = enable_mask
        self.enable_pose = enable_pose
        self.enable_optical_flow = enable_optical_flow
        self.enable_classifier = enable_classifier 
        self.model_name = 'yolov8n-pose.pt'  # Options: 'yolov8n.pt', 'yolov8n-seg.pt', 'yolov8n-pose.pt'
        self.confidence_threshold = 0.5
        self.keypoint_threshold = 0.5  # Confidence threshold for pose keypoints
        self.optical_flow_method = 'lucas_kanade'  # Options: 'lucas_kanade', 'farneback'
