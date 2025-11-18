class VisionConfig:
    def __init__(self, 
                enable_box=False, 
                enable_mask=False, 
                enable_pose=False, 
                enable_optical_flow=False, 
                enable_classifier=False, 
                enable_tracking=False, 
                enable_segmentation=False,
                enable_scene_graph=False,
                enable_overlay=True):

        self.enable_box = enable_box
        self.enable_mask = enable_mask
        self.enable_pose = enable_pose
        self.enable_optical_flow = enable_optical_flow
        self.enable_classifier = enable_classifier 
        self.enable_tracking = enable_tracking
        self.enable_scene_graph = enable_scene_graph
        self.enable_segmentation = enable_segmentation
        self.enable_overlay = enable_overlay
        
        # If segmentation is enabled, automatically enable mask drawing
        if enable_segmentation:
            self.enable_mask = True
        self.model_name = 'yolov8n-pose.pt'  # Default model
        # Options: 'yolov8n.pt', 'yolov8n-seg.pt', 'yolov8n-pose.pt'
        # Instead of single model, we could run in parallel to get more info
        if enable_segmentation:
            self.model_name = 'yolov8n-seg.pt'
        if enable_pose:
            self.model_name = 'yolov8n-pose.pt'

        self.confidence_threshold = 0.5
        self.keypoint_threshold = 0.5  # Confidence threshold for pose keypoints
        self.optical_flow_method = 'lucas_kanade'  # Options: 'lucas_kanade', 'farneback'
        # Tracking parameters
        self.tracking_max_age = 30  # Max frames to keep track without detection
        self.tracking_min_hits = 3  # Min detections before track is confirmed
        self.tracking_iou_threshold = 0.3  # Min IoU for matching
        self.draw_trajectories = True  # Draw trajectory paths
        # Scene graph parameters
        self.scene_graph_use_vlm = True  # Use VLM for semantic relationships
        self.scene_graph_vlm_provider = 'ollama'  # Options: 'ollama', 'openai'
        self.scene_graph_vlm_model = 'llava:7b'  # Model name (ollama: 'llava:7b', openai: 'gpt-4o', 'gpt-4o-mini')
        self.scene_graph_vlm_interval = 5  # Run VLM every N frames
        self.scene_graph_show_spatial = False  # Draw spatial relationships (can be cluttered)
