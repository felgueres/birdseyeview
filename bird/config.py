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
                enable_overlay=True,
                enable_depth=False,
                depth_model_size='small',
                enable_bg_removal=False,
                bg_removal_mode='mask',
                bg_depth_threshold=0.6,
                enable_temporal_segmentation=False,
                temporal_clip_model='openai/clip-vit-base-patch32',
                temporal_similarity_threshold=0.85,
                temporal_min_segment_length=5,
                enable_events=False,
                enable_event_serialization=False):

        self.enable_box = enable_box
        self.enable_mask = enable_mask
        self.enable_pose = enable_pose
        self.enable_optical_flow = enable_optical_flow
        self.enable_classifier = enable_classifier 
        self.enable_tracking = enable_tracking
        self.enable_scene_graph = enable_scene_graph
        self.enable_segmentation = enable_segmentation
        self.enable_overlay = enable_overlay
        self.enable_depth = enable_depth
        self.depth_model_size = depth_model_size  # 'small', 'base', or 'large'
        self.depth_alpha = 0.4  # Blending factor for depth overlay
        self.enable_bg_removal = enable_bg_removal
        self.bg_removal_mode = bg_removal_mode  # 'mask', 'depth', 'combined', or 'blur'
        self.bg_depth_threshold = bg_depth_threshold
        self.enable_temporal_segmentation = enable_temporal_segmentation
        self.temporal_clip_model = temporal_clip_model
        self.temporal_similarity_threshold = temporal_similarity_threshold
        self.temporal_min_segment_length = temporal_min_segment_length
        
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
        self.enable_events = enable_events
        self.enable_event_serialization = enable_event_serialization
