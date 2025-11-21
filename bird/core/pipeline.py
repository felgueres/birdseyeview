from bird.config import VisionConfig
from bird.vision.detector import ObjectDetector
from bird.vision.optical_flow import OpticalFlowTracker
from bird.vision.tracker import SimpleTracker
from bird.vision.scene_graph import SceneGraphBuilder
from bird.vision.overlay import InfoOverlay
from bird.vision.depth_estimator import DepthEstimator
import json
import cv2
import time


def run(camera, vision_config: VisionConfig):
    """
    Vision pipeline that processes camera frames.
    
    Orchestrates object detection, tracking, optical flow, and scene graph
    generation based on the provided configuration.
    
    Args:
        camera: Camera instance (Webcam or SonyA5000) that provides stream_frames()
        vision_config: VisionConfig instance with pipeline settings
    """
    detector = ObjectDetector(vision_config=vision_config) if vision_config.enable_box or vision_config.enable_segmentation else None
    flow_tracker = OpticalFlowTracker(method=vision_config.optical_flow_method) if vision_config.enable_optical_flow else None
    object_tracker = SimpleTracker(
        max_age=vision_config.tracking_max_age,
        min_hits=vision_config.tracking_min_hits,
        iou_threshold=vision_config.tracking_iou_threshold
    ) if vision_config.enable_tracking else None
    scene_graph_builder = SceneGraphBuilder(
        use_vlm=vision_config.scene_graph_use_vlm,
        vlm_provider=vision_config.scene_graph_vlm_provider,
        vlm_model=vision_config.scene_graph_vlm_model,
        vlm_interval=vision_config.scene_graph_vlm_interval
    ) if vision_config.enable_scene_graph else None
    depth_estimator = DepthEstimator(
        model_size=vision_config.depth_model_size
    ) if vision_config.enable_depth else None
    
    # Initialize info overlay
    overlay = InfoOverlay(position='right', width=250, alpha=0.7)
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    fps_frame_count = 0

    for frame in camera.stream_frames():
        frame_start_time = time.time()
        events = []
        
        # 1. Depth Estimation (if enabled, run early for object depth info)
        depth_map = None
        if depth_estimator:
            depth_map = depth_estimator.estimate_depth(frame)
        
        # 2. Object Detection - detect objects
        detections = []
        tracked_objects = []
        if detector:
            detections = detector.detect_objects(frame)
            
            # Add depth info to detections
            if depth_map is not None:
                for det in detections:
                    depth_info = depth_estimator.get_depth_for_bbox(depth_map, det['bbox'])
                    det['depth'] = depth_info['mean']
                    det['depth_stats'] = depth_info
            
            # Tracks object movement
            if object_tracker:
                tracked_objects = object_tracker.update(detections)
                frame = detector.draw_tracks(frame, tracked_objects)
                
                # Track new objects
                for obj in tracked_objects:
                    if len(obj['trajectory']) == 1:  # New track
                        events.append(f"New {obj['class']}")
            # Otherwise, just draws boxes
            else:
                frame = detector.draw_detections(frame, detections)
        
        # 3. Depth Visualization (blend with frame if enabled)
        if depth_map is not None:
            frame = depth_estimator.create_depth_overlay(frame, depth_map, alpha=vision_config.depth_alpha)
        
        # 4. Scene Graph - VLM analysis and draw (overrides visualizations on VLM frames)
        if scene_graph_builder:
            scene_graph = scene_graph_builder.build_graph(frame)
            if scene_graph:
                events.append("VLM update")
        
        # 5. Optical Flow - compute and draw
        motion_energy = 0
        tracked_points = 0
        if flow_tracker:
            if vision_config.optical_flow_method == 'lucas_kanade':
                old_pts, new_pts = flow_tracker.compute_sparse_flow(frame)
                if old_pts is not None and new_pts is not None:
                    frame = flow_tracker.draw_sparse_flow(frame, old_pts, new_pts)
                    
                    stats = flow_tracker.get_flow_statistics(old_points=old_pts, new_points=new_pts)
                    motion_energy = stats['motion_energy']
                    tracked_points = stats['num_tracked_points']
        
        # Calculate FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Build metrics dictionary - keep it simple
        metrics = {
            'FPS': fps,
            'Frame': frame_count,
        }
        
        if detector:
            if object_tracker:
                metrics['Tracked'] = object_tracker.get_active_track_count()
            else:
                metrics['Detections'] = len(detections)
        
        if flow_tracker and tracked_points > 0:
            metrics['Motion'] = motion_energy
        
        if depth_map is not None:
            depth_stats = depth_estimator.get_depth_statistics(depth_map)
            metrics['Avg Depth'] = depth_stats['mean_depth']
        
        # Pipeline timing
        frame_time = (time.time() - frame_start_time) * 1000
        metrics['ms/frame'] = frame_time
        
        # Draw overlay with metrics and events
        if vision_config.enable_overlay:
            frame = overlay.draw(frame, metrics, events)
        
        # Display and control
        cv2.imshow('BirdView Camera Feed', frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

