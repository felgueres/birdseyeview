from bird.config import VisionConfig
from bird.vision.detector import ObjectDetector
from bird.vision.optical_flow import OpticalFlowTracker
from bird.vision.tracker import SimpleTracker
from bird.vision.scene_graph import SceneGraphBuilder
import json
import cv2


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

    frame_count = 0

    for frame in camera.stream_frames():
        # 1. Object Detection - detect objects
        detections = []
        tracked_objects = []
        if detector:
            detections = detector.detect_objects(frame)
            
            # Tracks object movement
            if object_tracker:
                tracked_objects = object_tracker.update(detections)
                frame = detector.draw_tracks(frame, tracked_objects)
                if frame_count % 30 == 0:
                    print(f"Tracking {object_tracker.get_active_track_count()} objects:")
                    for obj in tracked_objects:
                        print(f"  - ID:{obj['track_id']} {obj['class']}: {obj['confidence']:.2f} (trajectory: {len(obj['trajectory'])} points)")
            # Otherwise, just draws boxes
            else:
                frame = detector.draw_detections(frame, detections)
                if frame_count % 30 == 0 and detections:
                    print(f"Detected {len(detections)} objects:")
                    for det in detections:
                        print(f"  - {det['class']}: {det['confidence']:.2f}")
        
        # 2. Scene Graph - VLM analysis and draw (overrides visualizations on VLM frames)
        if scene_graph_builder:
            scene_graph = scene_graph_builder.build_graph(frame)
            if scene_graph:  # Only on VLM frames
                frame = scene_graph_builder.draw_scene_graph(frame, scene_graph)
                print(f"\n=== Scene Graph ===")
                print(json.dumps(scene_graph, indent=4))
                # description = scene_graph_builder.format_graph_natural_language(scene_graph)
                print(f"==================\n")
        
        # 3. Optical Flow - compute and draw
        if flow_tracker:
            if vision_config.optical_flow_method == 'lucas_kanade':
                old_pts, new_pts = flow_tracker.compute_sparse_flow(frame)
                if old_pts is not None and new_pts is not None:
                    frame = flow_tracker.draw_sparse_flow(frame, old_pts, new_pts)
                    
                    if frame_count % 30 == 0:
                        stats = flow_tracker.get_flow_statistics(old_points=old_pts, new_points=new_pts)
                        print(f"Optical Flow - Motion energy: {stats['motion_energy']:.2f}, Tracked points: {stats['num_tracked_points']}")
        
        # Display and control
        cv2.imshow('BirdView Camera Feed', frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

