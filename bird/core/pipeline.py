from bird.config import VisionConfig
from bird.vision.detector import ObjectDetector
from bird.vision.optical_flow import OpticalFlowTracker
from bird.vision.tracker import SimpleTracker
from bird.vision.scene_graph import SceneGraphBuilder
from bird.vision.overlay import InfoOverlay
from bird.vision.depth_estimator import DepthEstimator
from bird.vision.background_remover import BackgroundRemover
from bird.core.dag import DAG
from bird.core.transforms import (
    DepthEstimationTransform,
    ObjectDetectionTransform,
    ObjectTrackingTransform,
    DrawDetectionsTransform,
    BackgroundRemovalTransform,
    DepthVisualizationTransform,
    SceneGraphTransform,
    OpticalFlowTransform,
    MetricsTransform,
    OverlayTransform,
    EventDetectionTransform,
    EventSerializationTransform,
)
from bird.events.serializer import EventSerializer
import cv2
import time


def run(camera, vision_config: VisionConfig):
    """
    Vision pipeline that processes camera frames using DAG-based transforms.

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
    bg_remover = BackgroundRemover(
        mode=vision_config.bg_removal_mode,
        depth_threshold=vision_config.bg_depth_threshold
    ) if vision_config.enable_bg_removal else None

    overlay = InfoOverlay(position='right', width=250, alpha=0.7)

    serializer = EventSerializer() if vision_config.enable_event_serialization else None

    transforms = []

    if depth_estimator:
        transforms.append(DepthEstimationTransform(
            depth_estimator=depth_estimator,
            run_every_n_frames=3
        ))

    if detector:
        detector.depth_estimator = depth_estimator
        transforms.append(ObjectDetectionTransform(detector=detector))

        if object_tracker:
            transforms.append(ObjectTrackingTransform(
                tracker=object_tracker,
                detector=detector
            ))
        else:
            transforms.append(DrawDetectionsTransform(detector=detector))

    if vision_config.enable_events and object_tracker:
        from bird.events.motion import RegionEntryEvent, RegionExitEvent
        from bird.events.interaction import PersonObjectInteractionEvent

        # Define region for entry/exit (whole frame by default)
        frame_region = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]

        event_detectors = [
            RegionEntryEvent(region=frame_region, cooldown=1.0),
            RegionExitEvent(region=frame_region, cooldown=1.0),
            PersonObjectInteractionEvent(distance_threshold=100.0, duration_threshold=1.0, cooldown=2.0),
        ]

        transforms.append(EventDetectionTransform(detectors=event_detectors))

    if bg_remover:
        transforms.append(BackgroundRemovalTransform(bg_remover=bg_remover))
    elif depth_estimator:
        transforms.append(DepthVisualizationTransform(
            depth_estimator=depth_estimator,
            alpha=vision_config.depth_alpha
        ))

    if scene_graph_builder:
        transforms.append(SceneGraphTransform(
            scene_graph_builder=scene_graph_builder,
            run_every_n_frames=vision_config.scene_graph_vlm_interval
        ))

    if flow_tracker:
        transforms.append(OpticalFlowTransform(flow_tracker=flow_tracker))

    transforms.append(MetricsTransform(
        detector=detector,
        tracker=object_tracker,
        depth_estimator=depth_estimator
    ))

    if vision_config.enable_overlay:
        transforms.append(OverlayTransform(overlay=overlay))

    if serializer:
        transforms.append(EventSerializationTransform(serializer=serializer))

    dag = DAG(transforms)

    frame_count = 0

    for frame in camera.stream_frames():
        frame_start_time = time.time()

        state = {
            'frame': frame,
            'frame_count': frame_count,
            'timestamp': frame_start_time,
            'events': [],
        }

        state = dag.forward(state)

        frame_time = (time.time() - frame_start_time) * 1000
        state['frame_time'] = frame_time

        cv2.imshow('BirdView Camera Feed', state['frame'])
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

