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
    TemporalSegmentationTransform,
    VLMSegmentEventTransform,
)
from bird.events.writer import EventWriter
import cv2
import time


def run(camera, vision_config: VisionConfig):
    """
    Main pipeline runner. Builds DAG from config and displays with cv2.imshow.

    Args:
        camera: Camera instance that provides stream_frames()
        vision_config: VisionConfig with pipeline settings
    """
    dag = build_dag_from_config(vision_config)
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


def build_dag_from_config(vision_config: VisionConfig):
    """Build DAG pipeline from vision config."""
    transforms = []
    detector = None
    object_tracker = None
    depth_estimator = None

    # Depth estimation
    if vision_config.enable_depth:
        depth_estimator = DepthEstimator(model_size=vision_config.depth_model_size)
        transforms.append(DepthEstimationTransform(
            depth_estimator=depth_estimator,
            run_every_n_frames=3
        ))

    # Object detection
    if vision_config.enable_box or vision_config.enable_segmentation:
        detector = ObjectDetector(vision_config=vision_config)
        if depth_estimator:
            detector.depth_estimator = depth_estimator
        transforms.append(ObjectDetectionTransform(detector=detector))

        # Object tracking
        if vision_config.enable_tracking:
            object_tracker = SimpleTracker(
                max_age=vision_config.tracking_max_age,
                min_hits=vision_config.tracking_min_hits,
                iou_threshold=vision_config.tracking_iou_threshold
            )
            transforms.append(ObjectTrackingTransform(
                tracker=object_tracker,
                detector=detector
            ))
        else:
            transforms.append(DrawDetectionsTransform(detector=detector))

    # Events
    if vision_config.enable_events and object_tracker:
        from bird.events.motion import RegionEntryEvent, RegionExitEvent
        from bird.events.interaction import PersonObjectInteractionEvent

        frame_region = [(0, 0), (1920, 0), (1920, 1080), (0, 1080)]

        event_detectors = [
            RegionEntryEvent(region=frame_region, cooldown=1.0),
            RegionExitEvent(region=frame_region, cooldown=1.0),
            PersonObjectInteractionEvent(distance_threshold=100.0, duration_threshold=1.0, cooldown=2.0),
        ]

        transforms.append(EventDetectionTransform(detectors=event_detectors))

    # Background removal or depth visualization
    if vision_config.enable_bg_removal:
        bg_remover = BackgroundRemover(
            mode=vision_config.bg_removal_mode,
            depth_threshold=vision_config.bg_depth_threshold
        )
        transforms.append(BackgroundRemovalTransform(bg_remover=bg_remover))
    elif depth_estimator:
        transforms.append(DepthVisualizationTransform(
            depth_estimator=depth_estimator,
            alpha=vision_config.depth_alpha
        ))

    # Scene graph
    if vision_config.enable_scene_graph:
        scene_graph_builder = SceneGraphBuilder(
            use_vlm=vision_config.scene_graph_use_vlm,
            vlm_provider=vision_config.scene_graph_vlm_provider,
            vlm_model=vision_config.scene_graph_vlm_model,
            vlm_interval=vision_config.scene_graph_vlm_interval
        )
        transforms.append(SceneGraphTransform(
            scene_graph_builder=scene_graph_builder,
            run_every_n_frames=vision_config.scene_graph_vlm_interval
        ))

    # Optical flow
    if vision_config.enable_optical_flow:
        flow_tracker = OpticalFlowTracker(method=vision_config.optical_flow_method)
        transforms.append(OpticalFlowTransform(flow_tracker=flow_tracker))

    # Temporal segmentation
    if vision_config.enable_temporal_segmentation:
        from bird.vision.temporal_segmenter import TemporalSegmenter

        temporal_segmenter = TemporalSegmenter(
            model_name=vision_config.temporal_clip_model,
            similarity_threshold=vision_config.temporal_similarity_threshold,
            min_segment_length=vision_config.temporal_min_segment_length
        )

        transforms.append(TemporalSegmentationTransform(
            temporal_segmenter=temporal_segmenter,
            similarity_threshold=vision_config.temporal_similarity_threshold,
            sample_rate=1
        ))

    if vision_config.enable_vlm_events and vision_config.enable_temporal_segmentation:
        transforms.append(VLMSegmentEventTransform(
            vlm_provider=vision_config.vlm_events_provider,
            vlm_model=vision_config.vlm_events_model,
            clip_duration_frames=vision_config.vlm_events_clip_duration,
            cooldown=vision_config.vlm_events_cooldown
        ))

    transforms.append(MetricsTransform(
        detector=detector,
        tracker=object_tracker,
        depth_estimator=depth_estimator
    ))

    if vision_config.enable_overlay:
        overlay = InfoOverlay(position='right', width=250, alpha=0.7)
        transforms.append(OverlayTransform(overlay=overlay))

    if vision_config.enable_event_serialization:
        writer = EventWriter()
        transforms.append(EventSerializationTransform(serializer=writer))

    return DAG(transforms)


def process_video_streaming(camera, vision_config: VisionConfig, yield_frames: bool = False):
    """
    Process video with real-time streaming based on config.
    Yields updates as the pipeline processes each frame.

    Args:
        camera: VideoFileCamera instance
        vision_config: VisionConfig with pipeline settings
        yield_frames: If True, yields base64-encoded processed frames for synchronized playback

    Yields:
        dict with update type and data
    """
    import base64

    dag = build_dag_from_config(vision_config)

    cap = cv2.VideoCapture(str(camera.path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    yield {
        'type': 'start',
        'total_frames': total_frames,
        'fps': fps
    }

    all_events = []
    all_similarities = []
    change_points = []
    frame_count = 0
    total_event_count = 0

    for frame in camera.stream_frames():
        frame_start_time = time.time()

        state = {
            'frame': frame,
            'frame_count': frame_count,
            'timestamp': frame_start_time,
            'events': [],
        }

        state = dag.forward(state)

        # Handle events
        if state.get('events'):
            for event in state['events']:
                event_data = {
                    'frame': frame_count,
                    'timestamp': frame_start_time,
                    'type': event['type'],
                    'confidence': event['confidence'],
                    'objects': event.get('objects', []),
                    'meta': event.get('meta', {})
                }
                all_events.append(event_data)
                total_event_count += 1

                yield {
                    'type': 'event',
                    'event': event_data,
                    'frame': frame_count,
                    'progress': (frame_count / total_frames) * 100 if total_frames > 0 else 0
                }

        # Handle temporal segmentation
        frame_similarity = state.get('frame_similarity')
        is_change_point = state.get('is_change_point', False)

        if frame_similarity is not None:
            all_similarities.append(frame_similarity)

            if is_change_point:
                change_points.append(frame_count)
                yield {
                    'type': 'change_point',
                    'frame': frame_count,
                    'similarity': frame_similarity,
                    'progress': (frame_count / total_frames) * 100 if total_frames > 0 else 0
                }

        # Yield frame for synchronized playback
        if yield_frames:
            processed_frame = cv2.cvtColor(state['frame'], cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            yield {
                'type': 'frame',
                'frame': frame_count,
                'image': frame_b64,
                'progress': (frame_count / total_frames) * 100 if total_frames > 0 else 0
            }

        if frame_count % 30 == 0:
            yield {
                'type': 'progress',
                'frame': frame_count,
                'progress': (frame_count / total_frames) * 100 if total_frames > 0 else 0,
                'similarities': all_similarities.copy() if all_similarities else []
            }

        frame_count += 1

    yield {
        'type': 'complete',
        'total_events': total_event_count,
        'total_frames': frame_count,
        'events': all_events,
        'similarities': all_similarities,
        'change_points': change_points,
        'sample_rate': 1
    }

