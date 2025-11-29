import numpy as np
import cv2
import time
from bird.core.dag import Transform
from bird.vision.detector import ObjectDetector
from bird.vision.tracker import SimpleTracker
from bird.vision.depth_estimator import DepthEstimator
from bird.vision.background_remover import BackgroundRemover
from bird.vision.scene_graph import SceneGraphBuilder
from bird.vision.optical_flow import OpticalFlowTracker
from bird.vision.overlay import InfoOverlay


class DepthEstimationTransform(Transform):
    def __init__(self, depth_estimator: DepthEstimator, run_every_n_frames: int = 3):
        super().__init__(
            name="depth_estimation",
            input_keys=["frame"],
            output_keys=["depth_map"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )
        self.depth_estimator = depth_estimator

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        depth_map = self.depth_estimator.estimate_depth(frame)
        return {"depth_map": depth_map}


class ObjectDetectionTransform(Transform):
    def __init__(self, detector: ObjectDetector):
        super().__init__(
            name="object_detection",
            input_keys=["frame", "depth_map"],
            output_keys=["detections"],
            run_every_n_frames=1,
            critical=True
        )
        self.detector = detector

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        depth_map = inputs.get("depth_map")

        detections = self.detector.detect_objects(frame)

        if depth_map is not None and hasattr(self.detector, 'depth_estimator'):
            for det in detections:
                depth_info = self.detector.depth_estimator.get_depth_for_bbox(depth_map, det['bbox'])
                det['depth'] = depth_info['mean']
                det['depth_stats'] = depth_info

        return {"detections": detections}


class ObjectTrackingTransform(Transform):
    def __init__(self, tracker: SimpleTracker, detector: ObjectDetector):
        super().__init__(
            name="object_tracking",
            input_keys=["frame", "detections", "events"],
            output_keys=["tracked_objects", "events", "frame"],
            run_every_n_frames=1,
            critical=False
        )
        self.tracker = tracker
        self.detector = detector

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        detections = inputs.get("detections", [])
        events = inputs.get("events", [])

        tracked_objects = self.tracker.update(detections)
        frame = self.detector.draw_tracks(frame, tracked_objects)

        for obj in tracked_objects:
            if len(obj['trajectory']) == 1:
                events.append(f"New {obj['class']}")

        return {
            "tracked_objects": tracked_objects,
            "events": events,
            "frame": frame
        }


class DrawDetectionsTransform(Transform):
    def __init__(self, detector: ObjectDetector):
        super().__init__(
            name="draw_detections",
            input_keys=["frame", "detections"],
            output_keys=["frame"],
            run_every_n_frames=1,
            critical=False
        )
        self.detector = detector

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        detections = inputs.get("detections", [])

        frame = self.detector.draw_detections(frame, detections)

        return {"frame": frame}


class BackgroundRemovalTransform(Transform):
    def __init__(self, bg_remover: BackgroundRemover):
        super().__init__(
            name="background_removal",
            input_keys=["frame", "detections", "tracked_objects", "depth_map"],
            output_keys=["frame"],
            run_every_n_frames=1,
            critical=False
        )
        self.bg_remover = bg_remover

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        detections = inputs.get("detections", [])
        tracked_objects = inputs.get("tracked_objects", [])
        depth_map = inputs.get("depth_map")

        objects = tracked_objects if tracked_objects else detections

        if self.bg_remover.mode == 'mask' and objects:
            frame = self.bg_remover.remove_with_masks(frame, objects)
        elif self.bg_remover.mode == 'depth' and depth_map is not None:
            frame = self.bg_remover.remove_with_depth(frame, depth_map)
        elif self.bg_remover.mode == 'combined' and depth_map is not None and objects:
            frame = self.bg_remover.remove_combined(frame, objects, depth_map)
        elif self.bg_remover.mode == 'blur' and objects:
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for obj in objects:
                if 'mask' in obj:
                    obj_mask = obj['mask']
                    if obj_mask.shape != (h, w):
                        obj_mask = cv2.resize(obj_mask, (w, h))
                    mask = np.maximum(mask, (obj_mask > 0.5).astype(np.uint8))
            frame = self.bg_remover.blur_background(frame, mask, blur_amount=25)

        return {"frame": frame}


class DepthVisualizationTransform(Transform):
    def __init__(self, depth_estimator: DepthEstimator, alpha: float = 0.4):
        super().__init__(
            name="depth_visualization",
            input_keys=["frame", "depth_map"],
            output_keys=["frame"],
            run_every_n_frames=1,
            critical=False
        )
        self.depth_estimator = depth_estimator
        self.alpha = alpha

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        depth_map = inputs.get("depth_map")

        if depth_map is not None:
            frame = self.depth_estimator.create_depth_overlay(frame, depth_map, alpha=self.alpha)

        return {"frame": frame}


class SceneGraphTransform(Transform):
    def __init__(self, scene_graph_builder: SceneGraphBuilder, run_every_n_frames: int = 30):
        super().__init__(
            name="scene_graph",
            input_keys=["frame", "events"],
            output_keys=["scene_graph", "events"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )
        self.scene_graph_builder = scene_graph_builder

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        events = inputs.get("events", [])

        scene_graph = self.scene_graph_builder.build_graph(frame)

        if scene_graph:
            events.append("VLM update")

        return {"scene_graph": scene_graph, "events": events}


class OpticalFlowTransform(Transform):
    def __init__(self, flow_tracker: OpticalFlowTracker):
        super().__init__(
            name="optical_flow",
            input_keys=["frame"],
            output_keys=["frame", "motion_energy", "tracked_points"],
            run_every_n_frames=1,
            critical=False
        )
        self.flow_tracker = flow_tracker

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]

        motion_energy = 0
        tracked_points = 0

        old_pts, new_pts = self.flow_tracker.compute_sparse_flow(frame)
        if old_pts is not None and new_pts is not None:
            frame = self.flow_tracker.draw_sparse_flow(frame, old_pts, new_pts)

            stats = self.flow_tracker.get_flow_statistics(old_points=old_pts, new_points=new_pts)
            motion_energy = stats['motion_energy']
            tracked_points = stats['num_tracked_points']

        return {
            "frame": frame,
            "motion_energy": motion_energy,
            "tracked_points": tracked_points
        }


class MetricsTransform(Transform):
    def __init__(self, detector: ObjectDetector = None, tracker: SimpleTracker = None,
                 depth_estimator: DepthEstimator = None):
        super().__init__(
            name="metrics",
            input_keys=["frame_count", "detections", "tracked_objects", "motion_energy",
                       "tracked_points", "depth_map", "frame_time"],
            output_keys=["metrics"],
            run_every_n_frames=1,
            critical=False
        )
        self.detector = detector
        self.tracker = tracker
        self.depth_estimator = depth_estimator
        self.fps = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0

    def forward(self, inputs: dict) -> dict:
        frame_count = inputs.get("frame_count", 0)
        detections = inputs.get("detections", [])
        tracked_objects = inputs.get("tracked_objects", [])
        motion_energy = inputs.get("motion_energy", 0)
        tracked_points = inputs.get("tracked_points", 0)
        depth_map = inputs.get("depth_map")
        frame_time = inputs.get("frame_time", 0)

        self.fps_frame_count += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.fps = self.fps_frame_count / (time.time() - self.fps_start_time)
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

        metrics = {
            'FPS': self.fps,
            'Frame': frame_count,
        }

        if self.detector:
            if self.tracker:
                metrics['Tracked'] = self.tracker.get_active_track_count()
            else:
                metrics['Detections'] = len(detections)

        if tracked_points > 0:
            metrics['Motion'] = motion_energy

        if depth_map is not None and self.depth_estimator:
            depth_stats = self.depth_estimator.get_depth_statistics(depth_map)
            metrics['Avg Depth'] = depth_stats['mean_depth']

        metrics['ms/frame'] = frame_time

        return {"metrics": metrics}


class OverlayTransform(Transform):
    def __init__(self, overlay: InfoOverlay):
        super().__init__(
            name="overlay",
            input_keys=["frame", "metrics", "events"],
            output_keys=["frame"],
            run_every_n_frames=1,
            critical=False
        )
        self.overlay = overlay

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        metrics = inputs.get("metrics", {})
        events = inputs.get("events", [])

        frame = self.overlay.draw(frame, metrics, events)

        return {"frame": frame}
