"""
Satellite-specific transforms for the DAG pipeline.

These transforms extend the video processing pipeline for satellite imagery:
- Temporal change detection across low-frequency satellite passes
- Land cover segmentation
- Change event extraction
- Georeferenced event storage
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
from bird.core.dag import Transform


class SatelliteChangeDetectionTransform(Transform):
    """
    Detect changes between satellite image pairs.

    Uses:
    - Simple pixel-wise differencing (baseline)
    - Can be upgraded to learned change detection models
    """

    def __init__(
        self,
        threshold: float = 0.15,
        min_change_area: int = 100,
        run_every_n_frames: int = 1
    ):
        super().__init__(
            name="satellite_change_detection",
            input_keys=["frame", "frame_count"],
            output_keys=["change_mask", "change_magnitude"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )
        self.threshold = threshold
        self.min_change_area = min_change_area
        self.previous_frame = None
        self.previous_frame_count = None

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        frame_count = inputs.get("frame_count", 0)

        change_mask = None
        change_magnitude = 0.0

        if self.previous_frame is not None:
            change_mask, change_magnitude = self._compute_change(
                self.previous_frame,
                frame
            )

        self.previous_frame = frame.copy()
        self.previous_frame_count = frame_count

        return {
            "change_mask": change_mask,
            "change_magnitude": change_magnitude
        }

    def _compute_change(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> tuple[Optional[np.ndarray], float]:
        """Compute change between two frames."""
        if frame1.shape != frame2.shape:
            return None, 0.0

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2

        gray1 = gray1.astype(np.float32) / 255.0
        gray2 = gray2.astype(np.float32) / 255.0

        diff = np.abs(gray2 - gray1)

        change_mask = (diff > self.threshold).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(change_mask)

        filtered_mask = np.zeros_like(change_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_change_area:
                filtered_mask[labels == i] = 1

        change_magnitude = float(np.sum(filtered_mask) / filtered_mask.size)

        return filtered_mask, change_magnitude


class LandCoverSegmentationTransform(Transform):
    """
    Placeholder for land cover segmentation.

    Future: integrate models like:
    - Prithvi (NASA/IBM)
    - SegFormer fine-tuned on satellite data
    - U-Net variants
    """

    def __init__(self, run_every_n_frames: int = 1):
        super().__init__(
            name="land_cover_segmentation",
            input_keys=["frame"],
            output_keys=["land_cover_mask"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]

        mask = self._simple_segmentation(frame)

        return {"land_cover_mask": mask}

    def _simple_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """
        Simple rule-based segmentation as placeholder.

        Classes:
        0 = background
        1 = vegetation (green)
        2 = water (blue)
        3 = built (gray/white)
        """
        if len(frame.shape) != 3:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        vegetation = (g > r) & (g > b) & (g > 100)
        mask[vegetation] = 1

        water = (b > r) & (b > g) & (b > 80)
        mask[water] = 2

        built = (np.abs(r - g) < 30) & (np.abs(g - b) < 30) & (r > 120)
        mask[built] = 3

        return mask


class SatelliteEventExtractionTransform(Transform):
    """
    Extract change events from satellite imagery.

    Events:
    - construction_detected
    - deforestation_detected
    - water_level_change
    - land_cover_change
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        run_every_n_frames: int = 1
    ):
        super().__init__(
            name="satellite_event_extraction",
            input_keys=["change_mask", "change_magnitude", "land_cover_mask", "frame_count", "timestamp"],
            output_keys=["events"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )
        self.min_confidence = min_confidence

    def forward(self, inputs: dict) -> dict:
        change_mask = inputs.get("change_mask")
        change_magnitude = inputs.get("change_magnitude", 0.0)
        land_cover_mask = inputs.get("land_cover_mask")
        frame_count = inputs.get("frame_count", 0)
        timestamp = inputs.get("timestamp", 0)

        events = []

        if change_mask is not None and change_magnitude > 0.01:
            event = self._extract_change_event(
                change_mask,
                change_magnitude,
                land_cover_mask,
                frame_count,
                timestamp
            )
            if event:
                events.append(event)

        return {"events": events}

    def _extract_change_event(
        self,
        change_mask: np.ndarray,
        change_magnitude: float,
        land_cover_mask: Optional[np.ndarray],
        frame_count: int,
        timestamp: float
    ) -> Optional[Dict]:
        """Extract structured change event."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            change_mask.astype(np.uint8)
        )

        if num_labels <= 1:
            return None

        largest_component_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        area = stats[largest_component_idx, cv2.CC_STAT_AREA]
        centroid = centroids[largest_component_idx]

        event_type = "land_cover_change"
        if land_cover_mask is not None:
            change_region = labels == largest_component_idx
            land_cover_values = land_cover_mask[change_region]
            dominant_class = np.bincount(land_cover_values.flatten()).argmax()

            if dominant_class == 3:
                event_type = "construction_detected"
            elif dominant_class == 1:
                event_type = "vegetation_change"
            elif dominant_class == 2:
                event_type = "water_level_change"

        return {
            "type": event_type,
            "confidence": min(change_magnitude * 10, 1.0),
            "objects": [],
            "meta": {
                "area_pixels": int(area),
                "centroid": centroid.tolist(),
                "change_magnitude": float(change_magnitude),
                "frame_count": frame_count
            }
        }


class GeoReferenceTransform(Transform):
    """
    Add georeferencing metadata to frames/events.

    Placeholder - would integrate with actual satellite metadata.
    """

    def __init__(
        self,
        bbox: Optional[List[float]] = None,
        run_every_n_frames: int = 30
    ):
        super().__init__(
            name="georeference",
            input_keys=["frame", "events"],
            output_keys=["geo_metadata", "events"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )
        self.bbox = bbox or [-122.5, 37.2, -121.7, 37.9]

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        events = inputs.get("events", [])

        geo_metadata = {
            "bbox": self.bbox,
            "crs": "EPSG:4326",
            "resolution_m": 10
        }

        for event in events:
            if "centroid" in event.get("meta", {}):
                event["meta"]["geo_centroid_approx"] = self._pixel_to_geo(
                    event["meta"]["centroid"],
                    frame.shape,
                    self.bbox
                )

        return {
            "geo_metadata": geo_metadata,
            "events": events
        }

    def _pixel_to_geo(
        self,
        pixel_coords: List[float],
        image_shape: tuple,
        bbox: List[float]
    ) -> List[float]:
        """Convert pixel coordinates to geographic coordinates (rough approximation)."""
        x, y = pixel_coords
        h, w = image_shape[:2]

        lon_min, lat_min, lon_max, lat_max = bbox

        lon = lon_min + (x / w) * (lon_max - lon_min)
        lat = lat_max - (y / h) * (lat_max - lat_min)

        return [lon, lat]
