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

class SatelliteEventExtractionTransform(Transform):
    """
    Extract change events from satellite imagery.

    Events:
    - construction_detected
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


class RadiometricPreprocessTransform(Transform):
    """
    Preprocess satellite imagery with normalization and spectral indices.

    Adds spectral indices that are useful for construction monitoring:
    - NDVI: vegetation (decreases during site clearing)
    - NDBI: built-up areas (increases during construction)
    - BSI: bare soil (increases during earthworks)
    """

    def __init__(
        self,
        add_indices: bool = True,
        normalize: bool = True,
        run_every_n_frames: int = 1
    ):
        super().__init__(
            name="radiometric_preprocess",
            input_keys=["frame", "frame_count"],
            output_keys=["frame", "indices", "normalized_frame"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )
        self.add_indices = add_indices
        self.normalize = normalize

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]

        normalized_frame = frame
        if self.normalize:
            normalized_frame = self._normalize(frame)

        indices = {}
        if self.add_indices and frame.shape[2] >= 3:
            indices = self._compute_indices(normalized_frame)

        return {
            "frame": frame,
            "normalized_frame": normalized_frame,
            "indices": indices
        }

    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        """Normalize to 0-1 range with percentile clipping."""
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0
        else:
            p2, p98 = np.percentile(frame, (2, 98))
            frame = np.clip((frame - p2) / (p98 - p2 + 1e-8), 0, 1)
        return frame

    def _compute_indices(self, frame: np.ndarray) -> dict:
        """Compute spectral indices from RGB (approximations for RGB-only data)."""
        indices = {}

        if frame.shape[2] == 3:
            r = frame[:, :, 0].astype(np.float32)
            g = frame[:, :, 1].astype(np.float32)
            b = frame[:, :, 2].astype(np.float32)

            indices["ndvi_approx"] = (g - r) / (g + r + 1e-8)
            indices["ndbi_approx"] = (r - g) / (r + g + 1e-8)
            indices["bsi_approx"] = ((r + b) - g) / ((r + b) + g + 1e-8)

        return indices


class BitemporalChangeSegmentationTransform(Transform):
    """
    Bitemporal change detection using segmentation.

    Produces spatial change masks showing WHERE changes occurred,
    superior to global change magnitude for construction monitoring.

    Currently uses classical approach (can be upgraded to U-Net/Siamese later).
    """

    def __init__(
        self,
        threshold: float = 0.15,
        min_change_area: int = 100,
        use_indices: bool = True,
        run_every_n_frames: int = 1
    ):
        super().__init__(
            name="bitemporal_change_segmentation",
            input_keys=["normalized_frame", "indices", "frame_count"],
            output_keys=["change_mask", "change_magnitude", "change_polygons", "previous_frame_viz"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )
        self.threshold = threshold
        self.min_change_area = min_change_area
        self.use_indices = use_indices
        self.previous_frame = None
        self.previous_indices = None

    def forward(self, inputs: dict) -> dict:
        frame = inputs.get("normalized_frame")
        indices = inputs.get("indices", {})

        change_mask = None
        change_magnitude = 0.0
        change_polygons = []
        previous_frame_viz = None

        if self.previous_frame is not None:
            change_mask, change_magnitude = self._compute_change(
                self.previous_frame,
                frame,
                self.previous_indices,
                indices
            )

            if change_mask is not None and change_magnitude > 0.01:
                change_polygons = self._extract_polygons(change_mask)

            previous_frame_viz = (self.previous_frame * 255).astype(np.uint8)

        self.previous_frame = frame.copy() if frame is not None else None
        self.previous_indices = indices.copy() if indices else None

        return {
            "change_mask": change_mask,
            "change_magnitude": change_magnitude,
            "change_polygons": change_polygons,
            "previous_frame_viz": previous_frame_viz
        }

    def _compute_change(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        indices1: dict,
        indices2: dict
    ) -> tuple[Optional[np.ndarray], float]:
        """Compute change using both RGB and spectral indices."""
        if frame1 is None or frame2 is None or frame1.shape != frame2.shape:
            return None, 0.0

        gray1 = cv2.cvtColor((frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray2 = cv2.cvtColor((frame2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        rgb_diff = np.abs(gray2 - gray1)

        if self.use_indices and indices1 and indices2 and "ndvi_approx" in indices1 and "ndvi_approx" in indices2:
            ndvi_diff = np.abs(indices2["ndvi_approx"] - indices1["ndvi_approx"])
            ndbi_diff = np.abs(indices2["ndbi_approx"] - indices1["ndbi_approx"])

            combined_diff = 0.5 * rgb_diff + 0.25 * ndvi_diff + 0.25 * ndbi_diff
        else:
            combined_diff = rgb_diff

        change_mask = (combined_diff > self.threshold).astype(np.uint8)

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

    def _extract_polygons(self, change_mask: np.ndarray) -> list:
        """Extract polygon boundaries from change mask."""
        contours, _ = cv2.findContours(
            change_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_change_area:
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                polygons.append({
                    "points": approx.reshape(-1, 2).tolist(),
                    "area": float(cv2.contourArea(contour))
                })

        return polygons
