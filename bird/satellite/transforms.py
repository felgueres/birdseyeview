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


class SolarPanelSegmentationTransform(Transform):
    def __init__(self, run_every_n_frames: int = 1, grid_size: int = 2048, aoi_bbox: Optional[List[float]] = None):
        super().__init__(
            name="solar_panel_segmentation",
            input_keys=["frame"],
            output_keys=["solar_mask", "solar_coverage_pct"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )
        self.grid_size = grid_size
        self.aoi_bbox = aoi_bbox

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]

        solar_mask = self._segment_solar_panels_grid(frame)

        total_pixels = solar_mask.size
        solar_pixels = np.sum(solar_mask > 0)
        solar_coverage_pct = 100.0 * solar_pixels / total_pixels

        return {
            "solar_mask": solar_mask,
            "solar_coverage_pct": solar_coverage_pct
        }

    def _segment_solar_panels_grid(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        print(f"  Splitting {h}x{w} frame into {self.grid_size}x{self.grid_size} grid cells")

        rows = (h + self.grid_size - 1) // self.grid_size
        cols = (w + self.grid_size - 1) // self.grid_size

        print(f"  Grid: {rows}x{cols} = {rows*cols} cells")

        for row in range(rows):
            for col in range(cols):
                y1 = row * self.grid_size
                y2 = min((row + 1) * self.grid_size, h)
                x1 = col * self.grid_size
                x2 = min((col + 1) * self.grid_size, w)

                chunk = frame[y1:y2, x1:x2]

                if chunk.size == 0:
                    continue

                print(f"  Processing grid cell [{row},{col}] ({y1}:{y2}, {x1}:{x2}) size={chunk.shape}")

                chunk_mask = self._segment_chunk(chunk)

                if chunk_mask is not None:
                    chunk_h, chunk_w = chunk_mask.shape
                    combined_mask[y1:y1+chunk_h, x1:x1+chunk_w] = np.maximum(
                        combined_mask[y1:y1+chunk_h, x1:x1+chunk_w],
                        chunk_mask
                    )

        return combined_mask

    def _segment_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        try:
            import modal
            import base64
            from PIL import Image
            import io

            Image.MAX_IMAGE_PIXELS = None

            segment_fn = modal.Function.from_name("birdview", "run_sam3_text")

            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(chunk, cv2.COLOR_RGB2BGR))
            image_bytes = buffer.tobytes()

            result = segment_fn.remote(
                image_bytes=image_bytes,
                text_prompt="solar panels"
            )

            if result and result.get("success") and result.get("masks"):
                masks_b64 = result["masks"]
                combined_mask = np.zeros(chunk.shape[:2], dtype=np.uint8)

                for mask_b64 in masks_b64:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_img = Image.open(io.BytesIO(mask_bytes))
                    mask_np = np.array(mask_img)
                    combined_mask = np.maximum(combined_mask, (mask_np > 0).astype(np.uint8))

                solar_pct = np.sum(combined_mask > 0) / combined_mask.size * 100
                print(f"    SAM3 found {solar_pct:.2f}% solar coverage in this chunk")
                return combined_mask

            return None

        except Exception as e:
            print(f"    SAM segmentation failed for chunk: {e}")
            return None


class SolarInstallationEventTransform(Transform):
    def __init__(self, min_coverage_pct: float = 5.0, run_every_n_frames: int = 1):
        super().__init__(
            name="solar_installation_event",
            input_keys=["solar_mask", "solar_coverage_pct", "frame_count", "timestamp"],
            output_keys=["events"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )
        self.min_coverage_pct = min_coverage_pct

    def forward(self, inputs: dict) -> dict:
        solar_mask = inputs.get("solar_mask")
        solar_coverage_pct = inputs.get("solar_coverage_pct", 0.0)
        frame_count = inputs.get("frame_count", 0)

        events = []

        if solar_mask is not None and solar_coverage_pct >= self.min_coverage_pct:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(solar_mask.astype(np.uint8))

            num_arrays = num_labels - 1 if num_labels > 1 else 0

            installation_type = "small_solar_installation"
            if solar_coverage_pct > 50:
                installation_type = "large_solar_farm"
            elif solar_coverage_pct > 20:
                installation_type = "medium_solar_installation"

            events.append({
                "type": installation_type,
                "confidence": min(solar_coverage_pct / 50.0, 1.0),
                "objects": [],
                "meta": {
                    "solar_coverage_pct": float(solar_coverage_pct),
                    "num_panel_arrays": num_arrays,
                    "frame_count": frame_count,
                    "description": f"Solar installation: {solar_coverage_pct:.1f}% coverage, {num_arrays} arrays"
                }
            })

        return {"events": events}


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
