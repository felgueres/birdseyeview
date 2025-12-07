"""
Process a batch of static satellite tiles through the DAG pipeline.

This processes individual tile images (not time series) through:
- Land cover segmentation
- Event extraction
- Embedding generation for semantic search

Allows you to search "solar" and get relevant tiles back.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from bird.core.dag import DAG
from bird.satellite.transforms import (
    LandCoverSegmentationTransform,
    SatelliteEventExtractionTransform,
)
from bird.events.database import EventDatabase
from bird.events.embedder import EventEmbedder
from bird.events.writer import EventWriter
from bird.core.transforms import VLMSegmentEventTransform
from bird.core.dag import Transform
import cv2
import base64


class TileVLMTransform(Transform):
    """VLM analysis specifically for static satellite tiles."""

    def __init__(self, run_every_n_frames: int = 1):
        super().__init__(
            name="tile_vlm",
            input_keys=["frame", "frame_count", "timestamp", "tile_name"],
            output_keys=["events"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )

        import os
        from openai import OpenAI
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def forward(self, inputs: dict) -> dict:
        frame = inputs["frame"]
        frame_count = inputs.get("frame_count", 0)
        timestamp = inputs.get("timestamp", 0)
        tile_name = inputs.get("tile_name", "unknown")

        events = []

        try:
            description = self._analyze_tile_with_vlm(frame)

            if description:
                events.append({
                    "type": "vlm_tile_analysis",
                    "confidence": 1.0,
                    "objects": [],
                    "meta": {
                        "description": description,
                        "frame_count": frame_count,
                        "tile_name": tile_name
                    }
                })
        except Exception as e:
            print(f"  VLM error for {tile_name}: {e}")

        return {"events": events}

    def _analyze_tile_with_vlm(self, frame: np.ndarray) -> str:
        """Analyze a single tile with VLM."""
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        base64_image = base64.b64encode(buffer).decode('utf-8')

        prompt = """Analyze this satellite imagery tile and describe what you see.
Focus on:
- Infrastructure (solar panels, buildings, roads, industrial facilities)
- Land use (agriculture, urban, desert, water)
- Notable features

Be specific and concise (2-3 sentences)."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150
        )

        return response.choices[0].message.content


class TileAnalysisTransform(Transform):
    """Generate an event for each tile based on segmentation analysis."""

    def __init__(self, run_every_n_frames: int = 1):
        super().__init__(
            name="tile_analysis",
            input_keys=["frame", "land_cover_mask", "frame_count", "timestamp", "tile_name"],
            output_keys=["events"],
            run_every_n_frames=run_every_n_frames,
            critical=False
        )

    def forward(self, inputs: dict) -> dict:
        land_cover_mask = inputs.get("land_cover_mask")
        frame_count = inputs.get("frame_count", 0)
        timestamp = inputs.get("timestamp", 0)
        tile_name = inputs.get("tile_name", "unknown")

        events = []

        if land_cover_mask is not None:
            total_pixels = land_cover_mask.size
            vegetation_pct = 100 * np.sum(land_cover_mask == 1) / total_pixels
            water_pct = 100 * np.sum(land_cover_mask == 2) / total_pixels
            built_pct = 100 * np.sum(land_cover_mask == 3) / total_pixels

            dominant_type = "background"
            if vegetation_pct > 30:
                dominant_type = "vegetation_area"
            if water_pct > 30:
                dominant_type = "water_area"
            if built_pct > 10:
                dominant_type = "built_area"

            description_parts = []
            if vegetation_pct > 10:
                description_parts.append(f"vegetation {vegetation_pct:.1f}%")
            if water_pct > 10:
                description_parts.append(f"water {water_pct:.1f}%")
            if built_pct > 5:
                description_parts.append(f"built/infrastructure {built_pct:.1f}%")

            description = "Tile contains: " + ", ".join(description_parts) if description_parts else "Mostly undeveloped land"

            event = {
                "type": dominant_type,
                "confidence": max(vegetation_pct, water_pct, built_pct) / 100.0,
                "objects": [],
                "meta": {
                    "tile_name": tile_name,
                    "vegetation_pct": float(vegetation_pct),
                    "water_pct": float(water_pct),
                    "built_pct": float(built_pct),
                    "description": description,
                    "frame_count": frame_count
                }
            }
            events.append(event)

        return {"events": events}


class TileBatchProcessor:
    """Process a batch of static tiles through DAG with searchable output."""

    def __init__(
        self,
        tiles_dir: str,
        session_dir: str = "./tile_analysis_sessions",
        enable_embeddings: bool = True,
        enable_vlm: bool = True
    ):
        self.tiles_dir = Path(tiles_dir)
        self.session_name = self.tiles_dir.name
        self.session_dir = Path(session_dir) / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.writer = EventWriter(
            session_dir=str(self.session_dir),
            enable_database=True,
            enable_embeddings=enable_embeddings
        )
        self.db = self.writer.database

        self.enable_vlm = enable_vlm
        self.dag = self._build_dag()

        self.results_dir = self.session_dir / "annotated_tiles"
        self.results_dir.mkdir(exist_ok=True)

    def _build_dag(self) -> DAG:
        """Build processing DAG for static tiles."""
        from bird.core.transforms import EventSerializationTransform

        transforms = [
            LandCoverSegmentationTransform(
                run_every_n_frames=1
            ),
            TileAnalysisTransform(
                run_every_n_frames=1
            )
        ]

        if self.enable_vlm:
            transforms.append(TileVLMTransform(run_every_n_frames=1))

        transforms.append(
            EventSerializationTransform(
                serializer=self.writer,
                run_every_n_frames=1
            )
        )

        return DAG(transforms=transforms)

    def get_tile_paths(self, pattern: str = "*.jpg") -> List[Path]:
        """Get all tile paths matching pattern."""
        tiles = sorted(self.tiles_dir.glob(pattern))
        return tiles

    def process_tiles(
        self,
        max_tiles: Optional[int] = None,
        pattern: str = "*.jpg",
        save_annotations: bool = True
    ):
        """Process all tiles through the DAG."""
        tiles = self.get_tile_paths(pattern)

        if max_tiles:
            tiles = tiles[:max_tiles]

        print(f"Processing {len(tiles)} tiles from {self.tiles_dir}")
        print(f"Session: {self.session_dir}")
        print(f"VLM enabled: {self.enable_vlm}")
        print("=" * 60)

        for i, tile_path in enumerate(tiles):
            print(f"\n[{i+1}/{len(tiles)}] Processing {tile_path.name}...")

            tile = cv2.imread(str(tile_path))
            if tile is None:
                print(f"  Failed to load {tile_path.name}, skipping")
                continue

            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

            state = {
                "frame": tile_rgb,
                "frame_count": i,
                "timestamp": float(i),
                "tile_name": tile_path.name,
                "tile_path": str(tile_path)
            }

            try:
                result = self.dag.forward(state)

                events = result.get("events", [])

                vlm_description = ""
                for event in events:
                    if event.get("type") in ["vlm_segment_event", "vlm_tile_analysis"]:
                        vlm_description = event.get("meta", {}).get("description", "")

                for event in events:
                    if "meta" not in event:
                        event["meta"] = {}
                    event["meta"]["tile_name"] = tile_path.name
                    event["meta"]["tile_path"] = str(tile_path)

                print(f"  Events: {len(events)}")
                if vlm_description:
                    print(f"  VLM: {vlm_description[:100]}...")

                if save_annotations:
                    self._save_annotated_tile(
                        tile_rgb,
                        result,
                        tile_path.name,
                        vlm_description
                    )

            except Exception as e:
                print(f"  Error processing {tile_path.name}: {e}")
                continue

        print("\n" + "=" * 60)
        print("Processing complete!")
        self.print_statistics()

    def _save_annotated_tile(
        self,
        tile: np.ndarray,
        result: dict,
        tile_name: str,
        vlm_description: str = ""
    ):
        """Save annotated tile with segmentation overlay."""
        annotated = tile.copy()

        land_cover_mask = result.get("land_cover_mask")
        if land_cover_mask is not None:
            overlay = np.zeros_like(annotated)

            overlay[land_cover_mask == 1] = [0, 255, 0]
            overlay[land_cover_mask == 2] = [0, 0, 255]
            overlay[land_cover_mask == 3] = [255, 100, 0]

            annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        events = result.get("events", [])
        for event in events:
            if "centroid" in event.get("meta", {}):
                centroid = event["meta"]["centroid"]
                x, y = int(centroid[0]), int(centroid[1])
                cv2.circle(annotated, (x, y), 10, (255, 0, 0), 2)

        if vlm_description:
            y_offset = 20
            for line in vlm_description[:200].split('\n')[:3]:
                cv2.putText(
                    annotated,
                    line[:80],
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
                y_offset += 15

        output_path = self.results_dir / tile_name
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    def print_statistics(self):
        """Print database statistics."""
        stats = self.db.get_event_statistics()
        print(f"\nTotal events detected: {stats['total_events']}")
        print(f"Event types:")
        for event_type, count in stats['by_type'].items():
            print(f"  {event_type}: {count}")

    def search_by_text(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic search for tiles matching query."""
        if not self.writer.embedder:
            print("Embeddings not enabled")
            return []

        print(f"\nSearching for: '{query}'")
        print("-" * 60)

        query_embedding = self.writer.embedder.embed_query(query)

        results = self.db.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
            model=self.writer.embedder.model_name
        )

        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results):
            tile_name = result.get("meta", {}).get("tile_name", "unknown")
            similarity = result.get("similarity", 0)
            event_type = result.get("type", "unknown")

            vlm_desc = ""
            if event_type in ["vlm_segment_event", "vlm_tile_analysis"]:
                vlm_desc = result.get("meta", {}).get("description", "")

            print(f"\n{i+1}. {tile_name} (similarity={similarity:.3f})")
            print(f"   Type: {event_type}")
            if vlm_desc:
                print(f"   Description: {vlm_desc[:200]}")

        return results

    def get_events(
        self,
        event_type: Optional[str] = None,
        min_confidence: float = 0.3,
        limit: int = 50
    ) -> List[Dict]:
        """Get events from database."""
        events = self.db.get_events(
            event_type=event_type,
            min_confidence=min_confidence,
            limit=limit
        )

        print(f"\nFound {len(events)} events:")
        for event in events[:10]:
            tile_name = event.get("meta", {}).get("tile_name", "unknown")
            print(f"  {tile_name}: {event['type']} (conf={event['confidence']:.2f})")

        return events

    def process_tiles_with_display(
        self,
        max_tiles: Optional[int] = None,
        pattern: str = "*.jpg",
        save_annotations: bool = True
    ):
        """Process tiles with live visualization display."""
        import cv2

        tiles = self.get_tile_paths(pattern)

        if max_tiles:
            tiles = tiles[:max_tiles]

        print(f"Processing {len(tiles)} tiles from {self.tiles_dir}")
        print(f"Session: {self.session_dir}")
        print(f"VLM enabled: {self.enable_vlm}")
        print("=" * 60)
        print("\nPress 'q' to quit, any other key to continue to next tile")

        for i, tile_path in enumerate(tiles):
            print(f"\n[{i+1}/{len(tiles)}] Processing {tile_path.name}...")

            tile = cv2.imread(str(tile_path))
            if tile is None:
                print(f"  Failed to load {tile_path.name}, skipping")
                continue

            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

            state = {
                "frame": tile_rgb,
                "frame_count": i,
                "timestamp": float(i),
                "tile_name": tile_path.name,
                "tile_path": str(tile_path)
            }

            try:
                result = self.dag.forward(state)

                events = result.get("events", [])

                vlm_description = ""
                for event in events:
                    if event.get("type") in ["vlm_segment_event", "vlm_tile_analysis"]:
                        vlm_description = event.get("meta", {}).get("description", "")

                for event in events:
                    if "meta" not in event:
                        event["meta"] = {}
                    event["meta"]["tile_name"] = tile_path.name
                    event["meta"]["tile_path"] = str(tile_path)

                print(f"  Events: {len(events)}")
                if vlm_description:
                    print(f"  VLM: {vlm_description[:100]}...")

                display_frame = self._create_display_frame(
                    tile_rgb,
                    result,
                    tile_path.name,
                    vlm_description
                )

                cv2.imshow("Satellite Analysis", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

                key = cv2.waitKey(0)
                if key == ord('q'):
                    print("\nStopping...")
                    break

                if save_annotations:
                    self._save_annotated_tile(
                        tile_rgb,
                        result,
                        tile_path.name,
                        vlm_description
                    )

            except Exception as e:
                print(f"  Error processing {tile_path.name}: {e}")
                continue

        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print("Processing complete!")
        self.print_statistics()

    def _create_display_frame(
        self,
        tile: np.ndarray,
        result: dict,
        tile_name: str,
        vlm_description: str = ""
    ) -> np.ndarray:
        """Create display frame with overlay and segmentation."""
        h, w = tile.shape[:2]

        display_width = w + 250
        display = np.zeros((h, display_width, 3), dtype=np.uint8)

        tile_with_overlay = tile.copy()
        land_cover_mask = result.get("land_cover_mask")

        if land_cover_mask is not None:
            overlay = np.zeros_like(tile_with_overlay)

            overlay[land_cover_mask == 1] = [0, 255, 0]
            overlay[land_cover_mask == 2] = [0, 0, 255]
            overlay[land_cover_mask == 3] = [255, 100, 0]

            tile_with_overlay = cv2.addWeighted(tile_with_overlay, 0.7, overlay, 0.3, 0)

        display[:h, :w] = tile_with_overlay

        panel_x = w
        y_offset = 30

        cv2.putText(display, tile_name, (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 40

        cv2.putText(display, "Land Cover:", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y_offset += 25

        events = result.get("events", [])
        for event in events:
            if event.get("type") in ["vegetation_area", "water_area", "built_area"]:
                meta = event.get("meta", {})

                veg_pct = meta.get("vegetation_pct", 0)
                water_pct = meta.get("water_pct", 0)
                built_pct = meta.get("built_pct", 0)

                cv2.putText(display, f"  Vegetation: {veg_pct:.1f}%",
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (0, 255, 0), 1)
                y_offset += 20

                cv2.putText(display, f"  Water: {water_pct:.1f}%",
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (0, 150, 255), 1)
                y_offset += 20

                cv2.putText(display, f"  Built: {built_pct:.1f}%",
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (255, 150, 0), 1)
                y_offset += 30
                break

        if vlm_description:
            y_offset += 10
            cv2.putText(display, "VLM Analysis:", (panel_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_offset += 25

            words = vlm_description.split()
            lines = []
            current_line = []
            max_chars = 24

            for word in words:
                test_line = ' '.join(current_line + [word])
                if len(test_line) <= max_chars:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]

            if current_line:
                lines.append(' '.join(current_line))

            for line in lines[:10]:
                cv2.putText(display, line, (panel_x + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
                y_offset += 16

        legend_y = h - 80
        cv2.putText(display, "Legend:", (panel_x + 10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        legend_y += 20

        cv2.rectangle(display, (panel_x + 10, legend_y - 8),
                     (panel_x + 25, legend_y + 2), (0, 255, 0), -1)
        cv2.putText(display, "Vegetation", (panel_x + 30, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        legend_y += 15

        cv2.rectangle(display, (panel_x + 10, legend_y - 8),
                     (panel_x + 25, legend_y + 2), (0, 0, 255), -1)
        cv2.putText(display, "Water", (panel_x + 30, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        legend_y += 15

        cv2.rectangle(display, (panel_x + 10, legend_y - 8),
                     (panel_x + 25, legend_y + 2), (255, 100, 0), -1)
        cv2.putText(display, "Built/Infrastructure", (panel_x + 30, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return display


def main():
    """Example usage."""
    processor = TileBatchProcessor(
        tiles_dir="mount_signal_tiles_512",
        session_dir="./tile_analysis_sessions",
        enable_embeddings=True,
        enable_vlm=True
    )

    processor.process_tiles(
        max_tiles=None,
        save_annotations=True
    )

    processor.search_by_text("solar panels")
    processor.search_by_text("industrial facility")
    processor.search_by_text("construction")


if __name__ == "__main__":
    main()
