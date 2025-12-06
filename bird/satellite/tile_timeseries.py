"""
Single-tile time series processor.

Treats satellite tile history as a video sequence:
- Each satellite pass = a frame
- 5-10 day revisit = low FPS video
- Run your existing DAG transforms on the sequence
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from bird.satellite.sentinel2 import Sentinel2Downloader, Sentinel2Tile


@dataclass
class TileTimeSeries:
    """Time series for a single tile."""
    tile_grid_id: str
    tiles: List[Sentinel2Tile]
    bbox: List[float]

    def __str__(self):
        return f"TileTimeSeries({self.tile_grid_id}, {len(self.tiles)} frames, {self.date_range()})"

    def date_range(self) -> str:
        if not self.tiles:
            return "empty"
        dates = sorted([t.date for t in self.tiles])
        return f"{dates[0]} to {dates[-1]}"


class TileTimeSeriesProcessor:
    """
    Download and process a single tile's history as a video sequence.

    This lets you use your existing video processing pipeline
    on satellite time series.
    """

    def __init__(
        self,
        tile_grid_id: str = "10SFG",
        output_dir: str = "./satellite_timeseries"
    ):
        """
        Args:
            tile_grid_id: Sentinel-2 tile grid (e.g., '10SFG' for SF Bay)
            output_dir: Directory to save frames and results
        """
        self.tile_grid_id = tile_grid_id
        self.output_dir = Path(output_dir) / tile_grid_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.downloader = Sentinel2Downloader()
        self.timeseries: Optional[TileTimeSeries] = None

    def search_tile_history(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-01",
        max_cloud_cover: float = 20
    ) -> TileTimeSeries:
        """
        Search for all historical images of this tile.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_cloud_cover: Max cloud cover %

        Returns:
            TileTimeSeries object
        """
        # Get bbox for this tile grid by searching with a large area
        # and filtering to the specific tile
        all_tiles = self.downloader.search(
            bbox=[-123.0, 36.5, -120.0, 38.5],  # Large SF Bay region
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            limit=200
        )

        # Filter to our specific tile grid
        tile_matches = []
        for tile in all_tiles:
            grid_id = tile.tile_id.split('_')[1]
            if grid_id == self.tile_grid_id:
                tile_matches.append(tile)

        # Sort by date
        tile_matches.sort(key=lambda t: t.date)

        # Get bbox from first tile
        bbox = tile_matches[0].bbox if tile_matches else []

        self.timeseries = TileTimeSeries(
            tile_grid_id=self.tile_grid_id,
            tiles=tile_matches,
            bbox=bbox
        )

        print(f"Found {len(tile_matches)} images for tile {self.tile_grid_id}")
        print(f"Date range: {self.timeseries.date_range()}")
        print(f"Bbox: {bbox}")

        return self.timeseries

    def download_timeseries(
        self,
        max_frames: Optional[int] = None,
        frame_size: tuple = (512, 512),
        skip_partial_coverage: bool = True,
        max_black_pct: float = 5.0
    ) -> List[np.ndarray]:
        """
        Download tile history as a sequence of frames.

        Args:
            max_frames: Limit number of frames (None = all)
            frame_size: Output size (width, height)
            skip_partial_coverage: Skip frames with nodata/black regions
            max_black_pct: Maximum % of black pixels allowed (default 5%)

        Returns:
            List of RGB frames as numpy arrays
        """
        if not self.timeseries:
            raise ValueError("Call search_tile_history() first")

        tiles_to_download = self.timeseries.tiles[:max_frames] if max_frames else self.timeseries.tiles

        frames = []
        metadata = []
        skipped = 0
        frame_idx = 0

        for i, tile in enumerate(tiles_to_download):
            print(f"[{i+1}/{len(tiles_to_download)}] Downloading {tile.date} (cloud={tile.cloud_cover:.1f}%)...")

            rgb = self.downloader.download_rgb_thumbnail(tile, output_size=frame_size)

            if rgb is not None:
                # Check for partial coverage (black pixels = nodata)
                if skip_partial_coverage:
                    black_pixels = np.all(rgb < 10, axis=2).sum()
                    total_pixels = rgb.shape[0] * rgb.shape[1]
                    black_pct = 100 * black_pixels / total_pixels

                    if black_pct > max_black_pct:
                        print(f"  Skipped: {black_pct:.1f}% nodata (partial satellite coverage)")
                        skipped += 1
                        continue

                frames.append(rgb)
                metadata.append({
                    'frame_idx': frame_idx,
                    'date': tile.date,
                    'cloud_cover': tile.cloud_cover,
                    'tile_id': tile.tile_id
                })

                # Save frame
                frame_path = self.output_dir / f"frame_{frame_idx:04d}_{tile.date}.jpg"
                cv2.imwrite(str(frame_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                frame_idx += 1

        print(f"\nDownloaded {len(frames)} frames ({skipped} skipped due to partial coverage)")
        print(f"Saved to: {self.output_dir}")

        # Save metadata
        import json
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'tile_grid_id': self.tile_grid_id,
                'bbox': self.timeseries.bbox,
                'frame_count': len(frames),
                'frames': metadata
            }, f, indent=2)

        return frames

    def frames_to_video(
        self,
        output_path: Optional[str] = None,
        fps: int = 2,
        add_labels: bool = True
    ):
        """
        Create video from downloaded frames.

        Args:
            output_path: Output video path (default: timeseries.mp4)
            fps: Frames per second (2 = slow, 10 = fast)
            add_labels: Add date labels to frames
        """
        if output_path is None:
            output_path = str(self.output_dir / "timeseries.mp4")

        frame_files = sorted(self.output_dir.glob("frame_*.jpg"))
        if not frame_files:
            print("No frames found. Run download_timeseries() first.")
            return

        first_frame = cv2.imread(str(frame_files[0]))
        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        import json
        meta_path = self.output_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                data = json.load(f)
                metadata = {f['frame_idx']: f for f in data['frames']}

        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))

            if add_labels and i in metadata:
                date = metadata[i]['date']
                cloud = metadata[i]['cloud_cover']

                cv2.putText(
                    frame,
                    f"{date}  |  cloud: {cloud:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    f"{date}  |  cloud: {cloud:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

            out.write(frame)

        out.release()
        print(f"Video saved: {output_path}")
        print(f"Duration: {len(frame_files)/fps:.1f}s @ {fps} fps")

    def get_frame_iterator(self):
        """
        Get iterator over frames for DAG processing.

        Yields:
            (frame_idx, timestamp, frame, metadata)
        """
        import json
        meta_path = self.output_dir / "metadata.json"

        if not meta_path.exists():
            print("No metadata found. Run download_timeseries() first.")
            return

        with open(meta_path) as f:
            data = json.load(f)

        for frame_meta in data['frames']:
            frame_idx = frame_meta['frame_idx']
            frame_path = self.output_dir / f"frame_{frame_idx:04d}_{frame_meta['date']}.jpg"

            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield (
                    frame_idx,
                    float(frame_idx),  # timestamp as frame index
                    frame,
                    frame_meta
                )


def main():
    """Example: download tile time series."""
    processor = TileTimeSeriesProcessor(
        tile_grid_id="10SFG",  # SF Bay Area tile
        output_dir="./satellite_timeseries"
    )

    timeseries = processor.search_tile_history(
        start_date="2024-06-01",
        end_date="2024-12-01",
        max_cloud_cover=15
    )

    print(f"\n{timeseries}\n")

    frames = processor.download_timeseries(max_frames=10)

    processor.frames_to_video(fps=2)

    print("\nNow you can process this with your DAG pipeline:")
    print("  from bird.satellite.tile_timeseries import TileTimeSeriesProcessor")
    print("  processor = TileTimeSeriesProcessor('10SFG')")
    print("  for frame_idx, timestamp, frame, meta in processor.get_frame_iterator():")
    print("      # Run your DAG transforms on each frame")


if __name__ == "__main__":
    main()
