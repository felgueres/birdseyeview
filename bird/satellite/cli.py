import argparse
from pathlib import Path
from typing import List
from datetime import datetime, timedelta
from dotenv import load_dotenv
import cv2
import numpy as np

from bird.satellite.sentinel2 import Sentinel2Downloader
from bird.satellite.aoi import get_aoi
from bird.satellite.create_timeseries_gif import create_gif_from_tiles
from bird.satellite.transforms import (
    RadiometricPreprocessTransform,
    BitemporalChangeSegmentationTransform
)
from bird.core.transforms import EventSerializationTransform
from bird.events.database import EventDatabase
from bird.events.writer import EventWriter
from bird.core.dag import DAG

load_dotenv()


def _detect_snow(frame: np.ndarray, threshold: float = 0.6) -> tuple[bool, float]:
    """
    Detect if image is covered with snow.

    Snow detection heuristics:
    - High brightness (RGB values all high)
    - Low color variance (all channels similar)
    - High percentage of bright pixels
    """
    if frame.dtype == np.uint8:
        frame_float = frame.astype(np.float32) / 255.0
    else:
        frame_float = frame

    brightness = np.mean(frame_float)

    r, g, b = frame_float[:, :, 0], frame_float[:, :, 1], frame_float[:, :, 2]
    color_variance = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))

    bright_pixels = np.sum((frame_float > 0.7).all(axis=2))
    bright_percentage = bright_pixels / (frame_float.shape[0] * frame_float.shape[1])

    snow_score = (brightness * 0.4) + ((1.0 - color_variance) * 0.3) + (bright_percentage * 0.3)

    is_snow = snow_score > threshold

    return is_snow, snow_score


def _visualize_change_mask(frame_before: np.ndarray, frame_after: np.ndarray, change_mask: np.ndarray) -> np.ndarray:
    """Create before/after/diff visualization."""
    if change_mask is None or frame_before is None:
        return frame_after.copy()

    overlay = frame_after.copy()
    change_overlay = np.zeros_like(frame_after)
    change_overlay[change_mask > 0] = [255, 0, 0]

    alpha = 0.5
    diff_viz = cv2.addWeighted(overlay, 1 - alpha, change_overlay, alpha, 0)

    cv2.putText(diff_viz, "Change", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return diff_viz


def _create_panel_display(panels: List[np.ndarray], frame_idx: int, filename: str) -> np.ndarray:
    """Create multi-panel display from visualization panels (horizontal layout)."""
    if not panels:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    target_height = 400
    resized_panels = []
    for panel in panels:
        h, w = panel.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(panel, (new_w, target_height))
        resized_panels.append(resized)

    combined = np.hstack(resized_panels)

    header_height = 40
    header = np.zeros((header_height, combined.shape[1], 3), dtype=np.uint8)

    cv2.putText(header, f"Frame {frame_idx}: {filename}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    final = np.vstack([header, combined])

    return final

def download_command(
    bbox: List[float],
    output_dir: str,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    max_cloud_cover: float = 20,
    tile_size: tuple = None,
    bands: List[str] = None
) -> None:
    """
    Download satellite imagery for date range.

    Args:
        bbox: [lon_min, lat_min, lon_max, lat_max]
        output_dir: Where to save tiles
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_cloud_cover: Maximum cloud cover percentage
        tile_size: (width, height) for tile resolution. Use None for full native resolution.
        bands: List of bands to download (e.g., ['red', 'green', 'blue', 'nir']).
               Defaults to RGB.
    """
    bands = ['red', 'green', 'blue'] if bands is None else bands

    downloader = Sentinel2Downloader()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    print(f"{'='*60}")
    print(f"Satellite Download")
    print(f"Bbox: {bbox}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Max cloud cover: {max_cloud_cover}%")
    print(f"Tile size: {tile_size if tile_size else 'Full native resolution'}")
    print(f"Bands: {', '.join(bands)}")
    print(f"{'='*60}\n")

    downloaded_count = 0
    skipped_count = 0

    current = start.replace(day=1)
    while current <= end:
        if current.year == end.year and current.month == end.month:
            month_end = end
        else:
            next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            month_end = next_month - timedelta(days=1)

        month_start = current

        print(f"\nSearching {month_start.strftime('%Y-%m')}...")
        print(f"  Date range: {month_start.date()} to {month_end.date()}")

        tiles = downloader.search(
            bbox=bbox,
            start_date=month_start.strftime("%Y-%m-%d"),
            end_date=month_end.strftime("%Y-%m-%d"),
            max_cloud_cover=max_cloud_cover,
            limit=10
        )

        if not tiles:
            print(f"  No tiles found with <{max_cloud_cover}% cloud cover")
            skipped_count += 1
        else:
            best_tile = tiles[-1]
            print(f"  Found: {best_tile}")

            tile_filename = f"{best_tile.date}_{best_tile.tile_id}.jpg"
            tile_path = output_path / tile_filename

            if tile_path.exists():
                print(f"  Already downloaded: {tile_path.name}")
                downloaded_count += 1
            else:
                print(f"  Downloading...")

                if tile_size is None:
                    img = downloader.download_tile(best_tile, bands=bands, crop_bbox=bbox)
                    if img is not None:
                        img = downloader._normalize_to_uint8(img)
                else:
                    if bands == ['red', 'green', 'blue']:
                        img = downloader.download_rgb_thumbnail(best_tile, output_size=tile_size, crop_bbox=bbox)
                    else:
                        img = downloader.download_tile(best_tile, bands=bands, crop_bbox=bbox)
                        if img is not None:
                            img = downloader._normalize_to_uint8(img)
                            img = cv2.resize(img, tile_size)

                if img is None:
                    print(f"  Failed to download")
                    skipped_count += 1
                else:
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        cv2.imwrite(str(tile_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    else:
                        cv2.imwrite(str(tile_path), img)
                    print(f"  Saved: {tile_path.name}")
                    downloaded_count += 1

        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"Downloaded: {downloaded_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total images: {len(list(output_path.glob('*.jpg')))}")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}")


def process_command(
    tiles_dir: str,
    bbox: List[float],
    output_dir: str,
    enable_embeddings: bool = True,
    display: bool = False,
    save_viz_frames: bool = False,
    create_gif: bool = False
) -> None:
    """
    Process downloaded satellite tiles through DAG pipeline.

    Args:
        tiles_dir: Directory containing downloaded tiles
        bbox: Bounding box for geo-referencing
        output_dir: Where to save processing results
        enable_embeddings: Enable embedding generation for semantic search
        display: Show live visualization window
    """
    tiles_path = Path(tiles_dir)
    tile_files = sorted(tiles_path.glob('*.jpg'))

    if not tile_files:
        print(f"No tiles found in {tiles_dir}")
        return

    print(f"{'='*60}")
    print(f"Processing Satellite Time Series")
    print(f"Tiles directory: {tiles_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Bbox: {bbox}")
    print(f"Embeddings enabled: {enable_embeddings}")
    print(f"Total frames: {len(tile_files)}")
    print(f"{'='*60}\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    viz_frames_dir = None
    if save_viz_frames:
        viz_frames_dir = output_path / "viz_frames"
        viz_frames_dir.mkdir(exist_ok=True)

    db = EventDatabase(session_dir=output_dir)
    writer = EventWriter(session_dir=output_dir, enable_database=True, enable_embeddings=enable_embeddings)

    transforms = [
        RadiometricPreprocessTransform(
            add_indices=True,
            normalize=True
        ),
        BitemporalChangeSegmentationTransform(
            threshold=0.15,
            min_change_area=100,
            use_indices=True
        ),
        EventSerializationTransform(serializer=writer)
    ]

    dag = DAG(transforms)

    print(f"Processing {len(tile_files)} frames through DAG...")
    print("-" * 60)

    if display:
        print("\nPress 'q' to quit, any other key to continue to next frame\n")

    skipped_snow = 0

    for frame_idx, tile_file in enumerate(tile_files):
        print(f"\nFrame {frame_idx}: {tile_file.name}")

        frame = cv2.imread(str(tile_file))
        if frame is None:
            print(f"  Failed to load, skipping")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        is_snow, snow_score = _detect_snow(frame_rgb, threshold=0.6)
        if is_snow:
            print(f"  Snow detected (score={snow_score:.3f}), skipping")
            skipped_snow += 1
            continue

        state = {
            "frame": frame_rgb,
            "frame_count": frame_idx,
            "timestamp": float(frame_idx)
        }

        try:
            result = dag.forward(state)

            events = result.get("events", [])
            progress_events = result.get("progress_events", [])
            all_events = events + progress_events

            if all_events:
                print(f"  Events detected: {len(all_events)}")
                for event in all_events[:5]:
                    print(f"    - {event['type']}: conf={event['confidence']:.2f}")

            change_mag = result.get("change_magnitude", 0)
            if change_mag > 0.01:
                print(f"  Change magnitude: {change_mag:.3f}")

            if display or save_viz_frames:
                change_mask = result.get("change_mask")
                previous_frame = result.get("previous_frame_viz")

                if previous_frame is not None:
                    viz_panels = []

                    before_panel = previous_frame.copy()
                    cv2.putText(before_panel, "Before", (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    viz_panels.append(before_panel)

                    after_panel = frame_rgb.copy()
                    cv2.putText(after_panel, "After", (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    viz_panels.append(after_panel)

                    if change_mask is not None:
                        change_viz = _visualize_change_mask(previous_frame, frame_rgb, change_mask)
                        viz_panels.append(change_viz)

                    combined = _create_panel_display(viz_panels, frame_idx, tile_file.name)

                    if save_viz_frames and viz_frames_dir is not None:
                        date_str = tile_file.stem.split('_')[0]
                        viz_filename = f"{date_str}_{frame_idx:04d}_viz.jpg"
                        viz_path = viz_frames_dir / viz_filename
                        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(viz_path), combined_bgr)

                    if display:
                        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Satellite Construction Monitoring", combined_bgr)

                        key = cv2.waitKey(0)
                        if key == ord('q'):
                            print("\nStopping...")
                            break

        except Exception as e:
            print(f"  Error processing: {e}")
            import traceback
            traceback.print_exc()
            continue

    if display:
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    stats = db.get_event_statistics()
    print(f"\nTotal events detected: {stats['total_events']}")
    print(f"Event types:")
    for event_type, count in stats['by_type'].items():
        print(f"  {event_type}: {count}")

    print(f"\nProcessing summary:")
    print(f"  Total frames: {len(tile_files)}")
    print(f"  Skipped (snow): {skipped_snow}")
    print(f"  Processed: {len(tile_files) - skipped_snow}")

    print(f"\nSession saved to: {output_dir}")
    print(f"Database: {output_dir}/events.db")

    if save_viz_frames and viz_frames_dir is not None:
        viz_frame_files = sorted(viz_frames_dir.glob('*.jpg'))
        print(f"\nVisualization frames saved: {len(viz_frame_files)}")
        print(f"Viz frames directory: {viz_frames_dir}")

        if create_gif and len(viz_frame_files) > 0:
            gif_path = output_path / "satellite_analysis.gif"
            print(f"\nCreating GIF from visualization frames...")

            create_gif_from_tiles(
                tiles_dir=str(viz_frames_dir),
                output_path=str(gif_path),
                duration=2000,
                add_date_labels=False,
                loop=0,
                resize_width=None
            )

    print(f"{'='*60}")


def main():
    """BirdView Satellite CLI"""
    parser = argparse.ArgumentParser(description='BirdView Satellite - Download and process satellite imagery')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    download_parser = subparsers.add_parser('download', help='Download satellite imagery')
    download_parser.add_argument('--aoi', type=str, default='microsoft_fairwater',
                                 help='Area of interest name')
    download_parser.add_argument('--bbox', type=float, nargs=4, default=None,
                                 help='Custom bounding box: lon_min lat_min lon_max lat_max')
    download_parser.add_argument('--output-dir', type=str, default=None,
                                 help='Output directory (default: ./data/satellite/<aoi_name>)')
    download_parser.add_argument('--start-date', type=str, default='2024-01-01',
                                 help='Start date YYYY-MM-DD (default: 2024-01-01)')
    download_parser.add_argument('--end-date', type=str, default='2025-12-31',
                                 help='End date YYYY-MM-DD (default: 2025-12-31)')
    download_parser.add_argument('--max-cloud-cover', type=float, default=20,
                                 help='Maximum cloud cover percentage (default: 20)')
    download_parser.add_argument('--tile-size', type=int, nargs=2, default=None,
                                 help='Tile resolution (width height). Defaults to full native resolution.')
    download_parser.add_argument('--bands', type=str, nargs='+', default=None,
                                 help='Bands to download (e.g., red green blue nir swir16 swir22). Defaults to RGB (red green blue).')

    process_parser = subparsers.add_parser('process', help='Process downloaded tiles through DAG')
    process_parser.add_argument('--aoi', type=str, default='microsoft_fairwater',
                               help='Area of interest name (must match download AOI)')
    process_parser.add_argument('--bbox', type=float, nargs=4, default=None,
                               help='Custom bounding box: lon_min lat_min lon_max lat_max')
    process_parser.add_argument('--tiles-dir', type=str, default=None,
                               help='Directory containing downloaded tiles (default: ./data/satellite/<aoi_name>)')
    process_parser.add_argument('--output-dir', type=str, default=None,
                               help='Output directory (default: ./data/satellite/<aoi_name>_processed)')
    process_parser.add_argument('--no-embeddings', action='store_true', default=False,
                               help='Disable embedding generation')
    process_parser.add_argument('--display', action='store_true', default=False,
                               help='Show live visualization window')
    process_parser.add_argument('--save-viz-frames', action='store_true', default=False,
                               help='Save visualization frames to create GIF later')
    process_parser.add_argument('--create-gif', action='store_true', default=False,
                               help='Automatically create GIF after processing (requires --save-viz-frames)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'download':
        if args.bbox:
            bbox = args.bbox
            aoi_name = "custom"
            print(f"Using custom bbox: {bbox}")
        else:
            aoi = get_aoi(args.aoi)
            bbox = aoi.bbox
            aoi_name = args.aoi
            print(f"Using AOI: {aoi.name}")

        output_dir = args.output_dir if args.output_dir else f"./data/satellite/{aoi_name}"
        tile_size = tuple(args.tile_size) if args.tile_size else None

        download_command(
            bbox=bbox,
            output_dir=output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            max_cloud_cover=args.max_cloud_cover,
            tile_size=tile_size,
            bands=args.bands
        )

    elif args.command == 'process':
        if args.bbox:
            bbox = args.bbox
            aoi_name = "custom"
            print(f"Using custom bbox: {bbox}")
        else:
            aoi = get_aoi(args.aoi)
            bbox = aoi.bbox
            aoi_name = args.aoi
            print(f"Using AOI: {aoi.name}")

        tiles_dir = args.tiles_dir if args.tiles_dir else f"./data/satellite/{aoi_name}"
        output_dir = args.output_dir if args.output_dir else f"./data/satellite/{aoi_name}_processed"

        process_command(
            tiles_dir=tiles_dir,
            bbox=bbox,
            output_dir=output_dir,
            enable_embeddings=not args.no_embeddings,
            display=args.display,
            save_viz_frames=args.save_viz_frames,
            create_gif=args.create_gif
        )


if __name__ == "__main__":
    main()
