import argparse
from pathlib import Path
from typing import List
from datetime import datetime, timedelta
from dotenv import load_dotenv
import cv2

from bird.satellite.sentinel2 import Sentinel2Downloader
from bird.satellite.aoi import get_aoi
from bird.satellite.transforms import SatelliteChangeDetectionTransform
from bird.core.transforms import EventSerializationTransform
from bird.events.database import EventDatabase
from bird.events.embedder import EventEmbedder
from bird.events.writer import EventWriter
from bird.core.dag import DAG

load_dotenv()

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
    display: bool = False
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

    db = EventDatabase(session_dir=output_dir)
    writer = EventWriter(session_dir=output_dir, enable_database=True, enable_embeddings=enable_embeddings)

    transforms = [
        SatelliteChangeDetectionTransform(
            threshold=0.15,
            min_change_area=100
        ),
        EventSerializationTransform(serializer=writer)
    ]

    dag = DAG(transforms)

    print(f"Processing {len(tile_files)} frames through DAG...")
    print("-" * 60)

    if display:
        print("\nPress 'q' to quit, any other key to continue to next frame\n")

    for frame_idx, tile_file in enumerate(tile_files):
        print(f"\nFrame {frame_idx}: {tile_file.name}")

        frame = cv2.imread(str(tile_file))
        if frame is None:
            print(f"  Failed to load, skipping")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        state = {
            "frame": frame_rgb,
            "frame_count": frame_idx,
            "timestamp": float(frame_idx)
        }

        try:
            result = dag.forward(state)

            events = result.get("events", [])
            if events:
                print(f"  Events detected: {len(events)}")
                for event in events[:3]:
                    print(f"    - {event['type']}: conf={event['confidence']:.2f}")

            change_mag = result.get("change_magnitude", 0)
            if change_mag > 0.01:
                print(f"  Change magnitude: {change_mag:.3f}")

            if display:
                display_frame = result.get("frame", frame_rgb)
                cv2.imshow("Satellite Time Series", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

                key = cv2.waitKey(0)
                if key == ord('q'):
                    print("\nStopping...")
                    break

        except Exception as e:
            print(f"  Error processing: {e}")
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

    print(f"\nSession saved to: {output_dir}")
    print(f"Database: {output_dir}/events.db")
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
            display=args.display
        )


if __name__ == "__main__":
    main()
