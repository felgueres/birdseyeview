import argparse
from pathlib import Path
from typing import List
from datetime import datetime, timedelta
from dotenv import load_dotenv
import cv2

from bird.satellite.sentinel2 import Sentinel2Downloader
from bird.satellite.aoi import get_aoi
from bird.satellite.process_tiles_batch import TileBatchProcessor

load_dotenv()

def download_command(
    bbox: List[float],
    output_dir: str,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    max_cloud_cover: float = 20,
    tile_size: tuple = None
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
    """
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
                    rgb = downloader.download_tile(best_tile, bands=['red', 'green', 'blue'], crop_bbox=bbox)
                    if rgb is not None:
                        rgb = downloader._normalize_to_uint8(rgb)
                else:
                    rgb = downloader.download_rgb_thumbnail(best_tile, output_size=tile_size, crop_bbox=bbox)

                if rgb is None:
                    print(f"  Failed to download")
                    skipped_count += 1
                else:
                    cv2.imwrite(str(tile_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
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
    output_dir: str,
    enable_vlm: bool = False,
    enable_solar: bool = False,
    display: bool = False
) -> None:
    """
    Process downloaded satellite tiles.

    Args:
        tiles_dir: Directory containing downloaded tiles
        output_dir: Where to save processing results
        enable_vlm: Enable VLM analysis
        enable_solar: Enable solar panel segmentation
        display: Show live visualization window
    """
    tiles_path = Path(tiles_dir)
    if not tiles_path.exists() or not list(tiles_path.glob('*.jpg')):
        print(f"No tiles found in {tiles_dir}")
        return

    print(f"{'='*60}")
    print(f"Processing Satellite Tiles")
    print(f"Tiles directory: {tiles_dir}")
    print(f"Output directory: {output_dir}")
    print(f"VLM enabled: {enable_vlm}")
    print(f"Solar segmentation: {enable_solar}")
    print(f"{'='*60}\n")

    processor = TileBatchProcessor(
        tiles_dir=str(tiles_dir),
        session_dir=output_dir,
        enable_embeddings=True,
        enable_vlm=enable_vlm,
        enable_solar_segmentation=enable_solar
    )

    if display:
        processor.process_tiles_with_display(save_annotations=True)
    else:
        processor.process_tiles(save_annotations=True)

    print(f"\n{'='*60}")
    print(f"Processing complete")
    print(f"Results saved to: {processor.session_dir}")
    print(f"Annotated tiles: {processor.results_dir}")
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
    download_parser.add_argument('--output-dir', type=str, default='./data/satellite/tiles',
                                 help='Output directory (default: ./data/satellite/tiles)')
    download_parser.add_argument('--start-date', type=str, default='2024-01-01',
                                 help='Start date YYYY-MM-DD (default: 2024-01-01)')
    download_parser.add_argument('--end-date', type=str, default='2025-12-31',
                                 help='End date YYYY-MM-DD (default: 2025-12-31)')
    download_parser.add_argument('--max-cloud-cover', type=float, default=20,
                                 help='Maximum cloud cover percentage (default: 20)')
    download_parser.add_argument('--tile-size', type=int, nargs=2, default=None,
                                 help='Tile resolution (width height). Defaults to full native resolution.')

    process_parser = subparsers.add_parser('process', help='Process downloaded tiles')
    process_parser.add_argument('--tiles-dir', type=str, required=True,
                               help='Directory containing downloaded tiles')
    process_parser.add_argument('--output-dir', type=str, default='./data/satellite/processed',
                               help='Output directory (default: ./data/satellite/processed)')
    process_parser.add_argument('--enable-vlm', action='store_true', default=False,
                               help='Enable VLM analysis (requires OpenAI_API_KEY in .env)')
    process_parser.add_argument('--enable-solar', action='store_true', default=False,
                               help='Enable solar panel segmentation')
    process_parser.add_argument('--display', action='store_true', default=False,
                               help='Show live visualization window')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'download':
        if args.bbox:
            bbox = args.bbox
            print(f"Using custom bbox: {bbox}")
        else:
            aoi = get_aoi(args.aoi)
            bbox = aoi.bbox
            print(f"Using AOI: {aoi.name}")

        tile_size = tuple(args.tile_size) if args.tile_size else None

        download_command(
            bbox=bbox,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            max_cloud_cover=args.max_cloud_cover,
            tile_size=tile_size
        )

    elif args.command == 'process':
        process_command(
            tiles_dir=args.tiles_dir,
            output_dir=args.output_dir,
            enable_vlm=args.enable_vlm,
            enable_solar=args.enable_solar,
            display=args.display
        )


if __name__ == "__main__":
    main()
