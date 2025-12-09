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


def download_and_process_latest(
    bbox: List[float],
    output_dir: str,
    max_cloud_cover: float = 20,
    enable_vlm: bool = True,
    enable_solar_segmentation: bool = False,
    lookback_days: int = 30,
    display: bool = True,
    show_multispectral: bool = True,
    tile_size: tuple = None,
    download_only: bool = False,
    force_download: bool = False
) -> None:
    """
    Download latest imagery for bbox and process through pipeline.

    Args:
        bbox: [lon_min, lat_min, lon_max, lat_max]
        output_dir: Where to save tiles and results
        max_cloud_cover: Maximum acceptable cloud cover %
        enable_vlm: Enable VLM analysis
        lookback_days: How many days to look back for imagery
        display: Show live visualization window
        show_multispectral: Show multispectral bands alongside RGB
        tile_size: (width, height) for tile resolution. Use None for full native resolution.
        download_only: If True, only download data without processing
        force_download: If True, bypass cache and force fresh download
    """
    downloader = Sentinel2Downloader()

    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    print(f"Searching for tiles covering bbox: {bbox}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Max cloud cover: {max_cloud_cover}%")
    print(f"Lookback: {lookback_days} days")
    print("=" * 60)

    tiles = downloader.search(
        bbox=bbox,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        max_cloud_cover=max_cloud_cover,
        limit=20
    )

    if not tiles:
        print(f"No tiles found with <{max_cloud_cover}% cloud cover")
        return

    print(f"\nFound {len(tiles)} tiles")

    latest_tile = tiles[-1]
    print(f"Latest tile: {latest_tile}")

    if show_multispectral and not download_only:
        from bird.satellite.multispectral_viewer import MultispectralViewer

        print("\n" + "=" * 60)
        print("Multispectral View")
        print("=" * 60)

        viewer = MultispectralViewer(output_dir=str(Path(output_dir) / "multispectral"))

        composite = viewer.display_tile_multispectral(
            tile=latest_tile,
            tile_size=tile_size,
            save_output=True
        )

        if composite is not None and display:
            import cv2
            window_name = f"Multispectral - {latest_tile.date}"
            cv2.imshow(window_name, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            print(f"\nPress any key to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

    tiles_dir = Path(output_dir) / "latest_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tile_filename = f"{latest_tile.tile_id}_{latest_tile.date}.jpg"
    tile_path = tiles_dir / tile_filename

    if tile_path.exists() and not force_download:
        print(f"\nUsing cached tile: {tile_path}")
        import cv2
        rgb = cv2.imread(str(tile_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        if force_download and tile_path.exists():
            print(f"\nForce download enabled, bypassing cache...")
            tile_path.unlink()
        print(f"\nDownloading RGB tile to {tiles_dir}...")
        print(f"Cropping to bbox: {bbox}")
        if tile_size is None:
            print("Using full native resolution (10980x10980 for full tile)")
            rgb = downloader.download_tile(latest_tile, bands=['red', 'green', 'blue'], crop_bbox=bbox)
            if rgb is not None:
                rgb = downloader._normalize_to_uint8(rgb)
        else:
            print(f"Using resolution: {tile_size}")
            rgb = downloader.download_rgb_thumbnail(latest_tile, output_size=tile_size, crop_bbox=bbox)

        if rgb is None:
            print("Failed to download tile")
            return

        import cv2
        cv2.imwrite(str(tile_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved: {tile_path}")

    if download_only:
        print("\n" + "=" * 60)
        print("Download complete (processing skipped)")
        print(f"Data saved to: {tiles_dir}")
        return

    print(f"\nProcessing through pipeline...")
    processor = TileBatchProcessor(
        tiles_dir=str(tiles_dir),
        session_dir=output_dir,
        enable_embeddings=True,
        enable_vlm=enable_vlm,
        enable_solar_segmentation=enable_solar_segmentation
    )

    if display:
        processor.process_tiles_with_display(max_tiles=1, save_annotations=True)
    else:
        processor.process_tiles(max_tiles=1, save_annotations=True)

    print("\n" + "=" * 60)
    print("Processing complete")
    print(f"Results saved to: {processor.session_dir}")
    print(f"Annotated tiles: {processor.results_dir}")


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
