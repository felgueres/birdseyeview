import argparse
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

from bird.satellite.sentinel2 import Sentinel2Downloader, Sentinel2Tile
from bird.satellite.aoi import get_aoi
from bird.satellite.process_tiles_batch import TileBatchProcessor

load_dotenv()


def polygon_to_bbox(polygon: List[Tuple[float, float]]) -> List[float]:
    """Convert polygon coordinates to bounding box [lon_min, lat_min, lon_max, lat_max]."""
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    return [min(lons), min(lats), max(lons), max(lats)]


def get_tile_grids_for_bbox(
    downloader: Sentinel2Downloader,
    bbox: List[float],
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31"
) -> List[str]:
    """
    Identify which Sentinel-2 tile grids cover this bbox.

    Returns list of tile grid IDs like ['10SFG', '10SEG'].
    """
    tiles = downloader.search(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=100,
        limit=50
    )

    tile_grids = set()
    for tile in tiles:
        grid_id = tile.tile_id.split('_')[1] if '_' in tile.tile_id else tile.tile_id
        tile_grids.add(grid_id)

    return sorted(list(tile_grids))


def download_and_process_latest(
    bbox: List[float],
    output_dir: str,
    max_cloud_cover: float = 20,
    enable_vlm: bool = True,
    enable_solar_segmentation: bool = False,
    lookback_days: int = 30,
    display: bool = True,
    show_multispectral: bool = True,
    tile_size: tuple = None
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

    if show_multispectral:
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

    if tile_path.exists():
        print(f"\nUsing cached tile: {tile_path}")
        import cv2
        rgb = cv2.imread(str(tile_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        print(f"\nDownloading RGB tile to {tiles_dir}...")
        if tile_size is None:
            print("Using full native resolution (10980x10980 for full tile)")
            rgb = downloader.download_tile(latest_tile, bands=['red', 'green', 'blue'])
            if rgb is not None:
                rgb = downloader._normalize_to_uint8(rgb)
        else:
            print(f"Using resolution: {tile_size}")
            rgb = downloader.download_rgb_thumbnail(latest_tile, output_size=tile_size)

        if rgb is None:
            print("Failed to download tile")
            return

        import cv2
        cv2.imwrite(str(tile_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved: {tile_path}")

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
    """Satellite Monitoring CLI - Track changes in satellite imagery"""
    parser = argparse.ArgumentParser(description='BirdView Satellite - Monitor changes from space')

    parser.add_argument('--aoi', type=str, default='bay_area',
                        help='Area of interest name (default: bay_area)')
    parser.add_argument('--bbox', type=float, nargs=4, default=None,
                        help='Custom bounding box: lon_min lat_min lon_max lat_max')
    parser.add_argument('--output-dir', type=str, default='./data/satellite',
                        help='Output directory (default: ./data/satellite)')
    parser.add_argument('--max-cloud-cover', type=float, default=20,
                        help='Maximum cloud cover percentage (default: 20)')
    parser.add_argument('--enable-vlm', action='store_true', default=False,
    help='Enable VLM analysis (requires OpenAI_API_KEY in .env)')
    parser.add_argument('--lookback-days', type=int, default=30,
                        help='Days to look back for imagery (default: 30)')
    parser.add_argument('--no-display', action='store_true', default=False,
                        help='Disable live visualization window')
    parser.add_argument('--no-multispectral', action='store_true', default=False,
                        help='Disable multispectral band display')
    parser.add_argument('--enable-solar', action='store_true', default=False,
                        help='Enable solar panel segmentation')
    parser.add_argument('--tile-size', type=int, nargs=2, default=None,
                        help='Tile resolution (width height). Use full native resolution if not specified. For SAM3 segmentation, use 2048 2048 or higher.')

    args = parser.parse_args()

    if args.bbox:
        bbox = args.bbox
        print(f"Using custom bbox: {bbox}")
    else:
        aoi = get_aoi(args.aoi)
        bbox = aoi.bbox
        print(f"Using AOI: {aoi.name}")
        print(f"Bbox: {bbox}")

    tile_size = tuple(args.tile_size) if args.tile_size else None

    print(f"\n{'='*60}")
    print(f"Satellite Monitoring")
    print(f"Bbox: {bbox}")
    print(f"Max cloud cover: {args.max_cloud_cover}%")
    print(f"VLM enabled: {args.enable_vlm}")
    print(f"Multispectral enabled: {not args.no_multispectral}")
    print(f"Tile size: {tile_size if tile_size else 'Full native resolution'}")
    print(f"{'='*60}\n")

    download_and_process_latest(
        bbox=bbox,
        output_dir=args.output_dir,
        max_cloud_cover=args.max_cloud_cover,
        enable_vlm=args.enable_vlm,
        enable_solar_segmentation=args.enable_solar,
        lookback_days=args.lookback_days,
        display=not args.no_display,
        show_multispectral=not args.no_multispectral,
        tile_size=tile_size
    )


if __name__ == "__main__":
    main()
