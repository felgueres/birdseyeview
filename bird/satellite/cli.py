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
    lookback_days: int = 30
) -> None:
    """
    Download latest imagery for bbox and process through pipeline.

    Args:
        bbox: [lon_min, lat_min, lon_max, lat_max]
        output_dir: Where to save tiles and results
        max_cloud_cover: Maximum acceptable cloud cover %
        enable_vlm: Enable VLM analysis
        lookback_days: How many days to look back for imagery
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

    tiles_dir = Path(output_dir) / "latest_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading latest tile to {tiles_dir}...")
    rgb = downloader.download_rgb_thumbnail(latest_tile, output_size=(512, 512))

    if rgb is None:
        print("Failed to download tile")
        return

    import cv2
    tile_filename = f"{latest_tile.tile_id}_{latest_tile.date}.jpg"
    tile_path = tiles_dir / tile_filename
    cv2.imwrite(str(tile_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved: {tile_path}")

    print(f"\nProcessing through pipeline...")
    processor = TileBatchProcessor(
        tiles_dir=str(tiles_dir),
        session_dir=output_dir,
        enable_embeddings=True,
        enable_vlm=enable_vlm
    )

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
                        help='Enable VLM analysis (requires OpenAI API key)')
    parser.add_argument('--lookback-days', type=int, default=30,
                        help='Days to look back for imagery (default: 30)')

    args = parser.parse_args()

    if args.bbox:
        bbox = args.bbox
        print(f"Using custom bbox: {bbox}")
    else:
        aoi = get_aoi(args.aoi)
        bbox = aoi.bbox
        print(f"Using AOI: {aoi.name}")
        print(f"Bbox: {bbox}")

    print(f"\n{'='*60}")
    print(f"Satellite Monitoring")
    print(f"Bbox: {bbox}")
    print(f"Max cloud cover: {args.max_cloud_cover}%")
    print(f"VLM enabled: {args.enable_vlm}")
    print(f"{'='*60}\n")

    download_and_process_latest(
        bbox=bbox,
        output_dir=args.output_dir,
        max_cloud_cover=args.max_cloud_cover,
        enable_vlm=args.enable_vlm,
        lookback_days=args.lookback_days
    )


if __name__ == "__main__":
    main()
