"""
Multispectral band viewer for Sentinel-2 imagery.

Display multispectral bands (NIR, SWIR, etc.) side by side with RGB
for a given date.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from bird.satellite.sentinel2 import Sentinel2Downloader, Sentinel2Tile


@dataclass
class BandConfig:
    """Configuration for a single band visualization."""
    name: str
    sentinel_band: str
    colormap: Optional[int] = None
    normalize: bool = True


class MultispectralViewer:
    """
    View multispectral Sentinel-2 bands side by side with RGB.

    Available Sentinel-2 bands:
    - RGB: red (B04), green (B03), blue (B02)
    - NIR: nir (B08) - vegetation reflectance
    - SWIR: swir16 (B11), swir22 (B12) - moisture, geology
    - Red Edge: rededge1-3 (B05-B07) - vegetation health
    - Coastal/Aerosol: coastal (B01)
    """

    DEFAULT_BANDS = [
        BandConfig("RGB", "rgb", None, False),
        BandConfig("NIR", "nir", cv2.COLORMAP_VIRIDIS, True),
        BandConfig("SWIR1", "swir16", cv2.COLORMAP_INFERNO, True),
        BandConfig("SWIR2", "swir22", cv2.COLORMAP_PLASMA, True),
    ]

    def __init__(
        self,
        downloader: Optional[Sentinel2Downloader] = None,
        output_dir: str = "./data/multispectral"
    ):
        self.downloader = downloader or Sentinel2Downloader()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def display_tile_multispectral(
        self,
        tile: Sentinel2Tile,
        bands: Optional[List[BandConfig]] = None,
        tile_size: Tuple[int, int] = (512, 512),
        save_output: bool = True
    ) -> Optional[np.ndarray]:
        """
        Display multispectral bands for a single tile.

        Args:
            tile: Sentinel2Tile to download and display
            bands: List of band configurations (default: RGB, NIR, SWIR1, SWIR2)
            tile_size: Size for each individual band display
            save_output: Save the composite visualization

        Returns:
            Composite visualization array or None
        """
        bands = bands or self.DEFAULT_BANDS

        print(f"\nProcessing tile: {tile}")
        print(f"Date: {tile.date}")
        print(f"Cloud cover: {tile.cloud_cover:.1f}%")
        print("=" * 60)

        band_images = []
        band_labels = []

        for band_config in bands:
            print(f"Downloading {band_config.name}...", end=" ")

            if band_config.sentinel_band == "rgb":
                band_data = self._download_rgb(tile, tile_size)
            else:
                band_data = self._download_band(
                    tile,
                    band_config.sentinel_band,
                    tile_size,
                    band_config.colormap,
                    band_config.normalize
                )

            if band_data is not None:
                band_images.append(band_data)
                band_labels.append(band_config.name)
                print("OK")
            else:
                print("FAILED")

        if not band_images:
            print("No bands downloaded successfully")
            return None

        composite = self._create_composite_view(
            band_images,
            band_labels,
            tile.date,
            tile.cloud_cover
        )

        if save_output:
            output_path = self.output_dir / f"{tile.tile_id}_{tile.date}_multispectral.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            print(f"\nSaved: {output_path}")

        return composite

    def display_date_multispectral(
        self,
        date: str,
        bbox: List[float],
        bands: Optional[List[BandConfig]] = None,
        max_cloud_cover: float = 20,
        tile_size: Tuple[int, int] = (512, 512),
        show_window: bool = True
    ) -> Optional[np.ndarray]:
        """
        Display multispectral bands for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
            bands: List of band configurations
            max_cloud_cover: Maximum cloud cover percentage
            tile_size: Size for each band display
            show_window: Display in OpenCV window

        Returns:
            Composite visualization or None
        """
        print(f"Searching for imagery on {date}...")
        print(f"Bbox: {bbox}")
        print(f"Max cloud cover: {max_cloud_cover}%\n")

        tiles = self.downloader.search(
            bbox=bbox,
            start_date=date,
            end_date=date,
            max_cloud_cover=max_cloud_cover,
            limit=5
        )

        if not tiles:
            print(f"No tiles found for {date} with <{max_cloud_cover}% cloud cover")
            return None

        print(f"Found {len(tiles)} tile(s) for {date}")
        tile = tiles[0]

        composite = self.display_tile_multispectral(
            tile,
            bands=bands,
            tile_size=tile_size,
            save_output=True
        )

        if composite is not None and show_window:
            self._show_interactive_window(composite, tile)

        return composite

    def display_date_range_multispectral(
        self,
        start_date: str,
        end_date: str,
        bbox: List[float],
        bands: Optional[List[BandConfig]] = None,
        max_cloud_cover: float = 20,
        tile_size: Tuple[int, int] = (384, 384),
        max_tiles: int = 10
    ):
        """
        Display multispectral bands for all tiles in a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            bbox: Bounding box
            bands: List of band configurations
            max_cloud_cover: Maximum cloud cover %
            tile_size: Size for each band display
            max_tiles: Maximum number of tiles to display
        """
        print(f"Searching for imagery: {start_date} to {end_date}")
        print(f"Bbox: {bbox}")
        print(f"Max cloud cover: {max_cloud_cover}%\n")

        tiles = self.downloader.search(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            limit=max_tiles
        )

        if not tiles:
            print("No tiles found")
            return

        print(f"Found {len(tiles)} tile(s)\n")

        for i, tile in enumerate(tiles[:max_tiles]):
            print(f"\n{'='*60}")
            print(f"Tile {i+1}/{len(tiles[:max_tiles])}")

            composite = self.display_tile_multispectral(
                tile,
                bands=bands,
                tile_size=tile_size,
                save_output=True
            )

            if composite is not None:
                self._show_interactive_window(composite, tile)

    def _download_rgb(
        self,
        tile: Sentinel2Tile,
        size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Download RGB composite."""
        rgb = self.downloader.download_rgb_thumbnail(tile, output_size=size)
        return rgb

    def _download_band(
        self,
        tile: Sentinel2Tile,
        band_name: str,
        size: Tuple[int, int],
        colormap: Optional[int] = None,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """Download and process a single band."""
        band_data = self.downloader.download_tile(
            tile,
            bands=[band_name]
        )

        if band_data is None:
            return None

        band_data = band_data.squeeze()

        if normalize:
            band_data = self._normalize_band(band_data)
        else:
            band_data = np.clip(band_data / band_data.max() * 255, 0, 255).astype(np.uint8)

        band_data = cv2.resize(band_data, size)

        if colormap is not None:
            band_data = cv2.applyColorMap(band_data, colormap)
            band_data = cv2.cvtColor(band_data, cv2.COLOR_BGR2RGB)
        else:
            band_data = cv2.cvtColor(band_data, cv2.COLOR_GRAY2RGB)

        return band_data

    def _normalize_band(self, data: np.ndarray) -> np.ndarray:
        """Normalize band data with percentile clipping."""
        data = data.astype(np.float32)

        p2, p98 = np.percentile(data, (2, 98))
        data = np.clip(data, p2, p98)

        data = (data - p2) / (p98 - p2 + 1e-8)
        data = (data * 255).astype(np.uint8)

        return data

    def _create_composite_view(
        self,
        band_images: List[np.ndarray],
        band_labels: List[str],
        date: str,
        cloud_cover: float
    ) -> np.ndarray:
        """Create composite visualization with all bands side by side."""
        n_bands = len(band_images)

        if n_bands == 0:
            return np.zeros((512, 512, 3), dtype=np.uint8)

        h, w = band_images[0].shape[:2]

        cols = min(n_bands, 4)
        rows = (n_bands + cols - 1) // cols

        padding = 10
        label_height = 40

        composite_w = cols * w + (cols + 1) * padding
        composite_h = rows * (h + label_height) + (rows + 1) * padding + 60

        composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
        composite[:] = 30

        title = f"Multispectral View - {date} (Cloud: {cloud_cover:.1f}%)"
        cv2.putText(
            composite, title,
            (padding, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

        y_offset = 60 + padding

        for i, (img, label) in enumerate(zip(band_images, band_labels)):
            row = i // cols
            col = i % cols

            x = padding + col * (w + padding)
            y = y_offset + row * (h + label_height + padding)

            composite[y:y+h, x:x+w] = img

            cv2.putText(
                composite, label,
                (x + 5, y + h + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA
            )

        return composite

    def _show_interactive_window(
        self,
        composite: np.ndarray,
        tile: Sentinel2Tile
    ):
        """Display composite in interactive window."""
        window_name = f"Multispectral View - {tile.date}"
        cv2.imshow(window_name, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        print(f"\nDisplaying {tile.date}")
        print("Press any key to continue, 'q' to quit")

        key = cv2.waitKey(0)
        cv2.destroyWindow(window_name)

        if key == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("User requested quit")


def create_false_color_composite(
    viewer: MultispectralViewer,
    tile: Sentinel2Tile,
    r_band: str = "nir",
    g_band: str = "red",
    b_band: str = "green",
    size: Tuple[int, int] = (512, 512)
) -> Optional[np.ndarray]:
    """
    Create false color composite (e.g., NIR-R-G for vegetation).

    Common false color combinations:
    - Agriculture: NIR-Red-Green (healthy vegetation = bright red)
    - Moisture: SWIR2-NIR-Red (water = dark blue)
    - Urban: SWIR2-SWIR1-Red (urban = cyan/white)

    Args:
        viewer: MultispectralViewer instance
        tile: Sentinel2Tile to process
        r_band: Band for red channel
        g_band: Band for green channel
        b_band: Band for blue channel
        size: Output size

    Returns:
        False color composite as RGB array
    """
    print(f"Creating false color composite: {r_band}-{g_band}-{b_band}")

    bands_data = []
    for band_name in [r_band, g_band, b_band]:
        data = viewer.downloader.download_tile(tile, bands=[band_name])
        if data is None:
            print(f"Failed to download {band_name}")
            return None
        bands_data.append(data.squeeze())

    composite = np.stack(bands_data, axis=-1)

    composite = viewer._normalize_band(composite.astype(np.float32))

    composite = cv2.resize(composite, size)

    return composite


def main():
    """Example usage of multispectral viewer."""
    import argparse
    from datetime import datetime, timedelta
    from bird.satellite.aoi import get_aoi

    parser = argparse.ArgumentParser(
        description='View Sentinel-2 multispectral bands side by side with RGB'
    )
    parser.add_argument(
        '--date', type=str, default=None,
        help='Date to view (YYYY-MM-DD), defaults to latest available'
    )
    parser.add_argument(
        '--aoi', type=str, default='bay_area',
        help='Area of interest (default: bay_area)'
    )
    parser.add_argument(
        '--bbox', type=float, nargs=4, default=None,
        help='Custom bbox: lon_min lat_min lon_max lat_max'
    )
    parser.add_argument(
        '--max-cloud-cover', type=float, default=20,
        help='Max cloud cover percentage (default: 20)'
    )
    parser.add_argument(
        '--lookback-days', type=int, default=30,
        help='Days to look back for imagery (default: 30)'
    )
    parser.add_argument(
        '--tile-size', type=int, default=512,
        help='Size of each band tile (default: 512)'
    )
    parser.add_argument(
        '--bands', type=str, nargs='+',
        choices=['rgb', 'nir', 'swir16', 'swir22', 'rededge1', 'rededge2', 'rededge3', 'coastal'],
        default=['rgb', 'nir', 'swir16', 'swir22'],
        help='Bands to display (default: rgb nir swir16 swir22)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./data/multispectral',
        help='Output directory (default: ./data/multispectral)'
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Do not show interactive window'
    )

    args = parser.parse_args()

    if args.bbox:
        bbox = args.bbox
    else:
        aoi = get_aoi(args.aoi)
        bbox = aoi.bbox

    if args.date is None:
        from bird.satellite.sentinel2 import Sentinel2Downloader

        downloader = Sentinel2Downloader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.lookback_days)

        print(f"Searching for latest imagery...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Bbox: {bbox}")
        print(f"Max cloud cover: {args.max_cloud_cover}%")
        print("=" * 60)

        tiles = downloader.search(
            bbox=bbox,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            max_cloud_cover=args.max_cloud_cover,
            limit=20
        )

        if not tiles:
            print(f"\nNo tiles found with <{args.max_cloud_cover}% cloud cover")
            print(f"Try increasing --max-cloud-cover or --lookback-days")
            return

        latest_tile = tiles[-1]
        date = latest_tile.date
        print(f"\nLatest tile found: {latest_tile.date}")
    else:
        date = args.date

    band_configs = []
    colormap_map = {
        'nir': cv2.COLORMAP_VIRIDIS,
        'swir16': cv2.COLORMAP_INFERNO,
        'swir22': cv2.COLORMAP_PLASMA,
        'rededge1': cv2.COLORMAP_VIRIDIS,
        'rededge2': cv2.COLORMAP_VIRIDIS,
        'rededge3': cv2.COLORMAP_VIRIDIS,
        'coastal': cv2.COLORMAP_OCEAN,
    }

    for band in args.bands:
        if band == 'rgb':
            band_configs.append(BandConfig("RGB", "rgb", None, False))
        else:
            band_configs.append(
                BandConfig(
                    band.upper(),
                    band,
                    colormap_map.get(band, cv2.COLORMAP_VIRIDIS),
                    True
                )
            )

    viewer = MultispectralViewer(output_dir=args.output_dir)

    tile_size = (args.tile_size, args.tile_size)

    viewer.display_date_multispectral(
        date=date,
        bbox=bbox,
        bands=band_configs,
        max_cloud_cover=args.max_cloud_cover,
        tile_size=tile_size,
        show_window=not args.no_display
    )


if __name__ == "__main__":
    main()
