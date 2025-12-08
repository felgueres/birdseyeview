"""
Sentinel-2 satellite imagery ingestion via AWS Open Data.

Usage:
    from bird.satellite.sentinel2 import Sentinel2Downloader

    downloader = Sentinel2Downloader()
    tiles = downloader.search(
        bbox=[-122.5, 37.2, -121.7, 37.9],  # San Francisco Bay Area
        start_date="2024-01-01",
        end_date="2024-12-01",
        max_cloud_cover=20
    )

    for tile in tiles[:5]:
        rgb_array = downloader.download_tile(tile, bands=['B04', 'B03', 'B02'])
"""

import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class Sentinel2Tile:
    """Metadata for a single Sentinel-2 tile."""
    tile_id: str
    date: str
    cloud_cover: float
    bbox: List[float]
    assets: Dict
    geometry: Dict

    def __str__(self):
        return f"Sentinel2Tile({self.tile_id}, {self.date}, cloud={self.cloud_cover:.1f}%)"


class Sentinel2Downloader:
    """
    Download Sentinel-2 imagery from AWS Open Data.

    Uses the Element 84 Earth Search STAC API.
    """

    STAC_API_URL = "https://earth-search.aws.element84.com/v1"

    def __init__(self):
        self.session = requests.Session()

    def search(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20,
        limit: int = 100
    ) -> List[Sentinel2Tile]:
        """
        Search for Sentinel-2 tiles.

        Args:
            bbox: [lon_min, lat_min, lon_max, lat_max]
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            limit: Maximum number of results

        Returns:
            List of Sentinel2Tile objects
        """
        search_params = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "limit": limit,
            "query": {
                "eo:cloud_cover": {"lt": max_cloud_cover}
            }
        }

        response = self.session.post(
            f"{self.STAC_API_URL}/search",
            json=search_params
        )
        response.raise_for_status()

        data = response.json()
        tiles = []

        for feature in data.get('features', []):
            tile = self._parse_feature(feature)
            if tile:
                tiles.append(tile)

        tiles.sort(key=lambda t: t.date)
        return tiles

    def _parse_feature(self, feature: Dict) -> Optional[Sentinel2Tile]:
        """Parse STAC feature into Sentinel2Tile."""
        try:
            properties = feature.get('properties', {})
            tile_id = feature.get('id', 'unknown')
            date = properties.get('datetime', '').split('T')[0]
            cloud_cover = properties.get('eo:cloud_cover', 100)

            bbox = feature.get('bbox', [])
            geometry = feature.get('geometry', {})
            assets = feature.get('assets', {})

            return Sentinel2Tile(
                tile_id=tile_id,
                date=date,
                cloud_cover=cloud_cover,
                bbox=bbox,
                assets=assets,
                geometry=geometry
            )
        except Exception as e:
            print(f"Failed to parse feature: {e}")
            return None

    def download_tile(
        self,
        tile: Sentinel2Tile,
        bands: List[str] = ['red', 'green', 'blue'],
        resolution: int = 10,
        crop_bbox: Optional[List[float]] = None
    ) -> Optional[np.ndarray]:
        """
        Download a Sentinel-2 tile.

        Args:
            tile: Sentinel2Tile object
            bands: List of band names (default RGB: red, green, blue)
                   Available: red, green, blue, nir, swir16, swir22, coastal, etc.
            resolution: Spatial resolution in meters (10, 20, or 60)
            crop_bbox: Optional [lon_min, lat_min, lon_max, lat_max] to crop to

        Returns:
            numpy array of shape (H, W, C) or None if download fails
        """
        try:
            import rasterio
        except ImportError:
            print("rasterio not installed. Run: pip install rasterio")
            return None

        print(f"Downloading {tile}...")

        band_arrays = []
        target_shape = None
        window = None
        transform = None

        for band in bands:
            if band not in tile.assets:
                print(f"Band {band} not available in assets")
                continue

            band_url = tile.assets[band].get('href')
            if not band_url:
                print(f"No URL for band {band}")
                continue

            try:
                with rasterio.open(band_url) as src:
                    if crop_bbox is not None and window is None:
                        from rasterio.windows import from_bounds
                        from rasterio.warp import transform_bounds

                        # Transform bbox from WGS84 to tile's CRS
                        bbox_in_tile_crs = transform_bounds(
                            'EPSG:4326',  # WGS84 lat/lon
                            src.crs,      # Tile's CRS (usually UTM)
                            *crop_bbox
                        )
                        print(f"Bbox in WGS84: {crop_bbox}")
                        print(f"Bbox in tile CRS: {bbox_in_tile_crs}")
                        print(f"Tile bounds: {src.bounds}")

                        window = from_bounds(*bbox_in_tile_crs, transform=src.transform)
                        transform = src.transform
                        print(f"Window for crop: {window}")

                    if window is not None:
                        data = src.read(1, window=window)
                    else:
                        data = src.read(1)

                    if target_shape is None:
                        target_shape = data.shape
                    elif data.shape != target_shape:
                        import cv2
                        data = cv2.resize(data, (target_shape[1], target_shape[0]))

                    band_arrays.append(data)
            except Exception as e:
                print(f"Failed to download band {band}: {e}")
                continue

        if not band_arrays:
            return None

        stacked = np.stack(band_arrays, axis=-1)
        if crop_bbox is not None:
            print(f"Cropped to bbox: {crop_bbox}")
            print(f"Cropped image shape: {stacked.shape}")

        if stacked.size == 0:
            print("ERROR: Cropped image is empty. Check if bbox is within tile bounds.")
            return None

        return stacked

    def download_rgb_thumbnail(
        self,
        tile: Sentinel2Tile,
        output_size: Tuple[int, int] = (512, 512),
        crop_bbox: Optional[List[float]] = None
    ) -> Optional[np.ndarray]:
        """
        Download RGB thumbnail (fast preview).

        Args:
            tile: Sentinel2Tile object
            output_size: (width, height) for output image
            crop_bbox: Optional [lon_min, lat_min, lon_max, lat_max] to crop to

        Returns:
            RGB numpy array (H, W, 3) in 0-255 range
        """
        if 'visual' in tile.assets:
            visual_url = tile.assets['visual'].get('href')
            if visual_url:
                return self._download_visual_thumbnail(visual_url, output_size, crop_bbox)

        rgb = self.download_tile(tile, bands=['red', 'green', 'blue'], crop_bbox=crop_bbox)
        if rgb is None:
            return None

        rgb = self._normalize_to_uint8(rgb)

        import cv2
        resized = cv2.resize(rgb, output_size)
        return resized

    def _download_visual_thumbnail(
        self,
        url: str,
        output_size: Tuple[int, int],
        crop_bbox: Optional[List[float]] = None
    ) -> Optional[np.ndarray]:
        """Download pre-rendered visual thumbnail."""
        try:
            import rasterio
            import cv2

            with rasterio.open(url) as src:
                if crop_bbox is not None:
                    from rasterio.windows import from_bounds
                    from rasterio.warp import transform_bounds

                    # Transform bbox from WGS84 to tile's CRS
                    bbox_in_tile_crs = transform_bounds(
                        'EPSG:4326',
                        src.crs,
                        *crop_bbox
                    )
                    window = from_bounds(*bbox_in_tile_crs, transform=src.transform)
                    data = src.read([1, 2, 3], window=window)
                    print(f"Cropped to bbox: {crop_bbox}")
                else:
                    data = src.read([1, 2, 3])
                data = np.transpose(data, (1, 2, 0))

            data = self._normalize_to_uint8(data)
            resized = cv2.resize(data, output_size)
            return resized
        except Exception as e:
            print(f"Failed to download visual: {e}")
            return None

    def _normalize_to_uint8(self, data: np.ndarray) -> np.ndarray:
        """Normalize to 0-255 uint8 range with basic contrast stretch."""
        if data.size == 0:
            print("ERROR: Cannot normalize empty array")
            return data

        data = data.astype(np.float32)

        p2, p98 = np.percentile(data, (2, 98))
        data = np.clip(data, p2, p98)

        data = (data - p2) / (p98 - p2 + 1e-8)
        data = (data * 255).astype(np.uint8)

        return data


if __name__ == "__main__":
    downloader = Sentinel2Downloader()

    # San Francisco Bay Area
    bbox = [-122.5, 37.2, -121.7, 37.9]

    print("Searching for Sentinel-2 tiles...")
    tiles = downloader.search(
        bbox=bbox,
        start_date="2024-01-01",
        end_date="2024-12-01",
        max_cloud_cover=10,
        limit=10
    )

    print(f"\nFound {len(tiles)} tiles:")
    for tile in tiles:
        print(f"  {tile}")

    if tiles:
        print(f"\nDownloading first tile...")
        rgb = downloader.download_rgb_thumbnail(tiles[0])
        if rgb is not None:
            import cv2
            cv2.imwrite("sentinel2_preview.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print("Saved: sentinel2_preview.jpg")
