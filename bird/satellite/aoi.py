"""
Area of Interest (AOI)
Best tool for drawing geometries: https://geojson.io
"""

from typing import List, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class AOI:
    """Area of Interest definition."""
    name: str
    bbox: List[float]  # [lon_min, lat_min, lon_max, lat_max]
    description: str

    def __str__(self):
        return f"{self.name} {self.bbox}"

    def area_km2(self) -> float:
        """Approximate area in km²."""
        lon_min, lat_min, lon_max, lat_max = self.bbox
        lat_avg = (lat_min + lat_max) / 2

        km_per_deg_lon = 111.32 * abs(np.cos(np.radians(lat_avg)))
        km_per_deg_lat = 111.32

        width_km = (lon_max - lon_min) * km_per_deg_lon
        height_km = (lat_max - lat_min) * km_per_deg_lat

        return width_km * height_km


AREAS_OF_INTEREST = {
    "microsoft_fairwater": AOI(
        name="Microsoft Fairwater AI Datacenter",
        bbox=[-87.91457190570361, 42.669424023512704, -87.89488996583513, 42.68356647516654],
        description="",
    ),
    "microsoft_fairwater_2": AOI(
        name="Microsoft Fairwater AI Datacenter 2",
        bbox=[-87.93291695909029, 42.669478598705524, -87.90547365048131, 42.683757574885476],
        description="",
    ),
    "meta_hyperion": AOI(
        name="Meta Hyperion AI Campus",
        bbox=[-91.80, 32.44, -91.70, 32.52],
        description="",
    ),
    "rowan_cinco": AOI(
        name="Rowan Digital Cinco Campus",
        bbox=[-98.92, 29.24, -98.85, 29.30],
        description="",
    ),
}

def get_aoi(name: str) -> AOI:
    """Get AOI by name."""
    if name not in AREAS_OF_INTEREST:
        raise ValueError(f"Unknown AOI: {name}. Available: {list(AREAS_OF_INTEREST.keys())}")
    return AREAS_OF_INTEREST[name]


def list_aois() -> Dict[str, AOI]:
    """List all available AOIs."""
    return AREAS_OF_INTEREST


if __name__ == "__main__":
    print("Available AOIs for California:\n")
    for name, aoi in AREAS_OF_INTEREST.items():
        print(f"{name}:")
        print(f"  {aoi.description}")
        print(f"  Bbox: {aoi.bbox}")
        print(f"  Area: {aoi.area_km2():.1f} km²")
