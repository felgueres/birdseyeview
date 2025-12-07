"""
Area of Interest (AOI) definitions for satellite monitoring.

Defines geographic regions for queryable Earth system.
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
    priority_event_types: List[str]

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
    "bay_area": AOI(
        name="San Francisco Bay Area",
        bbox=[-122.5, 37.2, -121.7, 37.9],
        description="Dense urban, tech infrastructure, ports, diverse land use",
        priority_event_types=["vegetation_change"]
    ),
    # 20.456915577634764, -100.07492154345282
    "san_gil": AOI(
        name="Mexico City",
        bbox=[20.456915577634764, -100.07492154345282, 20.466915577634764, -100.06492154345282],
        description="Urban, industrial, commercial, diverse land use",
        priority_event_types=["vegetation_change"]
    ),
    "topaz_solar": AOI(
        name="Topaz Solar Farm - California",
        bbox=[-120.10, 35.35, -120.02, 35.41],
        description="Large-scale solar installation, 550MW capacity, 9M panels",
        priority_event_types=["solar_panel_detection", "infrastructure_change"]
    ),
    "ivanpah_solar": AOI(
        name="Ivanpah Solar Power Facility",
        bbox=[-115.48, 35.53, -115.45, 35.56],
        description="Concentrated solar thermal plant in Mojave Desert",
        priority_event_types=["solar_panel_detection", "infrastructure_change"]
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
        print(f"  Priority events: {', '.join(aoi.priority_event_types)}")
        print()
