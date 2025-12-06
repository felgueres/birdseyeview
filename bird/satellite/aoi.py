"""
Area of Interest (AOI) definitions for satellite monitoring.

Defines geographic regions for queryable Earth system.
"""

from typing import List, Dict
from dataclasses import dataclass


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


import numpy as np


CALIFORNIA_AOIS = {
    "sf_bay": AOI(
        name="San Francisco Bay Area",
        bbox=[-122.5, 37.2, -121.7, 37.9],
        description="Dense urban, tech infrastructure, ports, diverse land use",
        priority_event_types=["construction_detected", "land_cover_change"]
    ),
    "central_valley": AOI(
        name="Central Valley Agriculture",
        bbox=[-121.5, 36.5, -120.0, 37.5],
        description="Major agricultural region, irrigation, crop cycles",
        priority_event_types=["vegetation_change", "water_level_change"]
    ),
    "la_metro": AOI(
        name="Los Angeles Metro",
        bbox=[-118.5, 33.7, -117.7, 34.3],
        description="Urban sprawl, port logistics, infrastructure",
        priority_event_types=["construction_detected", "land_cover_change"]
    ),
    "sierra_nevada": AOI(
        name="Sierra Nevada",
        bbox=[-120.5, 37.5, -119.0, 38.5],
        description="Wildfire monitoring, snow cover, forest health",
        priority_event_types=["vegetation_change", "fire_detected"]
    ),
    "imperial_valley": AOI(
        name="Imperial Valley Solar",
        bbox=[-115.7, 32.7, -114.6, 33.2],
        description="Solar farm development, desert agriculture",
        priority_event_types=["construction_detected", "solar_farm_detected"]
    )
}


def get_aoi(name: str) -> AOI:
    """Get AOI by name."""
    if name not in CALIFORNIA_AOIS:
        raise ValueError(f"Unknown AOI: {name}. Available: {list(CALIFORNIA_AOIS.keys())}")
    return CALIFORNIA_AOIS[name]


def list_aois() -> Dict[str, AOI]:
    """List all available AOIs."""
    return CALIFORNIA_AOIS


if __name__ == "__main__":
    print("Available AOIs for California:\n")
    for name, aoi in CALIFORNIA_AOIS.items():
        print(f"{name}:")
        print(f"  {aoi.description}")
        print(f"  Bbox: {aoi.bbox}")
        print(f"  Area: {aoi.area_km2():.1f} km²")
        print(f"  Priority events: {', '.join(aoi.priority_event_types)}")
        print()
