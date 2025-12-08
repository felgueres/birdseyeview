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
    "microsoft_fairwater": AOI(
        name="Microsoft Fairwater AI Datacenter",
        bbox=[-87.92, 42.66, -87.88, 42.69],
        description="Microsoft hyperscale AI supercomputer campus (former Foxconn site), Mount Pleasant, WI. 315 acres, hundreds of thousands of Nvidia GB200/GB300 GPUs. Under construction, online early 2026.",
        priority_event_types=["construction", "infrastructure", "cooling_systems", "substation"],
    ),
    "meta_hyperion": AOI(
        name="Meta Hyperion AI Campus",
        bbox=[-91.80, 32.44, -91.70, 32.52],
        description="Meta's largest datacenter (Hyperion), Franklin Farm megasite near Rayville, LA. 2,250 acres / 4M sq-ft for LLaMA/AI training, >2 GW compute. Multi-year construction through late 2020s.",
        priority_event_types=["construction", "earthworks", "infrastructure", "utility_corridors"],
    ),
    "rowan_cinco": AOI(
        name="Rowan Digital Cinco Campus",
        bbox=[-98.92, 29.24, -98.85, 29.30],
        description="Rowan Digital 300 MW hyperscale AI campus, Medina County near Lytle, TX. 440 acres for top-5 tech company. Ground broken Aug 2025, initial 60 MW Q4 2025/2026, full build ~2027.",
        priority_event_types=["construction", "grading", "infrastructure", "substation"],
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
