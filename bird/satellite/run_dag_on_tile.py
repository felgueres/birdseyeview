"""
Run your existing video DAG pipeline on satellite tile time series.

This shows how to use satellite imagery as input to your
existing transforms (segmentation, VLMs, tracking, etc).
"""

from bird.satellite.tile_timeseries import TileTimeSeriesProcessor
from bird.core.dag import DAG
from bird.satellite.transforms import (
    SatelliteChangeDetectionTransform,
    LandCoverSegmentationTransform,
    SatelliteEventExtractionTransform,
    GeoReferenceTransform
)
from bird.core.transforms import (
    TemporalSegmentationTransform,
    EventSerializationTransform
)
from bird.events.database import EventDatabase
from bird.events.embedder import EventEmbedder
from bird.events.writer import EventWriter


def create_satellite_dag(
    aoi_bbox,
    session_dir,
    enable_embeddings=True
):
    """
    Create DAG for satellite time series processing.

    Uses your existing DAG architecture + satellite-specific transforms.
    """
    db = EventDatabase(session_dir=session_dir)
    embedder = EventEmbedder() if enable_embeddings else None
    writer = EventWriter(db=db, embedder=embedder)

    transforms = [
        SatelliteChangeDetectionTransform(
            threshold=0.15,
            min_change_area=100
        ),
        LandCoverSegmentationTransform(),
        SatelliteEventExtractionTransform(),
        GeoReferenceTransform(bbox=aoi_bbox),
        EventSerializationTransform(serializer=writer)
    ]

    return DAG(transforms=transforms), db


def process_tile_timeseries(
    tile_grid_id: str = "10SFG",
    start_date: str = "2024-06-01",
    end_date: str = "2024-12-01",
    max_cloud_cover: float = 15,
    max_frames: int = 10
):
    """
    Process a tile's time series through the DAG.

    Args:
        tile_grid_id: Sentinel-2 tile (e.g., '10SFG')
        start_date: Start date
        end_date: End date
        max_cloud_cover: Max cloud %
        max_frames: Limit frames to process
    """
    print(f"Processing tile {tile_grid_id}")
    print("=" * 60)

    # 1. Setup processor
    processor = TileTimeSeriesProcessor(
        tile_grid_id=tile_grid_id,
        output_dir=f"./satellite_sessions/{tile_grid_id}"
    )

    # 2. Search and download
    timeseries = processor.search_tile_history(
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover
    )

    frames = processor.download_timeseries(max_frames=max_frames)

    if not frames:
        print("No frames downloaded")
        return

    # 3. Create DAG
    dag, db = create_satellite_dag(
        aoi_bbox=timeseries.bbox,
        session_dir=f"./satellite_sessions/{tile_grid_id}",
        enable_embeddings=True
    )

    # 4. Process frames through DAG
    print(f"\nProcessing {len(frames)} frames through DAG...")
    print("-" * 60)

    for frame_idx, timestamp, frame, meta in processor.get_frame_iterator():
        print(f"\nFrame {frame_idx}: {meta['date']} (cloud={meta['cloud_cover']:.1f}%)")

        state = {
            "frame": frame,
            "frame_count": frame_idx,
            "timestamp": timestamp
        }

        result = dag.forward(state)

        # Show detected events
        events = result.get("events", [])
        if events:
            print(f"  Events detected: {len(events)}")
            for event in events:
                print(f"    - {event['type']}: conf={event['confidence']:.2f}")

        # Show change magnitude
        change_mag = result.get("change_magnitude", 0)
        if change_mag > 0.01:
            print(f"  Change magnitude: {change_mag:.3f}")

    # 5. Query results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    stats = db.get_event_statistics()
    print(f"\nTotal events detected: {stats['total_events']}")
    print(f"Event types:")
    for event_type, count in stats['by_type'].items():
        print(f"  {event_type}: {count}")

    # Show all events
    print("\nAll events:")
    all_events = db.get_events(limit=50)
    for event in all_events:
        print(f"  {event['date'] if 'date' in event else event['frame']}: {event['type']} (conf={event['confidence']:.2f})")
        if 'meta' in event and 'change_magnitude' in event['meta']:
            print(f"    Change: {event['meta']['change_magnitude']:.3f}")

    # 6. Create video
    processor.frames_to_video(fps=2)

    print(f"\nSession saved to: ./satellite_sessions/{tile_grid_id}")
    print(f"Database: ./satellite_sessions/{tile_grid_id}/events.db")


def main():
    """
    Example: Process SF Bay tile over 6 months.

    This treats the satellite time series like a low-FPS video
    and runs your DAG pipeline on it.
    """
    process_tile_timeseries(
        tile_grid_id="10SFG",
        start_date="2024-06-01",
        end_date="2024-12-01",
        max_cloud_cover=15,
        max_frames=10
    )


if __name__ == "__main__":
    main()
