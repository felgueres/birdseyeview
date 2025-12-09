"""
Create a GIF from satellite tile time series.

Usage:
    python -m bird.satellite.create_timeseries_gif --tiles-dir ./data/satellite/microsoft_fairwater --output docs/satellite_timeseries.gif
"""

import argparse
import cv2
from pathlib import Path
from PIL import Image
import numpy as np


def create_gif_from_tiles(
    tiles_dir: str,
    output_path: str,
    duration: int = 500,
    add_date_labels: bool = True,
    loop: int = 0,
    resize_width: int = None
):
    """
    Create an animated GIF from satellite tiles.

    Args:
        tiles_dir: Directory containing tile JPG files
        output_path: Output GIF path
        duration: Duration per frame in milliseconds (default: 500ms)
        add_date_labels: Add date labels to frames
        loop: Number of loops (0 = infinite)
        resize_width: Resize width (maintains aspect ratio, default: None)
    """
    tiles_path = Path(tiles_dir)
    tile_files = sorted(tiles_path.glob('*.jpg'))

    if not tile_files:
        print(f"No tiles found in {tiles_dir}")
        return

    print(f"Found {len(tile_files)} tiles")
    print(f"Creating GIF: {output_path}")

    frames = []

    for tile_file in tile_files:
        print(f"  Processing {tile_file.name}")

        frame = cv2.imread(str(tile_file))
        if frame is None:
            print(f"    Failed to load, skipping")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if add_date_labels:
            date_str = tile_file.stem.split('_')[0]
            cv2.putText(
                frame_rgb,
                date_str,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA
            )
            cv2.putText(
                frame_rgb,
                date_str,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        pil_frame = Image.fromarray(frame_rgb)

        if resize_width:
            aspect_ratio = pil_frame.height / pil_frame.width
            new_height = int(resize_width * aspect_ratio)
            pil_frame = pil_frame.resize((resize_width, new_height), Image.Resampling.LANCZOS)

        frames.append(pil_frame)

    if not frames:
        print("No frames to save")
        return

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )

    print(f"\nGIF saved: {output_file}")
    print(f"Frames: {len(frames)}")
    print(f"Duration per frame: {duration}ms")
    print(f"Total duration: {len(frames) * duration / 1000:.1f}s")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Create animated GIF from satellite tile time series'
    )
    parser.add_argument(
        '--tiles-dir',
        type=str,
        required=True,
        help='Directory containing downloaded tiles'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./docs/satellite_timeseries.gif',
        help='Output GIF path (default: ./docs/satellite_timeseries.gif)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=500,
        help='Duration per frame in milliseconds (default: 500)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Disable date labels on frames'
    )
    parser.add_argument(
        '--resize-width',
        type=int,
        default=None,
        help='Resize width (maintains aspect ratio, default: original size)'
    )
    parser.add_argument(
        '--loop',
        type=int,
        default=0,
        help='Number of loops (0 = infinite, default: 0)'
    )

    args = parser.parse_args()

    create_gif_from_tiles(
        tiles_dir=args.tiles_dir,
        output_path=args.output,
        duration=args.duration,
        add_date_labels=not args.no_labels,
        loop=args.loop,
        resize_width=args.resize_width
    )


if __name__ == "__main__":
    main()
