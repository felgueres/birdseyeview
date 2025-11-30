import argparse
import subprocess
import time
import os
from dotenv import load_dotenv

from bird.config import VisionConfig
from bird.core.camera import Webcam, SonyA5000, VideoFileCamera
from bird.core.pipeline import run

load_dotenv()

def main():
    """BirdView CLI - Computer Vision Pipeline Runner"""
    parser = argparse.ArgumentParser(description='BirdView - Computer Vision Pipeline')
    parser.add_argument('--camera', nargs='?', default='webcam', choices=['webcam', 'sony', 'video'],
                        help='Camera source (default: webcam)')
    parser.add_argument('--camera-index', type=int, default=0,
                        help='Webcam index (default: 0)')
    parser.add_argument('--video-path', type=str, default=None,
                        help='Path to input video when using --camera video')
    parser.add_argument('--vlm', choices=['ollama', 'openai'], default=None,
                        help='VLM provider: ollama (local) or openai (API). Auto-detected from model if not specified.')
    parser.add_argument('--model', default=None,
                        help='VLM model name (default: llava:7b for ollama, gpt-4o for openai)')
    parser.add_argument('--enable-scene-graph', action='store_true',
                        help='Enable scene graph generation')
    parser.add_argument('--enable-tracking', action='store_true', default=False,
                        help='Enable object tracking (default: enabled)')
    parser.add_argument('--enable-box', action='store_true', default=False,
                        help='Enable bounding box detection (default: enabled)')
    parser.add_argument('--no-box', action='store_true', default=False,
                        help='Disable bounding box visualization')
    parser.add_argument('--enable-segmentation', action='store_true', default=False,
                        help='Enable segmentation (default: disabled)')
    parser.add_argument('--enable-depth', action='store_true', default=False,
                        help='Enable depth estimation (default: disabled)')
    parser.add_argument('--depth-model', choices=['small', 'base', 'large'], default='small',
                        help='Depth model size (default: small)')
    parser.add_argument('--remove-bg', action='store_true', default=False,
                        help='Enable background removal')
    parser.add_argument('--bg-mode', choices=['mask', 'depth', 'combined', 'blur'], 
                        default='mask',
                        help='Background removal mode (default: mask)')
    parser.add_argument('--no-overlay', action='store_true', default=False,
                        help='Disable metrics overlay (default: overlay enabled)')
    parser.add_argument('--enable-events', action='store_true', default=False,
                        help='Enable event detection (requires tracking)')
    parser.add_argument('--enable-event-serialization', action='store_true', default=False,
                        help='Save events to sessions/<timestamp>/events.jsonl')
    parser.add_argument('--enable-temporal-segmentation', action='store_true', default=False,
                        help='Enable temporal segmentation for scene change detection')
    parser.add_argument('--record', type=str, default=None,
                        help='Filepath to record video to e.g. workout.mp4)')
    parser.add_argument('--num-frames', type=int, default=200,
                        help='Number of frames to record (default: 200)')
    parser.add_argument('--fps', type=int, default=30,
                        help='FPS for recorded video (default: 30)')
    
    args = parser.parse_args()
    if args.model and not args.vlm:
        if args.model.startswith('gpt'):
            args.vlm = 'openai'
        else:
            args.vlm = 'ollama'

    if args.camera == "sony":
        print("Connecting to Sony A5000 via WiFi...")
        subprocess.Popen(
            ["./connect_alpha5000.sh", os.getenv("A5000_SSID"), os.getenv("A5000_password")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(5)
        camera = SonyA5000()
    elif args.camera == "video":
        if not args.video_path:
            raise SystemExit("--video-path is required when --camera video")
        camera = VideoFileCamera(args.video_path)
    else:
        camera = Webcam(camera_index=args.camera_index)
    
    if args.record:
        print(f"\n{'='*60}")
        print(f"RECORDING MODE")
        print(f"Camera: {args.camera}")
        print(f"Output: {args.record}")
        print(f"Frames: {args.num_frames}")
        print(f"FPS: {args.fps}")
        print(f"{'='*60}\n")
        camera.record_video(args.record, num_frames=args.num_frames, fps=args.fps)

    else:
        vision_config = VisionConfig(
            enable_scene_graph=args.enable_scene_graph,
            enable_tracking=args.enable_tracking,
            enable_segmentation=args.enable_segmentation,
            enable_box=(args.enable_box or args.enable_segmentation) and not args.no_box,
            enable_overlay=not args.no_overlay,
            enable_depth=args.enable_depth,
            depth_model_size=args.depth_model,
            enable_bg_removal=args.remove_bg,
            bg_removal_mode=args.bg_mode,
            enable_events=args.enable_events,
            enable_event_serialization=args.enable_event_serialization,
            enable_temporal_segmentation=args.enable_temporal_segmentation,
        )
        vision_config.scene_graph_vlm_provider = args.vlm
        vision_config.scene_graph_vlm_model = args.model
        print(f"\n{'='*60}")
        print(f"Camera: {args.camera}")
        print(f"Model: {vision_config.model_name}")
        if args.enable_scene_graph:
            print(f"VLM: {args.vlm} ({args.model})")
        print(f"Tracking: {args.enable_tracking}")
        if args.enable_events:
            print(f"Events: Enabled")
        if args.enable_event_serialization:
            print(f"Event Serialization: Enabled → sessions/<timestamp>/")
        print(f"{'='*60}\n")
        run(camera, vision_config=vision_config)


if __name__ == "__main__":
    main()

