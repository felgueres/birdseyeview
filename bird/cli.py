import argparse
import subprocess
import time
import os
from dotenv import load_dotenv

from bird.config import VisionConfig
from bird.core.camera import Webcam, SonyA5000
from bird.core.pipeline import run

load_dotenv()

def main():
    """BirdView CLI - Computer Vision Pipeline Runner"""
    parser = argparse.ArgumentParser(description='BirdView - Computer Vision Pipeline')
    parser.add_argument('camera', nargs='?', default='webcam', choices=['webcam', 'sony'],
                        help='Camera source (default: webcam)')
    parser.add_argument('--camera-index', type=int, default=0,
                        help='Webcam index (default: 0)')
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
    
    args = parser.parse_args()
    
    # Smart VLM provider and model detection
    if args.model and not args.vlm:
        # Auto-detect provider based on model name
        if args.model.startswith('gpt-'):
            args.vlm = 'openai'
        elif args.model.startswith('llava') or ':' in args.model:
            args.vlm = 'ollama'
        else:
            args.vlm = 'ollama'  # default
    elif args.vlm and not args.model:
        # Set default model based on provider
        if args.vlm == 'openai':
            args.model = 'gpt-4o'
        else:
            args.model = 'llava:7b'
    elif not args.vlm and not args.model:
        # Both not specified, use defaults
        args.vlm = 'ollama'
        args.model = 'llava:7b'
    
    # Setup camera
    if args.camera == "sony":
        print("Connecting to Sony A5000 via WiFi...")
        process = subprocess.Popen(
            ["./connect_alpha5000.sh", os.getenv("A5000_SSID"), os.getenv("A5000_password")], 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(5)
        camera = SonyA5000()
    else:  # webcam
        print("Using webcam...")
        camera = Webcam(camera_index=args.camera_index)
    
    vision_config = VisionConfig(
        enable_scene_graph=args.enable_scene_graph, 
        enable_tracking=args.enable_tracking,
        enable_segmentation=args.enable_segmentation,
        enable_box=args.enable_box or args.enable_segmentation,
        enable_overlay=not args.no_overlay,
        enable_depth=args.enable_depth,
        depth_model_size=args.depth_model,
        enable_bg_removal=args.remove_bg,
        bg_removal_mode=args.bg_mode,
    )
    vision_config.scene_graph_vlm_provider = args.vlm
    vision_config.scene_graph_vlm_model = args.model
    
    print(f"\n{'='*60}")
    print(f"Camera: {args.camera}")
    print(f"Model: {vision_config.model_name}")
    if args.enable_scene_graph:
        print(f"VLM: {args.vlm} ({args.model})")
    print(f"Tracking: {args.enable_tracking}")
    print(f"{'='*60}\n")
    
    run(camera, vision_config=vision_config)


if __name__ == "__main__":
    main()

