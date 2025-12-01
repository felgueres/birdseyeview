# Vision

This project builds tools to experiment with the latest in video understanding.

Some cool end goals could be:

- Fast video search and retrieval
- Video tools to handoff to LLMs

<picture> 
  <img alt="interactive ui demo" src="/docs/interactive_ui.png" width="450px""> 
</picture>

| Tasks                 | Description                                                                                       | Implemented |
| --------------------- | ------------------------------------------------------------------------------------------------- | ----------- |
| Detection             | What's in an image. Bounding box by frame.                                                        | X           |
| Classification        | Label or top-k labels.                                                                            | X           |
| Segmentation          | Mask every pixel per class found.                                                                 | X           |
| Pose segmentation     | Detect body joints like elbows, knees and provide coordinates.                                    | X           |
| Optical flow          | For every pixel, estimate how it moved from a frame to the next. Produces motion vector field.    | X           |
| Tracking              | Follow an object across time and assign ID. Produces trajectories of id -> sequence of positions. | X           |
| Scene understanding   | Build structured understanding of world (objects,layout,relations). Produces a scene graph.       | X           |
| Depth estimation      | Predict distance on every pixel from the camera. Per-pixel depth map.                             | X           |
| Temporal segmentation | Split by activity phases. Timeline segments.                                                      | X           |
| Event detection       | Identify meaningful changes (entry,exit,fall,flight,interaction). Timed events.                   | X           |
| Track and predict     | Track + forecast future positions. Trajectory with predicted path.                                | TBD         |

```
birdview/
├── bird/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── camera.py
│   │   ├── pipeline.py
│   │   ├── dag.py
│   │   └── transforms.py
│   ├── events/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── motion.py
│   │   ├── interaction.py
│   │   ├── serializer.py
│   │   └── visualizer.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── optical_flow.py
│   │   ├── depth_estimator.py
│   │   ├── background_remover.py
│   │   ├── scene_graph.py
│   │   ├── temporal_segmenter.py
│   │   └── overlay.py
│   └── viz/
│       └── visualize_segments.py
├── server/
│   ├── app.py
│   └── templates/
│       └── index.html
|
├── modal/
│   └── cli.py
├── yolov8n.pt
├── yolov8n-seg.pt
├── yolov8n-pose.pt
├── setup.py
├── connect_alpha5000.sh
└── README.md
```

## Setup

```bash
python3 -m pip install -e .
brew install ollama
brew services start ollama
ollama pull llava:7b # If 16GB+ RAM: ollama pull llava:13b
```

## Usage

```bash
# To process video file
python3 -m bird.cli --video-path ./data/entering.mp4 --camera video --enable-events --enable-event-serialization --enable-segmentation --enable-tracking

# To query
python3 -m bird.query_events --session sessions/2025-12-01T07-40-31 --query "child jumping"

modal deploy modal/cli.py
modal app stop birdview
modal app list
```

## Interactive

```bash
python3 server/app.py
```

## Search

<picture> 
  <img alt="interactive ui demo" src="/docs/query.png" width="450px""> 
</picture>
