This project builds vision + reasoning tools. An application I'm exploring is
intelligence for physical assets.

| Examples                                   |                                                    |
| ------------------------------------------ | -------------------------------------------------- |
| Segmenting solar plants on video           | <img src="/docs/interactive_ui.png" width="350px"> |
| Tracking progress on data center buildouts | <img src="/docs/fairwater.gif" width="350px">      |

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

```bash
# To inits the UI
python3 server/app.py
```

## Tasks

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
