# Vision 

| Tasks | Description | Implemented |
|------|-------------|-------------|
| Detection | What's in an image. Bounding box by frame. | X |
| Classification | Label or top-k labels. | X |
| Segmentation | Mask every pixel per class found. | X |
| Pose segmentation | Detect body joints like elbows, knees and provide coordinates. | X |
| Optical flow | For every pixel, estimate how it moved from a frame to the next. Produces motion vector field. | X |
| Tracking | Follow an object across time and assign ID. Produces trajectories of id -> sequence of positions. | X |
| Temporal segmentation | Split by activity phases. Timeline segments. | |
| Scene understanding | Build structured understanding of world (objects,layout,relations). Produces a scene graph. | X |
| Event detection | Identify meaningful changes (entry,exit,fall,flight,interaction). Timed events. | |
| Action recognition | What's happening in video. Action labels + time segments. | |
| Depth estimation | Predict distance on every pixel from the camera. Per-pixel depth map. | |
| Anomaly detection | Detect unusual behavior, appearance or events. Anomaly score. | |
| Object tracking + prediction | Track + forecast future positions. Trajectory with predicted path. | |
| 3D Object detection | Detect objects in 3D space (position, orientation, size). 3D boxes in world coordinates. | |
| Re-identificatino (ReID) | Recognize a previously seen object or in another camera. Produces embedding vector + id match. | |

```
birdview/
├── bird/   
│   ├── __init__.py
│   ├── cli.py                  # entry point 
│   ├── config.py               
│   ├── core/                   
│   │   ├── __init__.py
│   │   ├── camera.py           # Camera interfaces (webcam, Sony A5000)
│   │   └── pipeline.py         # Vision processing
│   └── vision/                 
│       ├── __init__.py
│       ├── detector.py         # Object detection (YOLO)
│       ├── optical_flow.py     # Optical flow estimation
│       ├── scene_graph.py      # Scene understanding with VLMs
│       └── tracker.py          # Object tracking
├── yolov8n.pt                  # YOLOv8 detection model
├── yolov8n-seg.pt              # YOLOv8 segmentation model
├── yolov8n-pose.pt             # YOLOv8 pose estimation model
├── setup.py                    
├── connect_alpha5000.sh        # Sony A5000 connection script
└── README.md
```

## Hardware

**Sony A5000**
TODO: add specs

## Usage

Run with **webcam** (default):
```bash
python3 -m bird.cli                    
python3 -m bird.cli --camera-index 1 
```

With **Ollama**
```bash
python3 -m bird.cli --enable-scene-graph                     # Uses llava:7b
python3 -m bird.cli --enable-scene-graph --model llava:13b
```

With **OpenAI GPT-4o**
```bash
python3 -m bird.cli --enable-scene-graph --model gpt-4o       # Auto-detects openai provider
python3 -m bird.cli --enable-scene-graph --model gpt-4o-mini  # Cheaper option
```

```bash
--camera-index N          # Webcam index (default: 0)
--vlm [ollama|openai]     # VLM provider (default: ollama)
--model MODEL_NAME        # Model name (default: llava:7b for ollama, gpt-4o for openai)
--enable-scene-graph      # Enable VLM scene graph analysis
--enable-tracking         # Enable object tracking (enabled by default)
--enable-box              # Enable bounding boxes (enabled by default)
```


## A5000 

To inspect camera:
<!-- Starts the camera -->
curl -H "Content-Type: application/json" -d '{"method":"startRecMode","params":[],"id":1,"version":"1.0"}' http://192.168.122.1:8080/sony/camera
<!-- Get available api -->
curl -H "Content-Type: application/json" -d '{"method":"getAvailableApiList","params":[],"id":1,"version":"1.0"}' http://192.168.122.1:8080/sony/camera
<!-- Starts live view -->
curl -H "Content-Type: application/json" -d '{"method":"startLiveview","params":[],"id":1,"version":"1.0"}' http://192.168.122.1:8080/sony/camera
<!-- Stops live view -->
curl -H "Content-Type: application/json" -d '{"method":"stopLiveview","params":[],"id":1,"version":"1.0"}' http://192.168.122.1:8080/sony/camera

**API Methods on startRecMode**
- `getVersions`
- `getMethodTypes`
- `getApplicationInfo`
- `getAvailableApiList`
- `getEvent`
- `actTakePicture`
- `stopRecMode`
- `startLiveview`
- `stopLiveview`
- `actZoom`
- `setSelfTimer`
- `getSelfTimer`
- `getAvailableSelfTimer`
- `getSupportedSelfTimer`
- `getExposureCompensation`
- `getAvailableExposureCompensation`
- `getSupportedExposureCompensation`
- `setShootMode`
- `getShootMode`
- `getAvailableShootMode`
- `getSupportedShootMode`
- `getSupportedFlashMode`

## VLM 

Uses llava or gpt4o

```bash
brew install ollama
brew services start ollama
ollama pull llava:7b # If 16GB+ RAM: ollama pull llava:13b
```

## Evals 

A large-scale benchmark dataset for event recognition in surveillance video
- https://viratdata.org - aerial and ground 
- Aerials are public but labeling not available