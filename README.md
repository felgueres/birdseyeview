# Vision 

This project is a backend repo where I'm learning what's possible in the world of computer vision today.  
The idea is to have primitives to experiment with evals and product ideas.  

I'm currently using a Logicam c920s Pro 1080 and a Sony A5000 and M3 16GB for local inference. 

Some cool end golas could be to build:
- Eagle view with multi cameras
- 3d objection projections
- A video editor that lets you mask objects with temporal consistency
- Video search on events

| Tasks | Description | Implemented |
|------|-------------|-------------|
| Detection | What's in an image. Bounding box by frame. | X |
| Classification | Label or top-k labels. | X |
| Segmentation | Mask every pixel per class found. | X |
| Pose segmentation | Detect body joints like elbows, knees and provide coordinates. | X |
| Optical flow | For every pixel, estimate how it moved from a frame to the next. Produces motion vector field. | X |
| Tracking | Follow an object across time and assign ID. Produces trajectories of id -> sequence of positions. | X |
| Scene understanding | Build structured understanding of world (objects,layout,relations). Produces a scene graph. | X |
| Depth estimation | Predict distance on every pixel from the camera. Per-pixel depth map. | X |
| Temporal segmentation | Split by activity phases. Timeline segments. | |
| Event detection | Identify meaningful changes (entry,exit,fall,flight,interaction). Timed events. | |
| Action recognition | What's happening in video. Action labels + time segments. | |
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
## Usage

```bash
python3 -m bird.cli                    
python3 -m bird.cli --camera-index 1 
python3 -m bird.cli --enable-scene-graph --model llava:13b
python3 -m bird.cli --enable-scene-graph --model gpt-4o

--camera-index N          # Webcam index (default: 0)
--vlm [ollama|openai]     # defaults ollama
--enable-scene-graph      # VLM scene graph 
--enable-tracking         
--enable-box              
--enable-depth

# To install ollama vlm
brew install ollama
brew services start ollama
ollama pull llava:7b # If 16GB+ RAM: ollama pull llava:13b
```
