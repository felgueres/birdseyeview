# Vision fundamental

| Perception Tasks | Description | Implemented |
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

This is the hardware I'm using:

**Sony A5000**
TODO: add specs

**Jetson Orin**   
GPU: 32 tensor cores, 1020Hhz  
CPU: 6-core Arm 64-bit CPU 1.5Mb L2 + 4MB L3, 1.7Ghz  
Memory: 8GB 128-bit LPDDR5 102 GB/s  
Power: 25W  

## Usage

### Basic Usage

Run with **webcam** (default):
```bash
python3 -m bird.core.camera                    # Webcam with tracking only
python3 -m bird.core.camera --camera-index 1   # Use webcam at index 1
```

Run with **Sony A5000**:
```bash
python3 -m bird.core.camera sony
```

### With Scene Graph (VLM Analysis)

Use **Ollama** (local, free):
```bash
python3 -m bird.core.camera --enable-scene-graph                     # Uses llava:7b (default)
python3 -m bird.core.camera --enable-scene-graph --model llava:13b   # Auto-detects ollama provider
```

Use **OpenAI GPT-4o** (API, requires OPENAI_API_KEY):
```bash
export OPENAI_API_KEY="sk-..."
python3 -m bird.core.camera --enable-scene-graph --model gpt-4o       # Auto-detects openai provider
python3 -m bird.core.camera --enable-scene-graph --model gpt-4o-mini  # Cheaper option
```

The system automatically detects the VLM provider based on the model name:
- Models starting with `gpt-` → OpenAI
- Models with `:` (like `llava:7b`) → Ollama
- You can still explicitly specify `--vlm ollama` or `--vlm openai` if needed

### Options

```bash
--camera-index N          # Webcam index (default: 0)
--vlm [ollama|openai]     # VLM provider (default: ollama)
--model MODEL_NAME        # Model name (default: llava:7b for ollama, gpt-4o for openai)
--enable-scene-graph      # Enable VLM scene graph analysis
--enable-tracking         # Enable object tracking (enabled by default)
--enable-box              # Enable bounding boxes (enabled by default)
```

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

**Applications**
- Automate finding key moments


## VLM Setup

### Option 1: Ollama (Local, Free)

```bash
brew install ollama
brew services start ollama

# Pull a model
ollama pull llava:7b

# If 16GB+ RAM:
ollama pull llava:13b
```

### Option 2: OpenAI (API, Paid)

```bash
pip install openai

# Set your API key
export OPENAI_API_KEY="sk-..."
```

## Notes
- Seems like all processors have a) calculate, b) draw methods, maybe that's a better refactor here