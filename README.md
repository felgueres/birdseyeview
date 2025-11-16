# Vision fundamental

| Perception Tasks | Description | Implemented |
|------|-------------|-------------|
| Detection | What's in an image. Bounding box by frame. | X |
| Classification | Label or top-k labels. | X |
| Segmentation | Mask every pixel per class found. | X |
| Pose segmentation | Detect body joints like elbows, knees and provide coordinates. | X |
| Optical flow | For every pixel, estimate how it moved from a frame to the next. Produces motion vector field. | X |
| Tracking | Follow an object across time and assign ID. Produces trajectories of id -> sequence of positions. | |
| Re-identificatino (ReID) | Recognize a previously seen object or in another camera. Produces embedding vector + id match. | |
| Depth estimation | Predict distance on every pixel from the camera. Per-pixel depth map. | |
| 3D Object detection | Detect objects in 3D space (position, orientation, size). 3D boxes in world coordinates. | |
| Scene understanding | Build structured understanding of world (objects,layout,relations). Produces a scene graph. | |
| Event detection | Identify meaningful changes (entry,exit,fall,flight,interaction). Timed events. | |
| Action recognition | What's happening in video. Action labels + time segments. | |
| Temporal segmentation | Split by activity phases. Timeline segments. | |
| Anomaly detection | Detect unusual behavior, appearance or events. Anomaly score. | |
| Object tracking + prediction | Track + forecast future positions. Trajectory with predicted path. | |

This is the hardware I'm using:

**Sony A5000**
TODO: add specs

**Jetson Orin**   
GPU: 32 tensor cores, 1020Hhz  
CPU: 6-core Arm 64-bit CPU 1.5Mb L2 + 4MB L3, 1.7Ghz  
Memory: 8GB 128-bit LPDDR5 102 GB/s  
Power: 25W  

To run:
$ python3 -m bird.core.camera

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
