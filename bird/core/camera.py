import subprocess
from bird.config import VisionConfig
from bird.vision.detector import ObjectDetector
from bird.vision.optical_flow import OpticalFlowTracker
from bird.vision.tracker import SimpleTracker
from bird.vision.scene_graph import SceneGraphBuilder
import cv2
import requests
import numpy as np
import time
import os
import time
from dotenv import load_dotenv
load_dotenv()

class SonyMethods:
    START_LIVEVIEW = 'startLiveview'
    STOP_LIVEVIEW = 'stopLiveview'
    START_RECMODE = 'startRecMode'


class Webcam:
    """Simple webcam interface compatible with SonyA5000"""
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
    
    def stream_frames(self):
        """Generator that yields frames from webcam"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise Exception(f"Failed to open webcam at index {self.camera_index}")
        
        # Get camera info
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"✓ Webcam opened: {width}x{height} @ {fps}fps")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Failed to read frame from webcam")
                    break
                yield frame
        finally:
            if self.cap:
                self.cap.release()

class SonyA5000:
    def __init__(self, camera_ip: str = "192.168.122.1", port: int = 8080):
        self.camera_ip = camera_ip
        self.port = port
        self.base_url = f"http://{camera_ip}:{port}/sony/camera"
        self.liveview_url = None
    
    def start_recording_mode(self) -> bool:
        response = requests.post(self.base_url, json={"method": SonyMethods.START_RECMODE, "params": [], "id":1, "version":"1.0"})
        return response.status_code == 200 and 'result' in response.json()
    
    def start_liveview(self) -> bool:
        response = requests.post(self.base_url, json={"method": SonyMethods.START_LIVEVIEW, "params": [], "id":1, "version":"1.0"})
        if response.status_code == 200:
            data = response.json()
            if "result" in data and data["result"]:
                self.liveview_url = data["result"][0]
                print(f"✓ Live view URL: {self.liveview_url}")
                return True
        return False
    
    def stream_frames(self):
        """Generator that yields frames from camera"""
        if not self.start_recording_mode():
            raise Exception("Failed to start recording mode")
        
        if not self.start_liveview():
            raise Exception("Failed to start live view")
        
        session = requests.Session()
        response = session.get(self.liveview_url, stream=True)
        
        buffer = b''
        
        for chunk in response.iter_content(chunk_size=8192):
            buffer += chunk
            
            # Look for JPEG frames in buffer
            while True:
                start = buffer.find(b'\xff\xd8')  # JPEG start
                if start == -1:
                    break
                    
                end = buffer.find(b'\xff\xd9', start)  # JPEG end
                if end == -1:
                    break
                
                # Extract JPEG frame
                jpeg_data = buffer[start:end + 2]
                buffer = buffer[end + 2:]
                
                nparr = np.frombuffer(jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    yield frame

def start_vision_pipeline(camera, vision_config: VisionConfig):
    detector = ObjectDetector(vision_config=vision_config) if vision_config.enable_box else None
    flow_tracker = OpticalFlowTracker(method=vision_config.optical_flow_method) if vision_config.enable_optical_flow else None
    object_tracker = SimpleTracker(
        max_age=vision_config.tracking_max_age,
        min_hits=vision_config.tracking_min_hits,
        iou_threshold=vision_config.tracking_iou_threshold
    ) if vision_config.enable_tracking else None
    scene_graph_builder = SceneGraphBuilder(
        use_vlm=vision_config.scene_graph_use_vlm,
        vlm_model=vision_config.scene_graph_vlm_model,
        vlm_interval=vision_config.scene_graph_vlm_interval
    ) if vision_config.enable_scene_graph else None

    frame_count = 0

    for frame in camera.stream_frames():
        # 1. Object Detection - detect objects
        detections = []
        tracked_objects = []
        if detector:
            detections = detector.detect_objects(frame)
            # Tracks object movement
            if object_tracker:
                tracked_objects = object_tracker.update(detections)
                frame = detector.draw_tracks(frame, tracked_objects)
                if frame_count % 30 == 0:
                    print(f"Tracking {object_tracker.get_active_track_count()} objects:")
                    for obj in tracked_objects:
                        print(f"  - ID:{obj['track_id']} {obj['class']}: {obj['confidence']:.2f} (trajectory: {len(obj['trajectory'])} points)")
            # Otherwise, just draws boxes
            else:
                frame = detector.draw_detections(frame, detections)
                if frame_count % 30 == 0 and detections:
                    print(f"Detected {len(detections)} objects:")
                    for det in detections:
                        print(f"  - {det['class']}: {det['confidence']:.2f}")
        
        # 2. Scene Graph - VLM analysis and draw (overrides visualizations on VLM frames)
        if scene_graph_builder:
            scene_graph = scene_graph_builder.build_graph(frame)
            if scene_graph:  # Only on VLM frames
                frame = scene_graph_builder.draw_scene_graph(frame, scene_graph)
                description = scene_graph_builder.format_graph_natural_language(scene_graph)
                print(f"\n=== Scene Graph ===")
                print(description)
                print(f"==================\n")
        
        # 3. Optical Flow - compute and draw
        if flow_tracker:
            if vision_config.optical_flow_method == 'lucas_kanade':
                old_pts, new_pts = flow_tracker.compute_sparse_flow(frame)
                if old_pts is not None and new_pts is not None:
                    frame = flow_tracker.draw_sparse_flow(frame, old_pts, new_pts)
                    
                    if frame_count % 30 == 0:
                        stats = flow_tracker.get_flow_statistics(old_points=old_pts, new_points=new_pts)
                        print(f"Optical Flow - Motion energy: {stats['motion_energy']:.2f}, Tracked points: {stats['num_tracked_points']}")
        
        # Display and control
        cv2.imshow('BirdView Camera Feed', frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    # Choose camera source
    camera_source = sys.argv[1] if len(sys.argv) > 1 else "webcam"
    
    if camera_source == "sony":
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

    elif camera_source == "webcam":
        print("Using webcam...")
        camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        camera = Webcam(camera_index=camera_index)

    else:
        print(f"Unknown camera source: {camera_source}")
        print("Usage: python -m bird.core.camera [webcam|sony] [webcam_index]")
        print("Examples:")
        print("  python -m bird.core.camera webcam      # Use default webcam (index 0)")
        print("  python -m bird.core.camera webcam 1    # Use webcam at index 1")
        print("  python -m bird.core.camera sony        # Use Sony A5000")
        sys.exit(1)
    
    vision_config = VisionConfig(enable_scene_graph=True, enable_tracking=True)
    start_vision_pipeline(camera, vision_config=vision_config)
