import subprocess
from bird.vision.detector import ObjectDetector
import cv2
import requests
import numpy as np
import time
import os
from dotenv import load_dotenv
load_dotenv()

class SonyMethods:
    START_LIVEVIEW = 'startLiveview'
    STOP_LIVEVIEW = 'stopLiveview'
    START_RECMODE = 'startRecMode'

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

def start_detection_generator(sony_cam, enable_detection=True, model_size: str = 'yolov8n.pt'):
    """Detection using generator"""
    
    detector = None

    if enable_detection: 
        detector = ObjectDetector(model_size=model_size, confidence_threshold=0.5)
    
    frame_count = 0
    print("Starting detection... Press 'q' to quit")
    
    for frame in sony_cam.stream_frames():
        if detector:
            detections = detector.detect_objects(frame)
            frame = detector.draw_detections(frame, detections)
            
            if frame_count % 30 == 0 and detections:
                print(f"Detected {len(detections)} objects:")
                for det in detections:
                    print(f"  - {det['class']}: {det['confidence']:.2f} at {det['center']}")
        
        cv2.imshow('Sony A5000 WiFi Camera', frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Connecting to A5000 WiFi")
    process = subprocess.Popen(["./connect_alpha5000.sh", os.getenv("A5000_SSID"), os.getenv("A5000_password")], 
                              stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
    time.sleep(5) # takes a bit to establish connection
    sony_cam = SonyA5000()
    start_detection_generator(sony_cam, enable_detection=True)