import requests
import numpy as np
import cv2

class SonyMethods:
    START_LIVEVIEW = 'startLiveview'
    STOP_LIVEVIEW = 'stopLiveview'
    START_RECMODE = 'startRecMode'


class Webcam:
    """Simple webcam interface compatible with SonyA5000"""
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
    
    def stream_frames(self, max_frames=None):
        """Generator that yields frames from webcam
        
        Args:
            max_frames: Optional limit on number of frames to capture
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise Exception(f"Failed to open webcam at index {self.camera_index}")
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"✓ Webcam opened: {width}x{height} @ {fps}fps")
        
        frame_count = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from webcam")
                    break
                yield frame
                frame_count += 1
                
                # Stop if we've reached max_frames
                if max_frames and frame_count >= max_frames:
                    break
        finally:
            if self.cap:
                self.cap.release()
    
    def record_video(self, output_path: str, num_frames: int = 200, fps: int = 30):
        """Record a video from the webcam
        
        Args:
            output_path: Path to save video (e.g., 'workout_1.mp4')
            num_frames: Number of frames to record
            fps: Frames per second for output video
        """
        writer = None
        frame_count = 0
        
        try:
            for frame in self.stream_frames(max_frames=num_frames):
                # Initialize video writer with first frame dimensions
                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    print(f"Recording {num_frames} frames to {output_path}...")
                    print("Press 'q' to stop recording early")
                
                writer.write(frame)
                frame_count += 1
                
                # Display frame with recording indicator
                display_frame = frame.copy()
                cv2.putText(display_frame, f"REC [{frame_count}/{num_frames}]", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Recording', display_frame)
                
                # Check for 'q' key to stop early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"\nRecording stopped early at {frame_count} frames")
                    break
                
                # Print progress
                if frame_count % 50 == 0:
                    print(f"Recorded {frame_count}/{num_frames} frames")
            
            print(f"✓ Recording complete: {output_path}")
            
        finally:
            if writer:
                writer.release()
            cv2.destroyAllWindows()

class SonyA5000:
    """
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
    """
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
    
    def stream_frames(self, max_frames=None):
        """Generator that yields frames from camera
        
        Args:
            max_frames: Optional limit on number of frames to capture
        """
        if not self.start_recording_mode():
            raise Exception("Failed to start recording mode")
        
        if not self.start_liveview():
            raise Exception("Failed to start live view")
        
        session = requests.Session()
        response = session.get(self.liveview_url, stream=True)
        
        buffer = b''
        frame_count = 0
        
        for chunk in response.iter_content(chunk_size=8192):
            buffer += chunk
            
            while True:
                start = buffer.find(b'\xff\xd8')  # JPEG start
                if start == -1:
                    break
                    
                end = buffer.find(b'\xff\xd9', start)  # JPEG end
                if end == -1:
                    break
                
                jpeg_data = buffer[start:end + 2]
                buffer = buffer[end + 2:]
                
                nparr = np.frombuffer(jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    yield frame
                    frame_count += 1
                    
                    # Stop if we've reached max_frames
                    if max_frames and frame_count >= max_frames:
                        return
    
    def record_video(self, output_path: str, num_frames: int = 200, fps: int = 30):
        """Record a video from the camera
        
        Args:
            output_path: Path to save video (e.g., 'workout_1.mp4')
            num_frames: Number of frames to record
            fps: Frames per second for output video
        """
        writer = None
        frame_count = 0
        
        try:
            for frame in self.stream_frames(max_frames=num_frames):
                # Initialize video writer with first frame dimensions
                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    print(f"Recording {num_frames} frames to {output_path}...")
                    print("Press 'q' to stop recording early")
                
                writer.write(frame)
                frame_count += 1
                
                # Display frame with recording indicator
                display_frame = frame.copy()
                cv2.putText(display_frame, f"REC [{frame_count}/{num_frames}]", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Recording', display_frame)
                
                # Check for 'q' key to stop early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"\nRecording stopped early at {frame_count} frames")
                    break
                
                # Print progress
                if frame_count % 50 == 0:
                    print(f"Recorded {frame_count}/{num_frames} frames")
            
            print(f"✓ Recording complete: {output_path}")
            
        finally:
            if writer:
                writer.release()
            cv2.destroyAllWindows()

class VideoFileCamera:
    """Video file interface compatible with Webcam/SonyA5000."""
    def __init__(self, path: str):
        self.path = path
        self.cap = None

    def stream_frames(self, max_frames=None):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise Exception(f"Failed to open video file {self.path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Video opened: {self.path} ({width}x{height} @ {fps}fps)")

        frame_count = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                yield frame
                frame_count += 1

                if max_frames and frame_count >= max_frames:
                    break
        finally:
            if self.cap:
                self.cap.release()

    def record_video(self, *args, **kwargs):
        raise NotImplementedError("Recording not supported for VideoFileCamera")
