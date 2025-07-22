import subprocess
from bird.vision.detector import ObjectDetector
import cv2
from cv2 import VideoCapture

def get_camera_names():
    cameras = []
    result = subprocess.run(['system_profiler', 'SPCameraDataType'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Model ID' in line or 'Product ID' in line:
                cam_info = line.strip().split(':')[-1].strip()
                cameras.append(cam_info)
    return cameras

def get_c920_specs():
    return {
        'max_resolution': (1920, 1080),
        'diagonal_fov_deg': 78
    }

def get_camera_with_detection(i: int, enable_detection: bool = True, 
                             model_size: str = 'yolov8n.pt',
                             classes_of_interest: list = None) -> VideoCapture:
    cap = cv2.VideoCapture(i)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cam_specs = get_c920_specs()
    
    print({'w': w, 'h': h, 'fps': fps, 'diagonal_fov_deg': cam_specs['diagonal_fov_deg']})

    detector = None
    if enable_detection: detector = ObjectDetector(model_size=model_size, confidence_threshold=0.5)
    
    frame_count = 0
    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                if detector:
                    detections = detector.detect_objects(frame)
                    if classes_of_interest:
                        detections = detector.filter_detections(detections, classes_of_interest)
                    frame = detector.draw_detections(frame, detections)
                    if frame_count % 30 == 0 and detections:
                        print(f"Detected {len(detections)} objects:")
                        for det in detections:
                            print(f"  - {det['class']}: {det['confidence']:.2f} at {det['center']}")
                
                cv2.imshow('cam', frame)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print('Camera not opened')
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Logitech C920 HD Pro Webcam with Object Detection")
    print("Controls:")
    print("  - 'q': Quit")
    print("  - Detection will show bounding boxes around detected objects")
    get_camera_with_detection(0, enable_detection=True)