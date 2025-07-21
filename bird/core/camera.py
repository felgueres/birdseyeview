import cv2
from cv2 import VideoCapture
import subprocess

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


def get_camera(i: int) -> VideoCapture:
    cap = cv2.VideoCapture(i)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    backend_name = cap.getBackendName()
    print({ 'w': w, 'h': h, 'fps': fps, 'backend_name': backend_name })

    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cv2.imshow('camout',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print('camera not opened')
            break


if __name__ == "__main__":
    get_camera(0)