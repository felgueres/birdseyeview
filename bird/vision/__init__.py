from .detector import ObjectDetector
from .optical_flow import OpticalFlowTracker

__all__ = ['ObjectDetector', 'OpticalFlowTracker']

# Common object classes that might be useful for birdview/surveillance
COMMON_CLASSES = [
    'person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 
    'dog', 'cat', 'bird', 'bottle', 'chair', 'laptop'
]

VEHICLE_CLASSES = ['car', 'bicycle', 'motorcycle', 'bus', 'truck']
PERSON_CLASSES = ['person']
ANIMAL_CLASSES = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow']
