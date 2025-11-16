class VisionConfig:
    def __init__(self, enable_box=False, enable_mask=True):
        self.enable_box = enable_box
        self.enable_mask = enable_mask
        self.model_name = 'yolov8n-seg.pt'
        self.confidence_threshold = 0.5
