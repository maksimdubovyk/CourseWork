import torch
import numpy as np
from typing import List
from DetectionResult import DetectionResult

class VehicleRecognizer:
    def __init__(self, yolo_weights_path: str, yolo_conf=0.5, yolo_iou=0.5):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights_path)
        self.model.conf = yolo_conf
        self.model.iou = yolo_iou
        self.class_names = ['bus', 'car', 'truck', 'van']

    def detect_vehicles(self, image: np.ndarray) -> List[DetectionResult]:
        results = self.model(image)
        boxes = results.xyxy[0].cpu().numpy()

        detected = []
        for x1, y1, x2, y2, conf, cls in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_name = self.class_names[int(cls)]
            detected.append(DetectionResult(class_name, conf, (x1, y1, x2, y2)))

        return detected