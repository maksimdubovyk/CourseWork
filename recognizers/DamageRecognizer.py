import torch
import numpy as np
from typing import List
from .DetectionResult import DetectionResult

class DamageRecognizer:
    def __init__(self, yolo_weights_path: str, yolo_conf=0.5, yolo_iou=0.5):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights_path)
        self.model.conf = yolo_conf
        self.model.iou = yolo_iou

        self.class_names = [
            'Front-windscreen-damage', 'Headlight-damage', 'Rear-windscreen-Damage',
            'Runningboard-Damage', 'Sidemirror-Damage', 'Taillight-Damage',
            'bonnet-dent', 'boot-dent', 'doorouter-dent', 'fender-dent',
            'front-bumper-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
        ]

    def detect_damages(self, image: np.ndarray) -> List[DetectionResult]:
        results = self.model(image)
        boxes = results.xyxy[0].cpu().numpy()

        detected = []
        for x1, y1, x2, y2, conf, cls in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_name = self.class_names[int(cls)]
            detection = DetectionResult(class_name, conf, (x1, y1, x2, y2))
            detected.append(detection)

        return detected