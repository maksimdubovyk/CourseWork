import torch
import easyocr
import numpy as np
from typing import Optional, Tuple
from .DetectionResult import DetectionResult

class PlateRecognizer:
    def __init__(self, yolo_weights_path: str, ocr_langs=None, yolo_conf=0.5, yolo_iou=0.5):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights_path)
        self.model.conf = yolo_conf
        self.model.iou = yolo_iou
        self.reader = easyocr.Reader(
            lang_list=ocr_langs or ['en'],
            recog_network='english_g2',
            verbose=False
        )

    def detect_plate(self, image: np.ndarray) -> Optional[DetectionResult]:
        results = self.model(image)
        boxes = results.xyxy[0].cpu().numpy()

        if len(boxes) == 0:
            return None

        x1, y1, x2, y2, conf, cls = boxes[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        return DetectionResult("plate", conf, (x1, y1, x2, y2))

    def recognize_text(self, plate_image: np.ndarray) -> Tuple[Optional[str], float]:
        ocr_result = self.reader.readtext(plate_image)
        if not ocr_result:
            return None, 0.0
        _, text, conf = ocr_result[0]
        return text.upper(), conf