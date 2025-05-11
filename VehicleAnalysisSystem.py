from typing import List
import numpy as np
from VehicleRecognizer import VehicleRecognizer
from PlateRecognizer import PlateRecognizer
from DamageRecognizer import DamageRecognizer
from ImageUtils import ImageUtils
from DetectionResult import DetectionResult
from RecognitionReport import RecognitionReport
from ColorRecognizer import ColorRecognizer, ColorName

class VehicleAnalysisSystem:
    def __init__(self,
                 vehicle_weights: str,
                 plate_weights: str,
                 damage_weights: str,
                 ocr_langs=None):
        self.vehicle_recognizer = VehicleRecognizer(vehicle_weights)
        self.plate_recognizer = PlateRecognizer(plate_weights, ocr_langs)
        self.damage_recognizer = DamageRecognizer(damage_weights)
        self.color_recognizer = ColorRecognizer()

    def analyze_image(self, image: np.ndarray) -> List[RecognitionReport]:
        vehicle_detections = self.vehicle_recognizer.detect_vehicles(image)
        reports = []

        for vehicle in vehicle_detections:
            crop = ImageUtils.extract_plate_image(image, vehicle.box)

            plate_detection: DetectionResult | None = self.plate_recognizer.detect_plate(crop)
            plate_number = None
            if plate_detection:
                plate_crop = ImageUtils.extract_plate_image(crop, plate_detection.box)
                plate_number, _ = self.plate_recognizer.recognize_text(plate_crop)

            damage_detections: List[DetectionResult] = self.damage_recognizer.detect_damages(crop)
            if not damage_detections:
                damage_detections = None

            car_color: ColorName | None = None
            try:
                car_color = self.color_recognizer.recognize_color(crop)
            except Exception as e:
                print(f"Помилка при розпізнаванні кольору: {e}")

            report = RecognitionReport(
                car_detection=vehicle,
                plate_detection=plate_detection,
                damage_detections=damage_detections,
                plate_number=plate_number,
                car_color=car_color
            )
            reports.append(report)

        return reports