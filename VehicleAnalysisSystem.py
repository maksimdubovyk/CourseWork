from typing import List
import numpy as np
from recognizers.VehicleRecognizer import VehicleRecognizer
from recognizers.PlateRecognizer import PlateRecognizer
from recognizers.DamageRecognizer import DamageRecognizer
from recognizers.CarBrandRecognizer import CarBrandRecognizer
from ImageUtils import ImageUtils
from recognizers.DetectionResult import DetectionResult
from RecognitionReport import RecognitionReport
from recognizers.ColorRecognizer import ColorRecognizer, ColorName

class VehicleAnalysisSystem:
    def __init__(self,
                 vehicle_weights: str,
                 plate_weights: str,
                 damage_weights: str,
                 brand_weights: str,
                 ocr_langs=None):
        self.vehicle_recognizer = VehicleRecognizer(vehicle_weights)
        self.plate_recognizer = PlateRecognizer(plate_weights, ocr_langs)
        self.damage_recognizer = DamageRecognizer(damage_weights)
        self.color_recognizer = ColorRecognizer()
        self.brand_recognizer = CarBrandRecognizer(brand_weights) 

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
                print(f"Error while recognizing color: {e}")

            brand_detections = self.brand_recognizer.detect_brands(crop)
            car_brand = brand_detections[0].class_name if brand_detections else None

            report = RecognitionReport(
                car_detection=vehicle,
                plate_detection=plate_detection,
                damage_detections=damage_detections,
                plate_number=plate_number,
                car_color=car_color,
                car_brand=car_brand
            )
            reports.append(report)

        return reports