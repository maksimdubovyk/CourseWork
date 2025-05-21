from collections import defaultdict
import time
from typing import List
import numpy as np
from recognizers.VehicleRecognizer import VehicleRecognizer
from recognizers.PlateRecognizer import PlateRecognizer
from recognizers.DamageRecognizer import DamageRecognizer
from recognizers.CarBrandRecognizer import CarBrandRecognizer
from recognizers.ColorRecognizer import ColorRecognizer, ColorName
from ImageUtils import ImageUtils
from recognizers.DetectionResult import DetectionResult
from RecognitionReport import RecognitionReport


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

        # counters
        self.timing_data = defaultdict(float)
        self.call_count = 0
        self.total_vehicle_count = 0

    def analyze_image(self, image: np.ndarray) -> List[RecognitionReport]:
        self.call_count += 1
        reports = []

        # VEHICLE RECOGNITION
        start = time.time()
        vehicle_detections = self.vehicle_recognizer.detect_vehicles(image)
        self.timing_data["VehicleRecognizer"] += time.time() - start

        self.total_vehicle_count += len(vehicle_detections)

        for vehicle in vehicle_detections:
            crop = ImageUtils.extract_plate_image(image, vehicle.box)

            # PLATE DETECTION
            start = time.time()
            plate_detection: DetectionResult | None = self.plate_recognizer.detect_plate(crop)
            self.timing_data["PlateRecognizer - detect_plate"] += time.time() - start

            plate_number = None
            if plate_detection:
                plate_crop = ImageUtils.extract_plate_image(crop, plate_detection.box)
                start = time.time()
                plate_number, _ = self.plate_recognizer.recognize_text(plate_crop)
                self.timing_data["PlateRecognizer - recognize_text"] += time.time() - start

            # DAMAGE DETECTION
            start = time.time()
            damage_detections = self.damage_recognizer.detect_damages(crop)
            self.timing_data["DamageRecognizer"] += time.time() - start
            if not damage_detections:
                damage_detections = None

            # COLOR DETECTION
            car_color = None
            try:
                start = time.time()
                car_color = self.color_recognizer.recognize_color(crop)
                self.timing_data["ColorRecognizer"] += time.time() - start
            except Exception:
                pass

            # BRAND DETECTION
            start = time.time()
            brand_detections = self.brand_recognizer.detect_brands(crop)
            self.timing_data["CarBrandRecognizer"] += time.time() - start
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

    def write_average_times(self, output_file: str = "recognition_times_avg.log"):
        with open(output_file, "w") as f:
            f.write(f"🔎 Загальна кількість кадрів: {self.call_count}\n")
            f.write(f"🚗 Загальна кількість розпізнаних авто: {self.total_vehicle_count}\n\n")

            for name, total_time in self.timing_data.items():
                time_per_frame = total_time / self.call_count if self.call_count else 0
                time_per_vehicle = total_time / self.total_vehicle_count if self.total_vehicle_count and name != "VehicleRecognizer" else None

                f.write(f"{name}:\n")
                f.write(f"  ├─ Середній час на кадр:   {time_per_frame:.4f} sec\n")
                if time_per_vehicle is not None:
                    f.write(f"  └─ Середній час на авто:   {time_per_vehicle:.4f} sec\n")
                else:
                    f.write(f"  └─ (не обчислюється на авто — це детектор авто)\n")
                f.write("\n")