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
from concurrent.futures import ThreadPoolExecutor, as_completed



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

        def process_vehicle(vehicle):
            crop = ImageUtils.extract_plate_image(image, vehicle.box)
            futures = {}

            with ThreadPoolExecutor() as sub_executor:
                futures['plate'] = sub_executor.submit(self._detect_plate, crop)
                futures['damage'] = sub_executor.submit(self._detect_damage, crop)
                futures['color'] = sub_executor.submit(self._detect_color_safe, crop)
                futures['brand'] = sub_executor.submit(self._detect_brand, crop)

                results = {k: f.result() for k, f in futures.items()}

            report = RecognitionReport(
                car_detection=vehicle,
                plate_detection=results['plate']["detection"],
                plate_number=results['plate']["number"],
                damage_detections=results['damage'],
                car_color=results['color'],
                car_brand=results['brand']
            )
            return report

        for vehicle in vehicle_detections:
            report = process_vehicle(vehicle)
            reports.append(report)

        return reports

    def _detect_plate(self, crop):
        result = {"detection": None, "number": None}
        start = time.time()
        plate_detection = self.plate_recognizer.detect_plate(crop)
        self.timing_data["PlateRecognizer - detect_plate"] += time.time() - start
        result["detection"] = plate_detection

        if plate_detection:
            plate_crop = ImageUtils.extract_plate_image(crop, plate_detection.box)
            start = time.time()
            plate_number, _ = self.plate_recognizer.recognize_text(plate_crop)
            self.timing_data["PlateRecognizer - recognize_text"] += time.time() - start
            result["number"] = plate_number
        return result

    def _detect_damage(self, crop):
        start = time.time()
        damages = self.damage_recognizer.detect_damages(crop)
        self.timing_data["DamageRecognizer"] += time.time() - start
        return damages if damages else None

    def _detect_color_safe(self, crop):
        try:
            start = time.time()
            color = self.color_recognizer.recognize_color(crop)
            self.timing_data["ColorRecognizer"] += time.time() - start
            return color
        except Exception:
            return None

    def _detect_brand(self, crop):
        start = time.time()
        detections = self.brand_recognizer.detect_brands(crop)
        self.timing_data["CarBrandRecognizer"] += time.time() - start
        return detections[0].class_name if detections else None

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