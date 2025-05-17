from recognizers.DetectionResult import DetectionResult
from typing import Optional
from recognizers.ColorRecognizer import ColorName

class RecognitionReport:
    def __init__(self,
                 car_detection: DetectionResult,
                 plate_detection: Optional[DetectionResult] = None,
                 damage_detections: Optional[list[DetectionResult]] = None,
                 plate_number: Optional[str] = None,
                 car_color: Optional[ColorName] = None,
                 car_brand: Optional[str] = None):

        self.car_detection: DetectionResult = car_detection
        self.car_plate_detection: Optional[DetectionResult] = plate_detection
        self.car_damage_detections: Optional[list[DetectionResult]] = damage_detections if damage_detections else None
        self.car_plate_number: Optional[str] = plate_number
        self.car_color: Optional[ColorName] = car_color
        self.car_brand: Optional[str] = car_brand
        self.car_damages: Optional[list[str]] = (
            [d.class_name for d in damage_detections] if damage_detections else None
        )

    def to_dict(self):
        return {
            "car_detection": self.car_detection.to_dict(),
            "car_plate_detection": self.car_plate_detection.to_dict() if self.car_plate_detection else None,
            "car_damage_detections": [d.to_dict() for d in self.car_damage_detections] if self.car_damage_detections else None,
            "car_plate_number": self.car_plate_number,
            "car_color": self.car_color.value if self.car_color else None,
            "car_brand": self.car_brand,
            "car_damages": self.car_damages
        }

    def __repr__(self):
        return (
            f"<RecognitionReport plate='{self.car_plate_number}', "
            f"color={self.car_color.value if self.car_color else 'None'}, "
            f"brand='{self.car_brand or 'None'}', "
            f"damages={self.car_damages or 'None'}, "
            f"car_box={self.car_detection.box}>"
        )