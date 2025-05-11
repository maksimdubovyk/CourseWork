import cv2
import numpy as np
from enum import Enum

class ColorName(Enum):
    RED = "Red"
    ORANGE = "Orange"
    YELLOW = "Yellow"
    GREEN = "Green"
    BLUE = "Blue"
    PURPLE = "Purple"
    BLACK = "Black"
    WHITE = "White"
    GRAY = "Gray"
    UNKNOWN = "Unknown"

class ColorRecognizer:
    def __init__(self, crop_scale: float = 0.75):
        self.crop_scale = crop_scale

        self.hsv_ranges = {
            ColorName.RED:    [(np.array([0, 70, 50]), np.array([10, 255, 255])), (np.array([170, 70, 50]), np.array([180, 255, 255]))],
            ColorName.ORANGE: [(np.array([11, 100, 100]), np.array([25, 255, 255]))],
            ColorName.YELLOW: [(np.array([26, 100, 100]), np.array([34, 255, 255]))],
            ColorName.GREEN:  [(np.array([35, 52, 72]), np.array([85, 255, 255]))],
            ColorName.BLUE:   [(np.array([86, 80, 2]), np.array([125, 255, 255]))],
            ColorName.PURPLE: [(np.array([126, 100, 100]), np.array([150, 255, 255]))],
            ColorName.BLACK:  [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            ColorName.WHITE:  [(np.array([0, 0, 200]), np.array([180, 50, 255]))],
            ColorName.GRAY:   [(np.array([0, 0, 51]), np.array([180, 50, 199]))]
        }

    def crop_center(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        new_w = int(w * self.crop_scale)
        new_h = int(h * self.crop_scale)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        return image[y1:y1 + new_h, x1:x1 + new_w]

    def recognize_color(self, image: np.ndarray) -> ColorName:
        cropped = self.crop_center(image)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        total_pixels = hsv.shape[0] * hsv.shape[1]
        color_areas = {}

        for color, ranges in self.hsv_ranges.items():
            mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                mask_total = cv2.bitwise_or(mask_total, mask)

            area = cv2.countNonZero(mask_total)
            color_areas[color] = area

        if not color_areas:
            return ColorName.UNKNOWN

        sorted_colors = sorted(color_areas.items(), key=lambda item: item[1], reverse=True)
        dominant_color, dominant_area = sorted_colors[0]

        if dominant_area / total_pixels < 0.1:
            return ColorName.UNKNOWN

        if len(sorted_colors) > 1:
            COLOR_OVERRIDE_RATIO = 0.2
            second_color, second_area = sorted_colors[1]
            if second_area >= dominant_area * COLOR_OVERRIDE_RATIO:
                return second_color

        return dominant_color