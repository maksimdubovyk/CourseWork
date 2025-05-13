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
    def __init__(self, crop_scale: float = 0.8):
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

    def _get_body_mask(self, car_crop: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

        exclude = {ColorName.BLACK, ColorName.GRAY}
        for color, ranges in self.hsv_ranges.items():
            if color in exclude:
                continue
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                mask_total = cv2.bitwise_or(mask_total, mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_final = np.zeros_like(mask_total)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_final, [biggest], -1, 255, -1)

        return mask_final

    def _count_color_pixels(self, pixels: np.ndarray) -> dict:
        color_counts = {}
        for color, ranges in self.hsv_ranges.items():
            count = 0
            for lower, upper in ranges:
                in_range = np.all((pixels >= lower) & (pixels <= upper), axis=1)
                count += np.count_nonzero(in_range)
            color_counts[color] = count
        return color_counts

    def recognize_color(self, car_crop: np.ndarray) -> ColorName:
        cropped = self.crop_center(car_crop)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        flat_pixels = hsv.reshape(-1, 3)

        color_counts = self._count_color_pixels(flat_pixels)
        black = color_counts.get(ColorName.BLACK, 0)
        gray = color_counts.get(ColorName.GRAY, 0)
        others = sum(v for k, v in color_counts.items() if k not in [ColorName.BLACK, ColorName.GRAY])

        if black > 0 and black >= 10 * others:
            return ColorName.BLACK
        if gray > 0 and gray >= 10 * others:
            return ColorName.GRAY

        mask = self._get_body_mask(cropped)
        masked_pixels = hsv[mask == 255]

        if masked_pixels.size == 0:
            return ColorName.UNKNOWN

        masked_counts = self._count_color_pixels(masked_pixels)
        sorted_colors = sorted(masked_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_color, dominant_area = sorted_colors[0]
        total = sum(masked_counts.values())

        if total == 0 or dominant_area / total < 0.1:
            return ColorName.UNKNOWN

        if len(sorted_colors) > 1:
            second_color, second_area = sorted_colors[1]
            if second_area >= dominant_area * 0.2:
                return second_color

        return dominant_color