import numpy as np
from typing import Tuple

class ImageUtils:
    @staticmethod
    def extract_plate_image(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]