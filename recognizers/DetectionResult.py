class DetectionResult:
    def __init__(self, class_name: str, confidence: float, box: tuple[int, int, int, int]):
        self.class_name = class_name
        self.confidence = confidence
        self.box = box  # (x1, y1, x2, y2)

    def __repr__(self):
        return f"<{self.class_name} ({self.confidence:.2f}) at {self.box}>"

    def to_dict(self):
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "box": self.box
        }