import cv2
from RecognitionReport import RecognitionReport

class ReportVisualizer:
    @staticmethod
    def draw_report(image, report: RecognitionReport):
        x_offset, y_offset = report.car_detection.box[0], report.car_detection.box[1]
        cv2.rectangle(image, report.car_detection.box[:2], report.car_detection.box[2:], (0, 255, 0), 2)
        cv2.putText(image, report.car_detection.class_name + ', color: ' + str(report.car_color), 
                    (x_offset, y_offset - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if report.car_plate_detection:
            plate_box = ReportVisualizer._shift_box(report.car_plate_detection.box, x_offset, y_offset)
            cv2.rectangle(image, plate_box[:2], plate_box[2:], (255, 0, 0), 2)
            text = report.car_plate_number or "Plate?"
            cv2.putText(image, text, (plate_box[0], plate_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if report.car_damage_detections:
            for damage in report.car_damage_detections:
                shifted_box = ReportVisualizer._shift_box(damage.box, x_offset, y_offset)
                cv2.rectangle(image, shifted_box[:2], shifted_box[2:], (0, 0, 255), 2)
                cv2.putText(image, damage.class_name, (shifted_box[0], shifted_box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return image

    @staticmethod
    def _shift_box(box, dx, dy):
        x1, y1, x2, y2 = box
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)