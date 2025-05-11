from VehicleAnalysisSystem import VehicleAnalysisSystem
from ReportVisualizer import ReportVisualizer
from VideoProcessor import VideoProcessor
import cv2

system = VehicleAnalysisSystem(
    vehicle_weights='car-detect-weights/weights/best.pt',
    plate_weights='car-numbers-weights/weights/best.pt',
    damage_weights='car-damage-weights/best.pt',
    ocr_langs=['en']
)


img = cv2.imread("test-data/test-image2.png")
reports = system.analyze_image(img)

for report in reports:
    img = ReportVisualizer.draw_report(img, report)

cv2.imshow("Report", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# processor = VideoProcessor(system)
# processor.process_video("video2.mp4", output_path="output.mp4", max_frames=-1)  # max_frames=-1 — без обмежень