from VehicleAnalysisSystem import VehicleAnalysisSystem
from ReportVisualizer import ReportVisualizer
from VideoProcessor import VideoProcessor
import cv2

system = VehicleAnalysisSystem(
    vehicle_weights='car-detect-weights/weights/best.pt',
    plate_weights='car-numbers-weights/weights/best.pt',
    damage_weights='car-damage-weights/best.pt',
    brand_weights='car-brand-weights/best.pt',
    ocr_langs=['en']
)


img = cv2.imread("test-data/test-plate-image1.png")
reports = system.analyze_image(img)

for report in reports:
    img = ReportVisualizer.draw_report(img, report)

cv2.imshow("Report", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# processor = VideoProcessor(system)
# processor.process_video("test-data/sample_video.mp4", output_path="output.mp4", max_frames=1000)  # max_frames=-1 — без обмежень