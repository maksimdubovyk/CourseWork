

import cv2
from VehicleAnalysisSystem import VehicleAnalysisSystem
from ReportVisualizer import ReportVisualizer

class VideoProcessor:
    def __init__(self, system: VehicleAnalysisSystem):
        self.system = system

    def process_video(self, video_path: str, output_path: str = None, max_frames: int = -1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Не вдалося відкрити відео: {video_path}")

        writer = None
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Аналіз зображення
            reports = self.system.analyze_image(frame)
            for report in reports:
                frame = ReportVisualizer.draw_report(frame, report)

            # Показ у вікні
            # cv2.imshow("Vehicle Recognition", frame)

            # Запис, якщо потрібно
            if writer:
                writer.write(frame)

            # Обмеження по кадрах
            frame_count += 1
            if max_frames > 0 and frame_count >= max_frames:
                break

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()