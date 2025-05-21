import argparse
import cv2
from VehicleAnalysisSystem import VehicleAnalysisSystem
from ReportVisualizer import ReportVisualizer
from VideoProcessor import VideoProcessor
import os

def main():
    parser = argparse.ArgumentParser(description="Vehicle Analysis Tool")
    parser.add_argument("mode", choices=["image", "video"], help="Mode: image or video")
    parser.add_argument("path", help="Path to image or video file")
    parser.add_argument("--output", help="Path to save output file", default=None)
    parser.add_argument("--max-frames", type=int, help="Max frames to process (video only)", default=-1)

    args = parser.parse_args()

    system = VehicleAnalysisSystem(
        vehicle_weights='car-detect-weights/weights/best.pt',
        plate_weights='car-numbers-weights/weights/best.pt',
        damage_weights='car-damage-weights/best.pt',
        brand_weights='car-brand-weights/best.pt',
        ocr_langs=['en']
    )

    if args.mode == "image":
        img = cv2.imread(args.path)
        if img is None:
            print(f"❌ Failed to load image: {args.path}")
            return

        reports = system.analyze_image(img)
        for report in reports:
            img = ReportVisualizer.draw_report(img, report)

        # Save image if --output specified
        if args.output:
            cv2.imwrite(args.output, img)
            print(f"✅ Saved annotated image to {args.output}")
        else:
            # Show image if no output file specified
            cv2.imshow("Report", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif args.mode == "video":
        output_path = args.output if args.output else "output.mp4"
        processor = VideoProcessor(system)
        processor.process_video(
            args.path,
            # output_path=output_path,
            max_frames=args.max_frames
        )
        print(f"✅ Video processed and saved to {output_path}")

if __name__ == "__main__":
    main()

# img = cv2.imread("test-data/test-plate-image1.png")
# reports = system.analyze_image(img)

# for report in reports:
#     img = ReportVisualizer.draw_report(img, report)

# cv2.imshow("Report", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# processor = VideoProcessor(system)
# processor.process_video("test-data/sample_video.mp4", output_path="output.mp4", max_frames=1000)  # max_frames=-1 — без обмежень