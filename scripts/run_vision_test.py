import os
from pathlib import Path
import argparse
import sys

# Add the project root to the Python path to allow imports from src
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
ENV_PATH = REPO_ROOT / ".env"


def load_local_env(env_path=ENV_PATH):
    """Load simple KEY=VALUE entries from .env without adding a dependency."""
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_args():
    """Parse runtime settings for the perception demo.

    Model paths can come from command-line arguments, shell environment
    variables, or the local `.env` file. Command-line arguments win when both
    are provided, which lets each machine keep its own defaults while still
    allowing one-off overrides.
    """
    load_local_env()
    parser = argparse.ArgumentParser(description="Run RT-DETR + MobileSAM on RealSense frames.")
    parser.add_argument(
        "--detr-weights",
        default=os.getenv("DETR_WEIGHTS", str(REPO_ROOT / "models" / "best.pt")),
        help="Path to RT-DETR weights. Defaults to DETR_WEIGHTS or models/best.pt.",
    )
    parser.add_argument(
        "--sam-weights",
        default=os.getenv("SAM_WEIGHTS", str(REPO_ROOT / "models" / "mobile_sam.pt")),
        help="Path to MobileSAM weights. Defaults to SAM_WEIGHTS or models/mobile_sam.pt.",
    )
    parser.add_argument("--width", default=640, type=int, help="RealSense stream width.")
    parser.add_argument("--height", default=480, type=int, help="RealSense stream height.")
    parser.add_argument("--fps", default=30, type=int, help="RealSense stream FPS.")
    parser.add_argument("--conf", default=0.5, type=float, help="RT-DETR confidence threshold.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Keep these imports here because they require installed camera/model
    # dependencies. This lets `python scripts/run_vision_test.py --help` work
    # even on a machine that is not set up for RealSense inference yet.
    import cv2
    import numpy as np

    from src.hardware.realsense import RealSenseCamera
    from src.perception.vision import VisionPipeline

    camera = RealSenseCamera(width=args.width, height=args.height, fps=args.fps)
    vision = VisionPipeline(detr_weights=args.detr_weights, sam_weights=args.sam_weights)
    
    camera.start()
    print("Pipeline active. Press 'q' to quit.")

    try:
        while True:
            color_image, depth_frame = camera.get_frames()
            if color_image is None:
                continue

            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = vision.detect(rgb_image, conf_threshold=args.conf)

            if len(results.boxes) > 0:
                for i in range(len(results.boxes)):
                    box = results.boxes.xyxy[i].cpu().numpy()
                    x_min, y_min, x_max, y_max = map(int, box)
                    
                    class_id = int(results.boxes.cls[i].cpu().numpy())
                    confidence = float(results.boxes.conf[i].cpu().numpy())
                    class_name = vision.detr_model.names[class_id]
                    class_color = vision.get_class_color(class_id)

                    mask = vision.segment(rgb_image, box)
                    spatial_coords, cx, cy = vision.extract_3d_coordinates(depth_frame, mask)

                    # Visualization
                    cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), class_color, 2)
                    cv2.putText(color_image, f"{class_name} {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2)
                    
                    class_mask = np.zeros_like(color_image, dtype=np.uint8)
                    class_mask[mask] = class_color
                    color_image = cv2.addWeighted(color_image, 1.0, class_mask, 0.4, 0)

                    if spatial_coords:
                        x_m, y_m, z_m = spatial_coords
                        cv2.drawMarker(color_image, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                        cv2.putText(color_image, f"X:{x_m:.3f} Y:{y_m:.3f} Z:{z_m:.3f}m", (cx + 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('Modular Vision Test', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
