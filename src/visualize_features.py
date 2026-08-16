from pathlib import Path

import cv2
import numpy as np


VIDEO_DIR = Path("data")
FEATURE_DIR = Path("features")
OUTPUT_DIR = Path("results")

VIDEOS = [
    "ball_dominos_seed5",
    "pendulum_seed5",
    "pool_seed5",
]

MASK_COLORS = [
    np.array([255, 80, 80]),   # projectile
    np.array([80, 255, 80]),   # target
]


def create_overlay(name: str) -> None:
    video_path = VIDEO_DIR / f"{name}.mp4"
    feature_path = FEATURE_DIR / f"{name}.npz"
    output_path = OUTPUT_DIR / f"{name}_overlay.mp4"

    with np.load(feature_path) as data:
        masks = data["masks"]
        tracks = data["tracks"].astype(np.float32)
        visibility = data["visibility"]
        height, width = data["frame_size"]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(width), int(height)),
    )

    frame_index = 0

    while frame_index < len(masks):
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (int(width), int(height)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for object_index, color in enumerate(MASK_COLORS):
            mask = masks[frame_index, object_index]
            rgb[mask] = 0.55 * rgb[mask] + 0.45 * color

        # Draw every fourth tracked point to avoid clutter.
        for point_index in range(0, tracks.shape[1], 4):
            if not visibility[frame_index, point_index]:
                continue

            x, y = tracks[frame_index, point_index]

            if 0 <= x < width and 0 <= y < height:
                cv2.circle(
                    rgb,
                    (int(x), int(y)),
                    2,
                    (80, 160, 255),
                    -1,
                )

        writer.write(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
        frame_index += 1

    cap.release()
    writer.release()
    print(f"Saved {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    for video_name in VIDEOS:
        create_overlay(video_name)


if __name__ == "__main__":
    main()