"""Real-time driver drowsiness detection using a webcam and OpenCV."""

from __future__ import annotations

import argparse
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2


@dataclass
class DetectorConfig:
    camera_index: int = 0
    closed_frames_limit: int = 20


CASCADE_URLS = {
    "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_eye_tree_eyeglasses.xml": "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml",
}


def get_cascade_path(filename: str) -> str:
    """Keep OpenCV cascade files inside this project and download them once if needed."""
    asset_folder = Path(__file__).resolve().parent / "assets"
    asset_folder.mkdir(exist_ok=True)
    cascade_path = asset_folder / filename
    if not cascade_path.exists():
        print(f"Downloading {filename} for first-time setup...")
        try:
            urlretrieve(CASCADE_URLS[filename], cascade_path)
        except Exception as error:
            raise RuntimeError(
                "Could not download the OpenCV detector file. Check your internet connection and run the program again."
            ) from error
    return str(cascade_path)


def play_alert() -> None:
    """Play a short non-blocking alert sound."""
    if platform.system() == "Windows":
        try:
            import winsound
            winsound.Beep(1000, 450)
        except RuntimeError:
            pass
    else:
        print("\a", end="", flush=True)


def parse_args() -> DetectorConfig:
    parser = argparse.ArgumentParser(description="Webcam driver drowsiness detector")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--frames", type=int, default=20, help="Frames before alert (default: 20)")
    args = parser.parse_args()
    return DetectorConfig(camera_index=args.camera, closed_frames_limit=args.frames)


def main() -> None:
    config = parse_args()
    face_cascade = cv2.CascadeClassifier(get_cascade_path("haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(get_cascade_path("haarcascade_eye_tree_eyeglasses.xml"))
    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("OpenCV cascade files were not found. Reinstall opencv-python.")

    camera = cv2.VideoCapture(config.camera_index)
    if not camera.isOpened():
        raise RuntimeError("Webcam could not be opened. Try --camera 1 or check camera permissions.")

    closed_frames = 0
    last_alert_time = 0.0
    print("Drowsiness detector started. Press Q to stop.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(120, 120))
            status, color, eyes_found = "NO FACE DETECTED", (0, 200, 255), 0

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)
                face_gray = gray[y:y + h, x:x + w]
                detected_eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(25, 25))
                eyes_found = min(len(detected_eyes), 2)
                for ex, ey, ew, eh in detected_eyes[:2]:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 220, 0), 2)

                closed_frames = closed_frames + 1 if eyes_found < 2 else 0
                if closed_frames >= config.closed_frames_limit:
                    status, color = "DROWSINESS ALERT - WAKE UP", (0, 0, 255)
                    if time.time() - last_alert_time > 1.2:
                        threading.Thread(target=play_alert, daemon=True).start()
                        last_alert_time = time.time()
                elif eyes_found < 2:
                    status, color = "EYES NOT DETECTED - MONITORING", (0, 165, 255)
                else:
                    status, color = "ATTENTIVE", (0, 200, 0)

            height, width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (width, 82), (20, 20, 20), -1)
            cv2.putText(frame, "Driver Drowsiness Detection", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(frame, status, (18, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(frame, f"Eyes detected: {eyes_found} | Closed frames: {closed_frames}", (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("Driver Drowsiness Detection - Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
