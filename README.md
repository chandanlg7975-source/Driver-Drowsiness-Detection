# Driver Drowsiness Detection

A Python webcam project that detects possible driver drowsiness using eye detection. It uses OpenCV Haar cascades to detect the driver's face and eyes. When fewer than two eyes are detected for several consecutive frames, the program displays a red warning and plays an alert sound.

## Features

- Live webcam monitoring
- Face and eye detection
- Closed-eye frame counter
- Visual drowsiness warning
- Windows alert beep

## Requirements

- Python 3.10 or later
- A working webcam

## Installation

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

The first run downloads two small OpenCV detector files automatically. Keep the internet on for that first run only.

Press `Q` in the webcam window to close the program.

## Optional settings

```powershell
python main.py --frames 20
```

- `--frames`: number of consecutive closed-eye frames before the alert.

## Important note

This is an academic demonstration project. It should not be used as the only safety system in a real vehicle.
