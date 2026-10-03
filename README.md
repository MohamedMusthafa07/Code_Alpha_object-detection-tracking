# CodeAlpha Object Detection and Tracking System

A real-time object detection and tracking system that I built for the CodeAlpha Internship - Task 4. It reads video from a webcam or a local video file, detects objects with a pretrained YOLO11 model, follows each object with the SORT tracker, and shows the label, tracking ID and confidence live on screen. Everything runs locally on my own machine: no API keys, no cloud service and no hidden costs.

## Project Overview

I wanted to build a computer-vision project that is reliable and easy to explain, so I combined a pretrained detector (YOLO11) with a classic tracker (SORT) that I implemented myself with NumPy and SciPy. Detection finds objects in each frame, and tracking connects them across frames so every object keeps the same ID while it stays in view.

The tracker is deliberately honest. An object only gets a tracking ID after it has been matched in several consecutive frames, and until then it is drawn in grey as `ID: pending`. The program never claims that an object is tracked when the tracker has not assigned it an ID.

## Problem Statement

A detector on its own only says what is in a single frame. It cannot tell whether the person in frame 10 is the same person as in frame 11, so it cannot count objects, follow them, or analyse their movement. This project adds tracking on top of detection so each object gets a persistent ID, and it does everything locally and in real time without any paid or online service.

## Features

- Two input modes: webcam or local video file (interactive menu or command-line flags)
- Pretrained YOLO11 nano model (80 COCO classes) for detection
- Bounding box, class label and confidence score for every detection
- SORT tracking (Kalman filter + Hungarian matching on IoU) with persistent IDs
- Live display such as `Person | ID: 3 | 92%`
- Grey `ID: pending` boxes for detections the tracker has not confirmed yet
- Live FPS, detection count and tracked-object count on screen
- Automatic GPU use if available, otherwise CPU (GPU is not required)
- Optional class filter (for example only people and cars)
- Optional saving of the annotated video
- Input validation: handles an invalid video path, wrong file type, empty or corrupted video
- Clear errors for a missing webcam, a missing or corrupted model, and missing dependencies
- Safe handling of frames with no detections
- Safe exit with `Q`, `Esc` or by closing the window; the camera and windows are always released
- 10 automated unit tests for the tracker
- 100% offline after the model has been downloaded once

## Technology Stack

| Area | Tools |
|---|---|
| Language | Python 3.10+ (tested on Python 3.12) |
| Video input and display | OpenCV |
| Object detection | Ultralytics YOLO11 (PyTorch) |
| Object tracking | SORT, implemented with NumPy and SciPy |
| Testing | unittest |

I implemented SORT myself instead of installing a third-party SORT package. This keeps the dependency list short and avoids packages that often fail to install on Windows.

## Architecture

```
Webcam / video file
   -> Input validation          (utils/video.py)
   -> Frame capture + resize    (OpenCV, only if the frame is very wide)
   -> YOLO detection            (detector/yolo_detector.py)
   -> SORT tracking             (tracker/sort_tracker.py)
   -> Drawing boxes and labels  (utils/drawing.py)
   -> OpenCV window (and optional .mp4 file)
```

`main.py` only connects these parts and runs the main loop. The detector, tracker and drawing code are separate modules, which keeps the project clean and easy to test.

## How Detection and Tracking Work

**1. Frame capture.** OpenCV reads one frame at a time from the webcam or video file. Frames wider than 1280 pixels are downscaled to save processing time.

**2. Detection with YOLO11.** The pretrained YOLO11 nano model returns, for each object, a bounding box, a class label and a confidence score. Detections below the confidence threshold (default 0.35) are discarded.

**3. Tracking with SORT.** Every tracked object has a Kalman filter that predicts where its box will be in the next frame. For each new frame:

- the filters predict new box positions
- detections are matched to predictions using IoU (overlap) and the Hungarian algorithm
- matched tracks are updated, and unmatched detections start new tracks
- tracks that are lost for too long (`--max-age`, default 30 frames) are removed

**4. Confirmation.** A track only shows an ID after it has been matched in `--min-hits` consecutive frames (default 3). Before that, the detection is drawn in grey as `ID: pending`.

The confidence value on screen is the YOLO detection score for that object. It is a confidence measure of the detector, not a guarantee that the label is correct.

## Project Structure

```
Code_Alpha_object-detection-tracking/
|-- main.py                  # Entry point, command-line options, main loop
|-- detector/
|   `-- yolo_detector.py     # YOLO model loading and inference
|-- tracker/
|   `-- sort_tracker.py      # SORT: Kalman filter + Hungarian matching
|-- utils/
|   |-- types.py             # Detection / TrackedObject data classes
|   |-- drawing.py           # Boxes, labels, FPS overlay
|   `-- video.py             # Webcam / video opening, validation, writer
|-- tests/
|   `-- test_sort_tracker.py # Unit tests for the tracker
|-- models/README.md         # How to get the model (weights are not committed)
|-- videos/README.md         # Put test videos here (not committed)
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Installation

Windows (PowerShell):

```
git clone https://github.com/MohamedMusthafa07/Code_Alpha_object-detection-tracking.git
cd Code_Alpha_object-detection-tracking
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

macOS / Linux:

```
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Tip: use Python 3.10, 3.11 or 3.12 (64-bit). In VS Code, press `Ctrl+Shift+P`, choose **Python: Select Interpreter** and pick the `.venv` entry.

Optional GPU: pip installs a CPU build of PyTorch on Windows. For NVIDIA GPU acceleration, install the CUDA build from https://pytorch.org/get-started/locally/ . The program detects it automatically.

## Model Setup

The YOLO weights are not stored in Git. The easiest way to get them is:

```
python main.py --webcam --download-model
```

This downloads the official `yolo11n.pt` (about 5 MB) into `models/` once. After that, no internet connection is needed. Manual download steps are in [models/README.md](models/README.md).

## How to Run

Interactive menu (choose webcam or video file):

```
python main.py
```

Webcam mode:

```
python main.py --webcam
```

Use another camera with `--camera-index 1`.

Video-file mode:

```
python main.py --video videos/sample.mp4
```

More examples:

```
python main.py --video videos/sample.mp4 --classes 0 2
python main.py --video videos/sample.mp4 --conf 0.5
python main.py --video videos/sample.mp4 --output output/result.mp4
python main.py --video videos/sample.mp4 --no-display --output output/result.mp4
```

`--classes 0 2` detects only people (class 0) and cars (class 2).

## Controls

| Key | Action |
|---|---|
| `Q` or `Esc` | Quit |
| Close the window (X) | Quit |
| `Ctrl+C` in the terminal | Quit |

## How to Test

```
python -m unittest discover -s tests -v
```

Expected result: `Ran 10 tests ... OK`

The tests cover IoU calculation, empty detections, ID assignment after confirmation, ID persistence for a moving object, separate IDs for different objects, track removal after `max_age`, recovery after a short occlusion, and invalid parameters.

Manual checks I recommend before using the project:

| Check | Command | Expected |
|---|---|---|
| Webcam | `python main.py --webcam` | Window with boxes, labels and IDs |
| Video file | `python main.py --video videos/sample.mp4` | Video plays with boxes and IDs |
| Invalid video path | `python main.py --video nope.mp4` | `[ERROR] Video file not found` |
| No camera | `python main.py --webcam --camera-index 5` | `[ERROR] Could not open webcam` |
| Exit | Press `Q` | Window closes, camera is released |

## Command-Line Options

| Option | Default | Description |
|---|---|---|
| `--webcam` | - | Use the webcam |
| `--video PATH` | - | Use a local video file |
| `--camera-index N` | 0 | Webcam index |
| `--model PATH` | `models/yolo11n.pt` | YOLO weights file |
| `--download-model` | off | Download the official weights if missing |
| `--conf` | 0.35 | Detection confidence threshold (0-1) |
| `--imgsz` | 640 | YOLO inference image size |
| `--device` | auto | `auto`, `cpu` or `cuda` |
| `--classes ID ...` | all | Only detect these COCO class IDs |
| `--max-age` | 30 | Frames to keep a lost track alive |
| `--min-hits` | 3 | Consecutive detections before an ID is shown |
| `--max-width` | 1280 | Downscale wider frames (0 = never) |
| `--output FILE.mp4` | - | Save the annotated video |
| `--no-display` | off | No window (use together with `--output`) |

Run `python main.py --help` for the full list.

## Expected Output

A window shows the live video. Each confirmed object has a coloured box and a label such as:

```
Person | ID: 3 | 92%
Car    | ID: 7 | 87%
```

The same object keeps the same ID and colour while it stays in view. Objects seen for fewer than 3 frames are grey and marked `ID: pending`. The top-left corner shows FPS, the device in use (`cpu` or `cuda:0`), the number of detections and the number of tracked objects.

## Tracking Behavior

If the detector finds nothing, the program keeps running and the overlay shows `Detections: 0`. If an object disappears, its track is kept for up to `--max-age` frames, but it is not drawn while it has no matching detection. If it comes back quickly, it usually keeps the same ID. After that, it receives a new ID.

## Performance Considerations

- The YOLO model is loaded once and warmed up with one dummy inference before the first real frame.
- Each frame needs only one model inference and one tracker update.
- Frames wider than `--max-width` are downscaled with area interpolation.
- The GPU is used automatically when available; otherwise the CPU is used.
- IoU matching is vectorized with NumPy.
- The webcam buffer is kept small to reduce latency.
- `--classes` and `--imgsz` let me trade accuracy for speed.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open webcam` | Close Zoom, Teams and browser tabs that use the camera; check Windows Settings > Privacy > Camera; try `--camera-index 1` |
| `Model file not found` | Run with `--download-model`, or follow [models/README.md](models/README.md) |
| `Automatic download failed` | Check your internet connection or firewall, or download the model manually |
| `OpenCV cannot open a window` | `pip uninstall opencv-python-headless` then `pip install opencv-python` |
| `ModuleNotFoundError` | Activate `.venv`, run `pip install -r requirements.txt`, and select the `.venv` interpreter in VS Code |
| `Video file not found` | Check the path; quote paths with spaces: `--video "videos/my clip.mp4"` |
| Low FPS | Use `--imgsz 480`, `--max-width 960`, or an NVIDIA GPU build of PyTorch |
| IDs switch between objects | Raise `--conf` or lower `--max-age`; SORT does not use appearance (see Limitations) |

## Limitations

- SORT uses only box motion, not appearance, so IDs can switch after long occlusions or when objects cross paths.
- A track's label is the label of its most recent matched detection and can flicker for ambiguous objects.
- YOLO11 nano favours speed over accuracy, so small or distant objects can be missed.
- FPS depends on the hardware; a typical laptop CPU gives roughly 5-20 FPS with the nano model.
- Webcam behaviour varies between camera drivers.

## What I Learned

- Running a pretrained YOLO model for real-time object detection
- How the SORT tracker works: Kalman filters, IoU matching and the Hungarian algorithm
- Reading video from a webcam or file with OpenCV and releasing resources correctly
- Designing a clean, modular Python project
- Validating input and handling errors with clear messages
- Writing unit tests for the tracking logic
- Publishing and documenting a project on GitHub

## Future Improvements

- Deep SORT or ByteTrack with appearance features for better re-identification
- Object counting and line-crossing counters
- Configuration file and model selection menu
- FP16 inference on supported GPUs
- Threaded video capture for lower latency
- Exporting tracking results to CSV

## Internship Information

Developed as part of the CodeAlpha Internship - Task 4: Object Detection and Tracking.

| Requirement | Implementation |
|---|---|
| 1. Real-time video input (webcam or file) with OpenCV | `utils/video.py`, `main.py` |
| 2. Pretrained detection model (YOLO) | `detector/yolo_detector.py` |
| 3. Process each frame and draw bounding boxes | `main.py` loop, `utils/drawing.py` |
| 4. Object tracking with SORT | `tracker/sort_tracker.py` |
| 5. Labels and tracking IDs displayed in real time | `utils/drawing.py` |

## Author

A. Mohamed Musthafa

GitHub: (https://github.com/MohamedMusthafa07)

LinkedIn: (https://www.linkedin.com/in/mohamedmusthafa07)