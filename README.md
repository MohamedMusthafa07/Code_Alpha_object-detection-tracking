Built as Task 4 of the CodeAlpha internship program.

Table of Contents
Description
Features
Technologies
How it works
Project structure
Installation
Model setup
Usage
Controls
Expected output
Command-line options
Troubleshooting
Limitations
Future improvements
Internship task requirements covered
Tests
Author
Description

Each video frame is read with OpenCV and passed to a pretrained YOLO11 model, which finds objects. The detections are then passed to a SORT tracker that assigns every object a persistent tracking ID. The annotated frame (bounding box, label, ID, confidence and FPS) is shown in real time.

Features
Webcam or local video-file input (interactive menu or command-line flags)
Pretrained YOLO11 nano model (80 COCO classes); uses an NVIDIA GPU automatically if available, otherwise CPU
Bounding boxes, class labels and confidence scores
SORT tracker (Kalman filter + Hungarian matching on IoU) implemented with NumPy and SciPy
On-screen display such as Person | ID: 3 | 92%
Honest tracking: detections the tracker has not confirmed yet are drawn in grey as ID: pending
Live FPS, detection count and tracked-object count
Optional saving of the annotated video
Clear error messages for a missing webcam, invalid video path, missing or corrupt model, and missing dependencies
Safe exit with Q / Esc; camera and windows are always released
Technologies
Purpose	Technology
Language	Python 3.10 - 3.12
Video input and display	OpenCV
Object detection	Ultralytics YOLO11 (PyTorch)
Object tracking	SORT (NumPy, SciPy)
How it works
Webcam / video file
        |
        v
OpenCV capture  -->  resize (only if very wide)
        |
        v
YOLO detector   -->  list of detections (box, label, confidence)
        |
        v
SORT tracker    -->  confirmed tracks (with ID) + pending detections
        |
        v
Drawing         -->  OpenCV window (and optional .mp4 file)

A track is shown with an ID only after it has been matched in --min-hits consecutive frames (default 3), and it is drawn only in frames where it was matched to a real detection. Lost objects are never drawn from prediction alone, so the display never claims an object is tracked when it is not.

Project structure
Code_Alpha_object-detection-tracking/
├── main.py                  # entry point, CLI and main loop
├── detector/
│   └── yolo_detector.py     # YOLO model loading and inference
├── tracker/
│   └── sort_tracker.py      # SORT: Kalman filter + Hungarian matching
├── utils/
│   ├── types.py             # Detection / TrackedObject data classes
│   ├── drawing.py           # boxes, labels, FPS overlay
│   └── video.py             # webcam / video opening, validation, writer
├── tests/
│   └── test_sort_tracker.py # unit tests for the tracker
├── models/README.md         # how to get the model (weights are not committed)
├── videos/README.md         # put test videos here (not committed)
├── requirements.txt
├── .gitignore
└── README.md
Installation

Tested on Windows with VS Code. Requires 64-bit Python 3.10, 3.11 or 3.12.

1. Clone the repository

git clone https://github.com/MohamedMusthafa07/Code_Alpha_object-detection-tracking.git
cd Code_Alpha_object-detection-tracking

2. Create and activate a virtual environment

python -m venv .venv
.venv\Scripts\activate

If PowerShell blocks activation, run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. (VS Code) Select the interpreter

Press Ctrl+Shift+P, choose Python: Select Interpreter, and pick the .venv entry.

Optional GPU: pip installs a CPU build of PyTorch on Windows. For NVIDIA GPU acceleration, install the CUDA build from https://pytorch.org/get-started/locally/ . The program detects it automatically (--device auto).

Model setup

The YOLO weights are not stored in Git. The easiest way to get them is:

python main.py --webcam --download-model

This downloads the official yolo11n.pt (about 5 MB) into models/ once. After that no internet connection is needed. Manual download steps are in models/README.md.

Usage

Interactive menu (choose webcam or video file):

python main.py

Webcam mode:

python main.py --webcam

Use another camera: python main.py --webcam --camera-index 1

Video-file mode:

python main.py --video videos/sample.mp4

More examples:

python main.py --video videos/sample.mp4 --classes 0 2
python main.py --video videos/sample.mp4 --conf 0.5
python main.py --video videos/sample.mp4 --output output/result.mp4
python main.py --video videos/sample.mp4 --no-display --output output/result.mp4

--classes 0 2 detects only people (0) and cars (2).

Controls
Key	Action
Q or Esc	Quit
Close the window (X)	Quit
Ctrl+C in the terminal	Quit
Expected output

A window shows the live video. Each confirmed object has a coloured box and a label such as Person | ID: 3 | 92%, and the same object keeps the same ID and colour while it stays in view. Objects seen for fewer than 3 frames are grey and marked ID: pending. The top-left corner shows FPS, the device in use (cpu or cuda:0), the number of detections and the number of tracked objects.

Command-line options
Option	Default	Description
--webcam	-	Use the webcam
--video PATH	-	Use a local video file
--camera-index N	0	Webcam index
--model PATH	models/yolo11n.pt	YOLO weights file
--download-model	off	Download official weights if missing
--conf	0.35	Detection confidence threshold (0-1)
--imgsz	640	YOLO inference image size
--device	auto	auto, cpu or cuda
--classes ID ...	all	Only detect these COCO class IDs
--max-age	30	Frames to keep a lost track alive
--min-hits	3	Consecutive detections before an ID is shown
--max-width	1280	Downscale wider frames (0 = never)
--output FILE.mp4	-	Save the annotated video
--no-display	off	No window (use with --output)

Run python main.py --help for the full list.

Troubleshooting
Problem	Fix
Could not open webcam	Close Zoom, Teams and browser tabs that use the camera; check Windows Settings > Privacy > Camera; try --camera-index 1
Model file not found	Run with --download-model, or follow models/README.md
Automatic download failed	Check your internet connection or firewall, or download the model manually
OpenCV cannot open a window	pip uninstall opencv-python-headless then pip install opencv-python
ModuleNotFoundError	Activate .venv, run pip install -r requirements.txt, and select the .venv interpreter in VS Code
Video file not found	Check the path; quote paths with spaces: --video "videos/my clip.mp4"
Low FPS	Use --imgsz 480, --max-width 960, or an NVIDIA GPU build of PyTorch
IDs switch between objects	Raise --conf or lower --max-age; SORT does not use appearance (see Limitations)
Limitations
SORT uses only box motion, not appearance, so IDs can switch after long occlusions or when objects cross paths.
A track's label is the label of its most recent matched detection and can flicker for ambiguous objects.
YOLO11 nano favours speed over accuracy; small or distant objects can be missed.
FPS depends on your hardware; a typical laptop CPU gives roughly 5-20 FPS with the nano model.
Webcam behaviour varies between camera drivers.
Future improvements
Deep SORT or ByteTrack with appearance features for better re-identification
Object counting and line-crossing counters
Configuration file and model selection menu
FP16 inference on supported GPUs
Threaded capture for lower latency
Exporting tracking results to CSV
Internship task requirements covered
Requirement	Implementation
1. Real-time video input (webcam or file) with OpenCV	utils/video.py, main.py
2. Pretrained detection model (YOLO)	detector/yolo_detector.py
3. Process each frame and draw bounding boxes	main.py loop, utils/drawing.py
4. Object tracking with SORT	tracker/sort_tracker.py
5. Labels and tracking IDs displayed in real time	utils/drawing.py
Tests
python -m unittest discover -s tests -v
Author

Mohamed Musthafa

GitHub: MohamedMusthafa07
LinkedIn: mohamedmusthafa07

Built during the CodeAlpha internship program.