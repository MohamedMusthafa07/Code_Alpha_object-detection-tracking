# Models

This folder holds the pretrained YOLO weights. The `.pt` file is **not** stored in Git (it is ignored by `.gitignore`).

Default model: **`yolo11n.pt`** (YOLO11 "nano", about 5 MB, trained on the 80 COCO classes, fast on CPU).

## Option 1 - automatic download (easiest)

From the project root:

```
python main.py --webcam --download-model
```

If `models/yolo11n.pt` is missing, the official file is fetched once from the Ultralytics GitHub releases and saved here. After that, no internet is needed.

## Option 2 - manual download

1. Open the official Ultralytics model documentation: https://docs.ultralytics.com/models/yolo11/
2. Download `yolo11n.pt` from the model table there (it links to the official GitHub release assets).
3. Save the file as `models/yolo11n.pt`.

## Using a different model

Any Ultralytics detection `.pt` file works (for example `yolo11s.pt`, which is more accurate but slower):

```
python main.py --webcam --model models/yolo11s.pt --download-model
```
