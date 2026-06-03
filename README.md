# AI Object Detection System (YOLOv8)

A comprehensive Python application for real-time object detection using YOLOv8, supporting images, videos, and live webcam feeds.

## Features

✨ **Object Detection Modes:**
- 📷 Image Detection - Detect objects in single images
- 🎥 Video Detection - Process video files with object detection
- 📹 Webcam Detection - Real-time detection from webcam feed

✨ **Advanced Features:**
- Confidence threshold customization
- Automatic output saving
- GPU/CPU support
- Detailed detection logging
- Frame-by-frame analysis

## Project Structure

```
AI Object Detection System (YOLOv8)/
│
├── image.py                  # Image detection script
├── object_detection.py       # Webcam detection script  
├── vedio.py                  # Video detection script
├── main.py                   # Main entry point
├── config.py                 # Configuration settings
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── inputs/
│   ├── images/              # Place your images here
│   └── videos/              # Place your videos here
│
└── outputs/                 # Detection results saved here
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ultralytics torch torchvision opencv-python numpy matplotlib Pillow
```

### 2. Verify Installation

```bash
python -c "import torch; print(torch.__version__); import cv2; print('OpenCV OK')"
```

## Configuration

Edit `config.py` to customize settings:

```python
# Model (nano=fastest, small/medium/large/xlarge=better accuracy)
MODEL_NAME = "yolov8n.pt"

# Confidence threshold (0-1)
CONFIDENCE_THRESHOLD = 0.5

# GPU usage
USE_GPU = True  # Set False for CPU only
```

## Usage

### Option 1: Image Detection

```bash
python image.py
```

**Setup:**
1. Place an image file named `image.jpg` in the project root
2. Run the script
3. Results saved in `outputs/` folder

### Option 2: Video Detection

```bash
python vedio.py
```

**Setup:**
1. Place a video file named `video.mp4` in the project root
2. Run the script
3. Results saved in `outputs/` folder with same format

### Option 3: Webcam Detection (Real-Time)

```bash
python object_detection.py
```

**Controls:**
- **Q** - Quit application
- **S** - Save current frame

**Note:** Requires webcam to be connected

### Option 4: Use Main Entry Point

```bash
python main.py
```

Interactive menu to choose detection mode.

## Supported Input Formats

**Images:** JPG, PNG, BMP, GIF, TIFF  
**Videos:** MP4, AVI, MOV, MKV, FLV  
**Webcam:** Any connected USB camera

## Model Variants

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| yolov8n | ⚡⚡⚡ Fastest | Good | 6.3 MB |
| yolov8s | ⚡⚡ Fast | Better | 21.5 MB |
| yolov8m | ⚡ Medium | Best | 49.7 MB |
| yolov8l | Medium | Excellent | 83.7 MB |
| yolov8x | Slow | Maximum | 135.0 MB |

Change in `config.py`: `MODEL_NAME = "yolov8m.pt"`

## Output

- Annotated images/videos with bounding boxes and confidence scores
- Detection logs with class names and confidence percentages
- Saved in `outputs/` directory with timestamp

## Example Output

```
==================================================
Detections found: 3
  1. person - Confidence: 95.23%
  2. car - Confidence: 87.45%
  3. bicycle - Confidence: 76.12%
==================================================
```

## Troubleshooting

### 1. PyTorch DLL Error
**Error:** `OSError: DLL initialization routine failed`

**Solution:**
```bash
pip uninstall torch torchvision
pip install torch torchvision
```

### 2. Model Download Failed
**Error:** `Model not found`

**Solution:** 
- First run automatically downloads models (~80 MB for nano)
- Ensure internet connection is active
- Models stored in `~/.yolov8/weights/`

### 3. Webcam Not Working
- Check if webcam is connected and not used by other apps
- Try: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`

### 4. Low Performance
- Use smaller model: `yolov8n` instead of `yolov8m`
- Reduce video resolution
- Enable GPU: Set `USE_GPU = True`
- Skip frames: Set `VIDEO_FRAME_SKIP = 2`

## Performance Tips

1. **For Speed:** Use `yolov8n.pt` (nano model)
2. **For Accuracy:** Use `yolov8l.pt` or `yolov8x.pt`
3. **GPU Required:** For real-time performance (NVIDIA GPU recommended)
4. **CPU Usage:** Works but slower; use for testing only

## Classes Detected

YOLOv8 detects 80 COCO dataset classes:
- Person, bicycle, car, motorcycle, bus, train, truck
- Dog, cat, horse, sheep, cow, elephant, bear, zebra
- Giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee
- Skis, snowboard, sports ball, kite, baseball bat, baseball glove
- Skateboard, surfboard, tennis racket, bottle, wine glass, cup
- Fork, knife, spoon, bowl, banana, apple, sandwich, orange
- Broccoli, carrot, hot dog, pizza, donut, cake, couch
- Bed, toilet, monitor, laptop, mouse, remote, keyboard
- Microwave, oven, toaster, sink, refrigerator, book, clock
- Vase, scissors, teddy bear, hair drier, toothbrush
- And more...

## API Reference

### image.py
- `detect_image(image_path)` - Run detection on image

### object_detection.py
- `webcam_detection()` - Start live webcam detection

### vedio.py
- `detect_video(video_path)` - Run detection on video

### utils.py
- `create_output_directories()` - Create output folders
- `check_file_exists(file_path)` - Verify file exists
- `log_detections(results)` - Print detection details
- `get_video_properties(video_path)` - Get video metadata
- `draw_detections(frame, results)` - Draw boxes on frame

## Requirements

- Python 3.8+
- GPU (NVIDIA) - Optional but recommended for speed
- 4GB+ RAM minimum
- 500MB free disk space

## License

YOLOv8 is open source. See ultralytics package for license details.

## Resources

- **YOLOv8 Documentation:** https://docs.ultralytics.com/
- **YOLOv8 GitHub:** https://github.com/ultralytics/ultralytics
- **COCO Dataset:** https://cocodataset.org/

## Support & Issues

For questions or issues:
1. Check the troubleshooting section above
2. Review YOLOv8 documentation
3. Check GitHub issues

---

**Happy Detecting! 🎯**
