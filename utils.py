import cv2
import os
from pathlib import Path

def create_output_directories():
    """Create necessary output directories if they don't exist."""
    directories = ["outputs", "inputs/images", "inputs/videos", "runs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def check_file_exists(file_path):
    """Check if a file exists."""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return False
    return True

def get_output_path(filename, extension=""):
    """Generate output file path."""
    if extension and not extension.startswith("."):
        extension = "." + extension
    return os.path.join("outputs", f"{filename}{extension}")

def log_detections(results, source_type="image"):
    """Log detection results."""
    for result in results:
        detections = result.boxes
        print(f"\n{'='*50}")
        print(f"Detections found: {len(detections)}")
        for i, box in enumerate(detections):
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = result.names[class_id]
            print(f"  {i+1}. {class_name} - Confidence: {confidence:.2%}")
        print(f"{'='*50}\n")

def get_video_properties(video_path):
    """Get video properties like FPS, frame count, resolution."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if fps > 0 else 0
    }

def draw_detections(frame, results, conf_threshold=0.5):
    """Draw bounding boxes and labels on frame."""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.conf[0] >= conf_threshold:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Put label
                label = f"{class_name}: {confidence:.2%}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame
