from ultralytics import YOLO
import os
import config
import utils

def detect_video(video_path):
    """
    Detect objects in a video file and save results.
    
    Args:
        video_path (str): Path to the video file
    """
    # Check if file exists
    if not utils.check_file_exists(video_path):
        return
    
    # Get video properties
    props = utils.get_video_properties(video_path)
    print(f"Video Properties:")
    print(f"  Resolution: {props['width']}x{props['height']}")
    print(f"  FPS: {props['fps']}")
    print(f"  Total frames: {props['frame_count']}")
    print(f"  Duration: {props['duration']:.2f} seconds")
    
    # Load model
    print(f"\nLoading model: {config.MODEL_NAME}")
    model = YOLO(config.MODEL_NAME)
    
    # Run inference on video
    print(f"Processing video: {video_path}")
    results = model.predict(
        source=video_path,
        conf=config.CONFIDENCE_THRESHOLD,
        save=True,
        save_dir=config.OUTPUT_PATH,
        verbose=False
    )
    
    # Log detections
    utils.log_detections(results, "video")
    
    print(f"✓ Video processing completed!")
    print(f"✓ Results saved to: {config.OUTPUT_PATH}")

if __name__ == "__main__":
    # Create output directories
    utils.create_output_directories()
    
    # Example: detect objects in video.mp4
    video_file = "video.mp4"
    
    if os.path.exists(video_file):
        detect_video(video_file)
    else:
        print(f"Please place a video file named '{video_file}' in the project directory")
        print(f"or modify the video_file variable in this script")
