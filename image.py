from ultralytics import YOLO
import cv2
import config
import utils
import os

def detect_image(image_path):
    """
    Detect objects in a single image.
    
    Args:
        image_path (str): Path to the image file
    """
    # Check if file exists
    if not utils.check_file_exists(image_path):
        return
    
    # Load model
    print(f"Loading model: {config.MODEL_NAME}")
    model = YOLO(config.MODEL_NAME)
    
    # Run inference
    print(f"Processing image: {image_path}")
    results = model.predict(
        source=image_path,
        conf=config.CONFIDENCE_THRESHOLD,
        save=True,
        save_dir=config.OUTPUT_PATH,
        verbose=False
    )
    
    # Log detections
    utils.log_detections(results, "image")
    
    print(f"✓ Detection completed!")
    print(f"✓ Results saved to: {config.OUTPUT_PATH}")

if __name__ == "__main__":
    # Create output directories
    utils.create_output_directories()
    
    # Example: detect objects in image.jpg
    image_file = "image.jpg"
    
    if os.path.exists(image_file):
        detect_image(image_file)
    else:
        print(f"Please place an image file named '{image_file}' in the project directory")
        print(f"or modify the image_file variable in this script")