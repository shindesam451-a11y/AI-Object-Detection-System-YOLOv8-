from ultralytics import YOLO
import cv2
import config
import utils

def webcam_detection():
    """
    Real-time object detection using webcam.
    Press 'Q' to quit, 'S' to save a frame.
    """
    # Load model
    print(f"Loading model: {config.MODEL_NAME}")
    model = YOLO(config.MODEL_NAME)
    
    # Open webcam
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam opened successfully!")
    print("Controls: Press 'Q' to quit, 'S' to save frame")
    print("-" * 50)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        frame_count += 1
        
        # Run YOLOv8 inference
        results = model(frame, verbose=False)
        
        # Draw detections on frame
        annotated_frame = results[0].plot()
        
        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display output
        cv2.imshow("YOLOv8 Real-Time Object Detection", annotated_frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            break
        elif key == ord('s') or key == ord('S'):
            output_path = utils.get_output_path(f"webcam_capture_{frame_count}", "jpg")
            cv2.imwrite(output_path, annotated_frame)
            print(f"✓ Frame saved: {output_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ Webcam session ended. Processed {frame_count} frames.")

if __name__ == "__main__":
    # Create output directories
    utils.create_output_directories()
    
    # Start webcam detection
    webcam_detection()