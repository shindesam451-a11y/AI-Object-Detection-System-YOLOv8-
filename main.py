"""
AI Object Detection System - Main Entry Point
YOLOv8 based object detection for images, videos, and webcam
"""

import os
import sys
import config
import utils
from image import detect_image
from vedio import detect_video
from object_detection import webcam_detection

def print_banner():
    """Display application banner."""
    print("\n" + "="*60)
    print(" " * 12 + "🎯 AI OBJECT DETECTION SYSTEM 🎯")
    print(" " * 18 + "Powered by YOLOv8")
    print("="*60 + "\n")

def print_menu():
    """Display main menu."""
    print("Select Detection Mode:")
    print("-" * 40)
    print("1. 📷  Image Detection")
    print("2. 🎥  Video Detection")
    print("3. 📹  Webcam Detection (Real-Time)")
    print("4. ⚙️   Settings")
    print("5. ℹ️   Information")
    print("6. 🚪  Exit")
    print("-" * 40)

def show_settings():
    """Display current settings."""
    print("\n" + "="*60)
    print("CURRENT SETTINGS")
    print("="*60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"Use GPU: {config.USE_GPU}")
    print(f"Webcam Index: {config.WEBCAM_INDEX}")
    print(f"Output Directory: {config.OUTPUT_PATH}")
    print("="*60 + "\n")

def show_info():
    """Display system information."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("PyTorch: Not properly installed")
    
    try:
        import cv2
        print(f"OpenCV Version: {cv2.__version__}")
    except:
        print("OpenCV: Not installed")
    
    try:
        from ultralytics import YOLO
        print(f"Ultralytics: Available")
    except:
        print("Ultralytics: Not installed")
    
    print("="*60 + "\n")

def image_detection_menu():
    """Handle image detection workflow."""
    print("\n" + "-"*40)
    print("IMAGE DETECTION MODE")
    print("-"*40)
    
    # List available images
    image_dir = config.IMAGE_INPUT_PATH
    if os.path.exists(image_dir):
        images = [f for f in os.listdir(image_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if images:
            print(f"\nAvailable images in {image_dir}:")
            for i, img in enumerate(images, 1):
                print(f"  {i}. {img}")
    
    image_path = input("\nEnter image path (or press Enter for 'image.jpg'): ").strip()
    
    if not image_path:
        image_path = "image.jpg"
    
    if os.path.exists(image_path):
        print(f"\n✓ Processing: {image_path}")
        detect_image(image_path)
    else:
        print(f"\n✗ Error: Image file not found - {image_path}")

def video_detection_menu():
    """Handle video detection workflow."""
    print("\n" + "-"*40)
    print("VIDEO DETECTION MODE")
    print("-"*40)
    
    # List available videos
    video_dir = config.VIDEO_INPUT_PATH
    if os.path.exists(video_dir):
        videos = [f for f in os.listdir(video_dir) 
                 if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))]
        if videos:
            print(f"\nAvailable videos in {video_dir}:")
            for i, vid in enumerate(videos, 1):
                print(f"  {i}. {vid}")
    
    video_path = input("\nEnter video path (or press Enter for 'video.mp4'): ").strip()
    
    if not video_path:
        video_path = "video.mp4"
    
    if os.path.exists(video_path):
        print(f"\n✓ Processing: {video_path}")
        detect_video(video_path)
    else:
        print(f"\n✗ Error: Video file not found - {video_path}")

def main():
    """Main application loop."""
    # Create output directories
    utils.create_output_directories()
    
    print_banner()
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            image_detection_menu()
        elif choice == "2":
            video_detection_menu()
        elif choice == "3":
            print("\n" + "-"*40)
            print("WEBCAM DETECTION MODE")
            print("-"*40)
            print("Starting webcam... (Press 'Q' to quit)")
            webcam_detection()
        elif choice == "4":
            show_settings()
        elif choice == "5":
            show_info()
        elif choice == "6":
            print("\n👋 Thank you for using AI Object Detection System!")
            print("Goodbye!\n")
            sys.exit(0)
        else:
            print("\n✗ Invalid choice. Please try again.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted by user.")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()
