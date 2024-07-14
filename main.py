from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

def calculate_pixel_mm_ratio(image_path, known_width_mm):
    # Load the image
    img = cv2.imread(image_path)
    
    # Run YOLOv8 inference on the image
    results = model(img)
    
    # Get the first detected object (assuming it's our reference object)
    boxes = results[0].boxes
    if len(boxes) == 0:
        raise ValueError("No objects detected in the calibration image")
    
    # Get the bounding box of the first detected object
    x1, y1, x2, y2 = boxes[0].xyxy[0]
    pixel_width = x2 - x1
    
    # Calculate pixel to mm ratio
    # pixel_mm_ratio = known_width_mm / pixel_width
    pixel_mm_ratio = 1
    
    return pixel_mm_ratio

# Function to calculate dimensions
def calculate_dimensions(bbox, pixel_mm_ratio):
    width = bbox[2] * pixel_mm_ratio
    height = bbox[3] * pixel_mm_ratio
    return width/10, height/10

# Main process for real-time video
def process_video_stream(pixel_mm_ratio):
    # Open webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame,conf=0.65)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                cls = int(box.cls[0])
                class_name = model.names[cls]
                conf = float(box.conf[0])
                
                
                width_mm, height_mm = calculate_dimensions((x1, y1, bbox_width, bbox_height), pixel_mm_ratio)
                
                
                # Draw bounding box and dimensions on frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                
                label_top = f'{class_name}'
                label_bottom = f'{width_mm:.1f}cm x {height_mm:.1f}cm'
                

                cv2.putText(frame, label_top, (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(frame, label_bottom, (int(x1), int(y2)+25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        
        label_head = "Place Object in Upright Position."
        cv2.putText(frame, label_head, (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,200), 2)

        # Display the result
        cv2.imshow('Real-time Object Detection and Measurement', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
calibration_image_path = 'calib.jpg'
known_width_mm = 75*7  # The known width of the reference object in millimeters

try:
    pixel_mm_ratio = calculate_pixel_mm_ratio(calibration_image_path, known_width_mm)
    print(f"Calculated pixel to mm ratio: {pixel_mm_ratio}")
    process_video_stream(pixel_mm_ratio)
except Exception as e:
    print(f"An error occurred: {e}")




