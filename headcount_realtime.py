import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-face.pt')

# Load the video stream
video_path = "face detection[1].mp4"
cap = cv2.VideoCapture(video_path)

# Get the frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the two straight lines
line1_x = 70
line2_x = frame_width - 70

# Initialize head count
head_count = 0
# Initialize set to store processed IDs
processed_ids = set()
total_ids = []
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf=0.70, classes=0, persist=True)

        annotated_frame = results[0].plot()
        bbox = results[0].boxes.xyxy.to('cpu').tolist()
        id_tensor = results[0].boxes.id
        id_values = []

        if bbox != []:
            x1, y1, x2, y2 = bbox[0]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Draw a circle at the center of the bounding box
            # cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            if id_tensor is not None:
                id_values = [int(id_item) for id_item in id_tensor.tolist()]
                print("Detected object IDs:", id_values)
            else:
                print("No ID associated with the detected object.")

            

            # Check if the center crosses the two lines
            if center_x < line1_x or center_x > line2_x:
                for id_value in id_values:
                    if id_value not in total_ids:
                        head_count += 1
                        total_ids.append(id_value) 
                # Get the ID of the head
        
                   
            


        cv2.putText(annotated_frame, f'Heads: {head_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'pid: {total_ids}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw the two lines
        cv2.line(annotated_frame, (line1_x, 0), (line1_x, frame_height), (0, 255, 0), 2)
        cv2.line(annotated_frame, (line2_x, 0), (line2_x, frame_height), (0, 255, 0), 2)

        cv2.imshow('Video', annotated_frame)

        # Pause the video if 'p' key is pressed
        key = cv2.waitKey(1) & 0xFF

        if key == ord("p"):
            cv2.waitKey(-1)  # Wait indefinitely until any key is pressed
        elif key == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
cv2.destroyAllWindows()