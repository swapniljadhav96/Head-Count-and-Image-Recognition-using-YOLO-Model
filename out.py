from collections import defaultdict

import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import pickle
with open('svm_classifier_mega.pkl', 'rb') as f:
    svm_clf = pickle.load(f)
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "final_vid.mp4"
cap = cv2.VideoCapture(video_path)


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('head_count_output_final.mp4', fourcc, fps, (width, height))

# Store the track history
track_history = defaultdict(lambda: [])

# Initialize variables for head count and previous point status
prev_centers = {}
inside_quadrilateral = {}

# Function to check if a point is inside the quadrilateral
def point_inside_quadrilateral(point, quadrilateral):
    x, y = point
    a, b, c, d = quadrilateral
    return (a[0] - b[0]) * (y - b[1]) - (x - b[0]) * (a[1] - b[1]) > 0 and \
           (b[0] - c[0]) * (y - c[1]) - (x - c[0]) * (b[1] - c[1]) > 0 and \
           (c[0] - d[0]) * (y - d[1]) - (x - d[0]) * (c[1] - d[1]) > 0 and \
           (d[0] - a[0]) * (y - a[1]) - (x - a[0]) * (d[1] - a[1]) > 0

# Load the quadrilateral coordinates
quadrilateral_coordinates = [(76, 158), (82, 662), (362, 661), (357, 159)]

# Initialize head count
in_count = 0
label = None
out_count = 0
highest_prob = 0.0
highest_label = None
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True,classes = 0)
        
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        bbox = results[0].boxes.xyxy.to('cpu').tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        if bbox != []:
            x1, y1, x2, y2 = bbox[0]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            expand_amount = 20
            expanded_x1 = max(0, x1)
            expanded_y1 = max(0, y1)
            expanded_x2 = min(frame.shape[1], x2)
            expanded_y2 = min(frame.shape[0], y2)
            frame_copy = frame.copy()
            face_region = frame_copy[int(expanded_y1):int(expanded_y2), int(expanded_x1):int(expanded_x2)]
            face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(face_region_rgb)

            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_encoding = face_recognition.face_encodings(face_region_rgb, [(top, right, bottom, left)])[0]
                label = svm_clf.predict([face_encoding])[0]
                prob = np.max(svm_clf.predict_proba([face_encoding]))

                if prob > highest_prob:
                    highest_prob = prob
                    highest_label = label

                if highest_prob < 0.35:
                    highest_label = "None"
                print(label)
                cv2.putText(annotated_frame, label, (int(x1)+10, int(y1) - 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
            

        # Update the head count based on the previous and current point status
        for box, track_id in zip(boxes, track_ids):
            center = (box[0],box[1])
            prev_center = prev_centers.get(track_id, None)
            prev_inside = inside_quadrilateral.get(track_id, False)

            if prev_center is not None and point_inside_quadrilateral(prev_center, quadrilateral_coordinates) and not point_inside_quadrilateral(center, quadrilateral_coordinates):
                out_count += 1
                highest_prob = 0.0
                highest_label = None
            elif prev_center is not None and not point_inside_quadrilateral(prev_center, quadrilateral_coordinates) and point_inside_quadrilateral(center, quadrilateral_coordinates):
                in_count += 1

            # Update the previous center and its status
            prev_centers[track_id] = center
            inside_quadrilateral[track_id] = point_inside_quadrilateral(center, quadrilateral_coordinates)

        # Display the annotated frame with the head count
        #cv2.putText(annotated_frame, f'In Count: {in_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(annotated_frame, f'Out Count: {out_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.polylines(annotated_frame, [np.array(quadrilateral_coordinates)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
