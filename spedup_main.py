import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import face_recognition
# Load the YOLOv8 model
model = YOLO('yolov8n-face.pt')

# Load the trained SVM classifier
with open('svm_classifier_mega.pkl', 'rb') as f:
    svm_clf = pickle.load(f)
# Load the video stream
    
highest_prob = 0.0
highest_label = None

video_path = "VID_20240313_131305.mp4"
cap = cv2.VideoCapture(video_path)
# Get video properties

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_final2.mp4', fourcc, fps, (width, height))
# Get the frame size

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the two straight lines
line1_x = 150
line2_x = frame_width - 150

# Initialize head count
head_count = 0
# Initialize set to store processed IDs
processed_ids = set()
total_ids = []
labels_entered = []
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf=0.70, classes=0, persist=True)

        annotated_frame = results[0].plot()
        bbox = results[0].boxes.xyxy.to('cuda').tolist()
        id_tensor = results[0].boxes.id
        id_values = []

        if bbox != []:
            x1, y1, x2, y2 = bbox[0]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            expand_amount = 20
            expanded_x1 = max(0, x1 - expand_amount)
            expanded_y1 = max(0, y1 - expand_amount)
            expanded_x2 = min(frame.shape[1], x2 + expand_amount)
            expanded_y2 = min(frame.shape[0], y2 + expand_amount)
            frame_copy = frame.copy()
            face_region = frame_copy[int(expanded_y1):int(expanded_y2), int(expanded_x1):int(expanded_x2)]
            #face_region = cv2.resize(face_region, (0, 0), fx=0.5, fy=0.5)
            face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(face_region_rgb)

            # Draw a circle at the center of the bounding box
            # cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            if id_tensor is not None:
                id_values = [int(id_item) for id_item in id_tensor.tolist()]
                print("Detected object IDs:", id_values)
            else:
                print("No ID associated with the detected object.")

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
                #cv2.putText(annotated_frame, highest_label+" "+ str(highest_prob), (int(x1)+10, int(y1) - 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

                #cv2.rectangle(face_region, (left, top), (right, bottom), (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, label, (int(x1)+10, int(y1) - 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

            # for id_value in id_values:
            #     if id_value not in total_ids:
            #         if center_x < line1_x or center_x > line2_x:
            #             head_count += 1
            #             total_ids.append(id_value) 
            #         highest_prob = 0.0
            #         highest_label = None

            if center_x < line1_x or center_x > line2_x:
                for id_value in id_values:
                    if id_value not in total_ids:
                        head_count += 1
                        total_ids.append(id_value) 
                        highest_prob = 0.0
                        highest_label = None
                # Get the ID of the head
        


        cv2.putText(annotated_frame, f'Heads: {head_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'pid: {total_ids}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw the two lines
        cv2.line(annotated_frame, (line1_x, 0), (line1_x, frame_height), (0, 255, 0), 2)
        cv2.line(annotated_frame, (line2_x, 0), (line2_x, frame_height), (0, 255, 0), 2)

        cv2.imshow('Video', annotated_frame)
        out.write(annotated_frame)

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
