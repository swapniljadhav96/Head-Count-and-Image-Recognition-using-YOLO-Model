import cv2
import numpy as np
from sklearn.svm import SVC
import pickle
from ultralytics import YOLO
import face_recognition

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the trained SVM classifier
with open('svm_classifier_mega.pkl', 'rb') as f:
    svm_clf = pickle.load(f)

# Load the video stream
video_path = "face detection[1].mp4"
cap = cv2.VideoCapture(video_path)
highest_prob = 0.0
highest_label = None
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf=0.88, classes = 0,persist=True)

        annotated_frame = results[0].plot()
        bbox = results[0].boxes.xyxy.to('cuda').tolist()

        if bbox:
            x1, y1, x2, y2 = bbox[0]
            # Crop the face region from the frame
            frame_copy = frame.copy()
            face_region = frame_copy[int(y1):int(y2), int(x1):int(x2)]
            #face_region = cv2.resize(face_region, (0, 0), fx=0.5, fy=0.5)
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
                print(label)

                cv2.rectangle(face_region, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (int(x1)+10, int(y1) - 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            

            cv2.imshow('Video', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            cv2.waitKey(-1)  # Wait indefinitely until any key is pressed
        elif key == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()