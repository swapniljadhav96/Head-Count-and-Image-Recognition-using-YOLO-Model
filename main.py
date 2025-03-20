import cv2
import face_recognition
import numpy as np
from sklearn.svm import SVC
import pickle
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-face.pt')


# # Load the trained SVM classifier
with open('svm_classifier_mega.pkl', 'rb') as f:
    svm_clf = pickle.load(f)

# Load the video stream
video_path = "face detection[1].mp4"
cap = cv2.VideoCapture(video_path)
# Loop through the video frames
highest_prob = 0.0
highest_label = None

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        #results = model.track(frame, conf=0.70, persist=True)

        #annotated_frame = results[0].plot()
        annotated_frame = frame.copy()
        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]
            # Predict the label for the face using the SVM classifier
            label = svm_clf.predict([face_encoding])[0]
            # Get the probability of the predicted label
            prob = np.max(svm_clf.predict_proba([face_encoding]))
            print(highest_prob)

            # Check if the probability is higher than the current highest probability
            if prob > highest_prob:
                highest_prob = prob
                highest_label = label

            # Draw a box around the face and label it
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(annotated_frame, highest_label, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Video', annotated_frame)
        # Pause the video if 'p' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            cv2.waitKey(-1)  # Wait indefinitely until any key is pressed
        elif key == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()