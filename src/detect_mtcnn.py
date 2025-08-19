import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from playsound import playsound


# Load trained CNN model
model = tf.keras.models.load_model("models/drowsiness_cnn.h5")
print("[INFO] Model loaded successfully.")

# Initialize MTCNN detector
detector = MTCNN()
print("[INFO] MTCNN initialized.")

def preprocess_eye(eye):
    eye = cv2.equalizeHist(eye)  # improve contrast
    eye = cv2.resize(eye, (224, 224))
    eye = eye / 255.0
    eye = eye.reshape(1, 224, 224, 1)
    return eye

def start_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("[INFO] Webcam started.")
    consecutive_drowsy = 0
    threshold = 15

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detector.detect_faces(rgb_frame)
        print(f"[DEBUG] MTCNN detected {len(results)} face(s)")

        for result in results:
            x, y, w, h = result['box']
            x, y = max(0, x), max(0, y)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            # Extract eyes from MTCNN keypoints
            keypoints = result['keypoints']
            left_eye = keypoints.get('left_eye')
            right_eye = keypoints.get('right_eye')

            if left_eye and right_eye:
                for eye_point in [left_eye, right_eye]:
                    ex, ey = eye_point
                    # Define a small region around the eye
                    ex -= 15
                    ey -= 15
                    ew = eh = 30

                    eye_frame = frame[ey:ey+eh, ex:ex+ew]
                    if eye_frame.shape[0] < 30 or eye_frame.shape[1] < 30:
                        continue  # Skip small detections

                    eye_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
                    eye_input = preprocess_eye(eye_gray)

                    pred = model.predict(eye_input, verbose=0)[0][0]
                    label = "Drowsy" if pred > 0.5 else "Alert"
                    color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)

                    if label == "Drowsy":
                        consecutive_drowsy += 1
                    else:
                        consecutive_drowsy = 0

                    label_text = f"{label} ({pred:.2f})"
                    print(f"[DEBUG] Prediction: {pred:.2f}, Label: {label}, Count: {consecutive_drowsy}")

                    cv2.putText(frame, label_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    break  # Use one eye for prediction

        if consecutive_drowsy >= threshold:
            # cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (50, 60),
            #             cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (50, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

            try:
                playsound("alarm.mp3", block=False)
            except:
                print("[WARNING] Couldn't play alarm sound.")

        cv2.imshow("MTCNN Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stopped.")
