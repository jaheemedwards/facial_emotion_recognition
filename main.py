import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("model/facial_emotion_model.keras")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load OpenCV pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion(image):
    """
    Predict the emotion of a face image.
    Expects a color image (BGR from OpenCV).
    """
    # Resize to 96x96
    img_resized = cv2.resize(image, (96, 96))

    # Normalize
    img_normalized = img_resized / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    # Predict
    predictions = model.predict(img_batch)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

    return class_names[class_idx], confidence

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Predict emotion on the detected face
        label, conf = predict_emotion(face_img)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Facial Emotion Recognition", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
