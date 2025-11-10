import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Charger le modèle
model = load_model("../models/emotion_cnn_model.h5")

# Charger le classifieur HaarCascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_and_predict(image_path):
    # Lire l'image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    predictions = []

    for (x, y, w, h) in faces:
        # Extraire le visage en couleur
        face_img = img[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype("float") / 255.0
        face_img = np.expand_dims(face_img, axis=0)  # shape = (1,48,48,3)

        # Prédire l'émotion
        pred = model.predict(face_img)
        emotion_idx = np.argmax(pred)
        emotion = emotion_labels[emotion_idx]
        confidence = np.max(pred)

        # Dessiner le rectangle et le label
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{emotion} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        predictions.append({
            "emotion": emotion,
            "confidence": float(confidence),
            "box": [int(x), int(y), int(w), int(h)]
        })

    # Convertir BGR → RGB pour Matplotlib et afficher
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

    return predictions


# Tester
results = detect_and_predict("img.jpg")  # <- récupérer la liste des prédictions
print(results)  # pour voir le dictionnaire avec emotion, confidence, box
