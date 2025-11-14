import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Charger le modèle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "emotion_cnn_model.h5")

model = load_model(MODEL_PATH)
print("model.input_shape:", model.input_shape)
print("model.output_shape:", model.output_shape)
# Charger le classifieur HaarCascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprisd']

def detect_and_predict(image_path,show=False):
    # Lire l'image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    predictions = []

    for (x, y, w, h) in faces:
        # Extraire le visage en couleur
      face_img = img[y:y+h, x:x+w]                # Garde la couleur
      face_img = cv2.resize(face_img, (48, 48))   # Redimensionne
      face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convertit BGR -> RGB
      face_img = np.expand_dims(face_img, axis=0)    # shape = (1,48,48,3)
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
        })

    # Convertir BGR → RGB pour Matplotlib et afficher
    if show :
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(5, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

    return predictions


# Tester
if __name__ == "__main__":
    results = detect_and_predict("images/img.jpg",show=True)
    print(results)
