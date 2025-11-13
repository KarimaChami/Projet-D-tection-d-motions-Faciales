from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from db import SessionLocal, engine, Base, Prediction


# Créer les tables si elles n'existent pas
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Charger le modèle
model = load_model("models/emotion_cnn_model.h5")
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Dépendance pour la session SQLAlchemy
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Charger Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.post("/predict_emotion")
async def predict_emotion(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Sauvegarder temporairement l'image
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    # Lire l'image et détecter les visages
    img = cv2.imread(temp_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    predictions = []

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48,48))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        pred = model.predict(face_img)
        emotion_idx = np.argmax(pred)
        emotion = emotion_labels[emotion_idx]
        confidence = float(np.max(pred))

        # Sauvegarder la prédiction dans la DB
        db_pred = Prediction(emotion=emotion, confidence=confidence)
        db.add(db_pred)
        db.commit()

        predictions.append({"emotion": emotion, "confidence": confidence})

    return {"predictions": predictions, "num_faces": len(faces)}

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    preds = db.query(Prediction).order_by(Prediction.timestamp.desc()).all()
    return [{"emotion": p.emotion, "confidence": p.confidence, "timestamp": p.timestamp} for p in preds]
