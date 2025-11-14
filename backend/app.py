from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from backend.db import SessionLocal, engine, Base
from backend.schemas import Prediction
from Notebook.detect_and_predict import detect_and_predict

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
      # Save temporary image
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    # Call ML function
    predictions = detect_and_predict(temp_file,show=False)

    # Save each prediction in DB
    for pred in predictions:
        db_pred = Prediction(
            emotion=pred["emotion"],
            confidence=pred["confidence"]
        )
        db.add(db_pred)
    db.commit()

    return {"predictions": predictions, "num_faces": len(predictions)}


@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    preds = db.query(Prediction).order_by(Prediction.timestamp.desc()).all()
    return [{"emotion": p.emotion, "confidence": p.confidence, "timestamp": p.timestamp} for p in preds]
