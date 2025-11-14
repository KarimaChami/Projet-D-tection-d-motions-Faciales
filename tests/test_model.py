from tensorflow.keras.models import load_model
from fastapi.testclient import TestClient
import sys
import os
from app import app


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
client = TestClient(app)


def test_sauvegarde_model():
    model_path = "models/emotion_cnn_model.h5"
    model = load_model(model_path)

    assert model is not None, "Le modèle n'a pas été chargé correctement"
    assert model.input_shape == (None, 48, 48, 3) # vérifier que le modèle attend bien des images de taille 48×48 avec 3 canaux (RVB)
    assert model.output_shape == (None, 7) # vérifier que le modèle prédit bien 7 émotions différentes.
    print("Modèle chargé avec succès")



def test_prediction_format():
    # Ouvre une image de test en mode binaire
    with open("Notebook/img.jpg", "rb") as image_file:
        response = client.post("/predict_emotion", files={"file": image_file})
    assert response.status_code == 200, "La requête a échoué" # Vérifie que la requête a réussi
    # Vérifie la structure/format du résultat
    data = response.json()
    preds = data["predictions"]
    assert isinstance(preds, list), "'predictions' n'est pas une liste"
    for pred in preds:
        assert "emotion" in pred, "La clé 'emotion' est absente dans une prédiction"
        assert "confidence" in pred, "La clé 'confidence' est absente dans une prédiction"



