
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_prediction_format():
    # Ouvre une image de test en mode binaire
    with open("Notebook/images/img.jpg", "rb") as image_file:
        response = client.post("/predict_emotion", files={"file": image_file})
    assert response.status_code == 200, "La requête a échoué" # Vérifie que la requête a réussi
    # Vérifie la structure/format du résultat
    data = response.json()
    preds = data["predictions"]
    assert isinstance(preds, list), "'predictions' n'est pas une liste"
    for pred in preds:
        assert "emotion" in pred, "La clé 'emotion' est absente dans une prédiction"
        assert "confidence" in pred, "La clé 'confidence' est absente dans une prédiction"

