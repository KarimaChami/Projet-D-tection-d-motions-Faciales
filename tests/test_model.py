from tensorflow.keras.models import load_model





def test_sauvegarde_model():
    model_path = "models/emotion_cnn_model.h5"
    model = load_model(model_path)

    assert model is not None, "Le modèle n'a pas été chargé correctement"
    assert model.input_shape == (None, 48, 48, 3) # vérifier que le modèle attend bien des images de taille 48×48 avec 3 canaux (RVB)
    assert model.output_shape == (None, 7) # vérifier que le modèle prédit bien 7 émotions différentes.
    print("Modèle chargé avec succès")

def test_prediction_format():
  pred = model.predict(sample_input)
  
