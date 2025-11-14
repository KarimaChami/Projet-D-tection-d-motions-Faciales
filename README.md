# Projet-D-tection-d-motions-FacialessnmabPq# 🫀 API de Prédiction du Risque Cardiovasculaire

## 📘 Contexte du Projet

Ce projet est un prototype d'**API d'Intelligence Artificielle** conçu pour l'analyse émotionnelle à partir d'images faciales. L'objectif est de valider la faisabilité d'un futur produit SaaS capable de mesurer les réactions des utilisateurs lors de tests produits ou d'expériences UX.

L'API utilise une combinaison de **vision par ordinateur** (OpenCV/Haar Cascade) et d'**apprentissage profond** (CNN) pour :
1. **Détecter** un visage sur une image.
2. **Prédire** l'émotion correspondante.
3. **Enregistrer** l'historique des prédictions dans une base de données **PostgreSQL**.


## 📂 Structure du projet
```
📁 Projet-D-tection-d-motions-Faciales
│
├── backend/
│   ├── app.py        
│   ├── db.py                           
│   |── schemas.py             
│── models/
│   ├── emotion_cnn_model.h5      
├── Notebook/
│   ├── images/ 
│   ├── detect_and_predict.py       
├── test/
│   ├── __init__.py
│   ├── test_model.py
├── test/
│   ├──workflows/
│   │  ├──.github/
│   │     ├──test.yml                           
├── README.md
└── requirements.txt
```
## 👥 Travail individuel

### Rôles et Responsabilités

* **Entraînement et Modélisation du CNN :**
Le modèle a été développé dans le notebook training/emotion_cnn_training.ipynb on utilisant googleColab.
  * Architecture : CNN séquentiel avec Conv2D, MaxPooling2D, Flatten, Dense, et Dropout.
  * Perte : categorical_crossentropy.
  * Optimiseur : adam.
  * Prétraitement : Normalisation et Augmentation des données pour améliorer la performance.
  * Modèle Sauvegardé : training/models/emotion_model.h5.

* **Base de données :**
Les prédictions sont stockées avec :
  * émotion (String)
  * confiance (Float)
  * timestamp auto-généré (UTC)


---

## 🛠️ Stack Technique

* **Langage :** Python 3.x
* **API Web :** FastAPI
* **Modèle ML :** TensorFlow / Keras (CNN)
* **Vision par Ordinateur :** OpenCV (pour la détection faciale via Haar Cascade)
* **Base de Données :** PostgreSQL
* **ORM :** SQLAlchemy
* **Tests :** `pytest`
* **CI/CD :** GitHub Actions

---


## 🚀 Installation & Lancement

### 1. Cloner le projet

```bash
git clone https://github.com/KarimaChami/Projet-D-tection-d-motions-Faciales.git
cd Projet-D-tection-d-motions-Faciales
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source env/bin/activate       # macOS / Linux
env\Scripts\activate          # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer le serveur FastAPI

```bash
uvicorn backend.app:app --reload
```

### 5. Accéder à la documentation interactive

Ouvrir dans le navigateur :

```
http://127.0.0.1:8000/docs
```

---

## 📡 Endpoints Principaux

### ➕ POST /predict_emotion

Reçoit un fichier image via UploadFile, détecter le visage,Passe la région du visage au modèle CNN pour la prédiction puis Retourne l’émotion prédite et le score et Chaque prédiction est automatiquement insérée dans la base

**Exemple de requête :**
passer une image d'un person en colère
**Réponse :**

```json
{
  "predictions": [
    {
      "emotion": "Angry",
      "confidence": 0.5139312744140625
    }
  ],
  "num_faces": 1
}
```

### 📋 GET /history

Lister tous les prédictions enregistrées dans la base PostgreSQL.

---

## 🧪 Tests Unitaires
**test_sauvegarde_model() :**
    Vérifier que ton modèle est bien sauvegarde et peut etre recharge sans erreur
**test_sauvegarde_model() :**
    Vérification du format de la prédiction.
### Lancer les tests

```bash
pytest -v
```

### Vérifications principales

* Le statut `200` pour les routes
* Les predictions bien enregistrées et listées

---

---
### GitHub Actions

Un workflow est configuré pour exécuter automatiquement les tests unitaires à chaque push ou Pull Request, assurant une intégration continue stable.

---


## 📦 Livrables

1. Notebook d’entraînement du CNN.
2. Script detect_and_predict.py (OpenCV + CNN).
3. API FastAPI (main.py) avec routes /predict_emotion et /history.
4. Base PostgreSQL fonctionnelle.
5. Tests unitaires + workflow GitHub Actions.
6. Projet versionné sur GitHub.
6. Documentation :
   * `README.md`
   * `requirements.txt`
7. Lien Jira du projet.

---

## 🏁 Critères de Performance


| ✅ Précision et cohérence des prédictions. 
| ✅ Bon fonctionnement de la détection faciale.      
| ✅ Intégration stable du modèle dans l’API. 
| ✅ Historique PostgreSQL fonctionnel.   
| ✅ Jira                             |
                                       |
---


Projet réalisé par **Karima Chami**  dans le cadre d' **API d'Intelligence Artificielle** – Simplon Maghreb, 2025.
