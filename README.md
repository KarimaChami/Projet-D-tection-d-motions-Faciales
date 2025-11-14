# Projet-D-tection-d-motions-FacialessnmabPq# 🫀 API de Prédiction du Risque Cardiovasculaire

## 📘 Contexte du Projet

Ce projet consiste à **détecter les visages** dans des images.puis à **prédire l’émotion** à l’aide de CNN (TensorFlow/Keras) et de **Haar Cascade (OpenCV)**,et intégrer ce modèle dans une API FastAPI connectée à une base de données **PostgreSQL**.


L’objectif final est de fournir une API simple et efficace qui :

* Enregistre les informations d’un patient dans une base de données.
* Retourne un **score de risque cardiovasculaire** à l’aide d’un modèle de Machine Learning.

---

## 👥 Travail individuel

### Rôles et Responsabilités

* **Base de données :**
Les prédictions sont stockées avec :
  * émotion (String)
  * confiance (Float)
  * timestamp auto-généré (UTC)

* **Développeur IA / Data :**

  * Nettoyage et préparation du dataset.
  * Entraînement et sauvegarde du modèle de Machine Learning.
  * Intégration du modèle dans la route `/predict_risk`.

### Collaboration GitHub

* Une branche `main`.
* Deux branches de fonctionnalités :

  * `feature/api`
  * `feature/ml`

---

## ⚙️ Technologies Utilisées

* **Backend :** FastAPI
* **Base de données :** PostgreSql
* **Validation :** Pydantic
* **Machine Learning :** Scikit-learn 
* **Tests :** Pytest + TestClient
* **Gestion de version :** GitHub

---

## 🧱 Structure du Projet

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
* Les données des patients bien enregistrées et listées

---

## 📊 Partie Machine Learning

### Étapes principales :

1. Chargement et nettoyage du dataset.
2. Transformation des variables numériques et catégorielles.
3. Séparation des données : `X` (features) et `y` (cible).
4. Création d’un **pipeline Scikit-learn** (préprocessing + modèle).
5. Entraînement du modèle.
6. Sauvegarde du modèle :

   ```python
   joblib.dump(model, "model.joblib")
   ```
7. (Bonus) Optimisation avec `GridSearchCV`.

---

## 🧾 Modalités Pédagogiques

* **Travail en binôme**
* **Durée :** 5 jours (du 27/10/2025 au 31/10/2025)
* **Livraison :**

  * API fonctionnelle
  * Modèle ML intégré
  * Tests unitaires validés
  * Documentation complète (README + Swagger)

---

## 📦 Livrables

1. Code source complet sur GitHub.
2. Base SQLite contenant les patients.
3. Modèle ML entraîné (`model.joblib`).
4. Tests unitaires (`pytest`).
5. Documentation :

   * `README.md`
   * Swagger `/docs`
6. Lien Jira du projet.

---

## 🏁 Critères de Performance

| Critère           | Description                                           |
| ----------------- | ----------------------------------------------------- |
| ✅ Fonctionnalité  | Routes `/patients` et `/predict_risk` opérationnelles |
| ✅ Précision       | Modèle bien entraîné et intégré                       |
| ✅ Qualité du Code | Structure claire, bien commentée                      |
| ✅ Collaboration   | Branches Git bien gérées et commits cohérents         |
| ✅ Documentation   | README complet + Swagger                              |
| ✅ Tests           | API validée par Pytest                                |

---

## 🧑‍💻 Auteur

* **Karima Chami** – Développeuse Backend

Projet réalisé dans le cadre du **brief API de Machine Learning** – Simplon Maghreb, 2025.
