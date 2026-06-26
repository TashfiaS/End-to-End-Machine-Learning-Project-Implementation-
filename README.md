# End-to-End Machine Learning Project — US Visa Approval Prediction

Predicts whether a US visa application will be **Certified** or **Denied** based on applicant and employer details.  
Built with a modular ML pipeline: data ingestion from MongoDB → validation → transformation → model training → FastAPI web app.

**Live Demo:** https://usvisa-predictor.onrender.com

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10 |
| ML | scikit-learn, XGBoost, CatBoost |
| Imbalanced Data | imbalanced-learn (SMOTEENN) |
| Data Drift | scipy (KS-test + chi-square) |
| Web App | FastAPI, Jinja2, Uvicorn |
| Database | MongoDB Atlas |
| Serialization | dill |
| Config | PyYAML |

---

## Dataset

- **Source:** [EasyVisa Dataset — Kaggle](https://www.kaggle.com/datasets/moro23/easyvisa-dataset)
- **Target column:** `case_status` (Certified / Denied)
- **Features:** continent, education, job experience, no. of employees, prevailing wage, region, and more

---

## Prerequisites

- Anaconda: https://www.anaconda.com/download
- Git: https://git-scm.com/downloads
- Visual Studio Code: https://code.visualstudio.com/download
- MongoDB Atlas account: https://www.mongodb.com/products/platform/atlas-database

---

## Clone the Repository

```bash
git clone https://github.com/TashfiaS/End-to-End-Machine-Learning-Project-Implementation-.git
cd End-to-End-Machine-Learning-Project-Implementation-
```

---

## Create Conda Environment

```bash
conda --no-plugins create -n usvisa python=3.10 -y --solver=classic
```
```bash
conda activate usvisa
```
```bash
pip install -r requirements.txt
```

---

## Environment Variable

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/"
```

---

## Pipeline Workflow

```
MongoDB
   └── Data Ingestion        → pulls data, splits train/test
         └── Data Validation    → schema check + drift detection
               └── Data Transformation  → feature engineering, encoding, SMOTEENN
                     └── Model Trainer       → GridSearchCV (RF, XGBoost, GradientBoosting)
                           └── saved_models/model.pkl
```

### Component order

```
1. constants
2. config_entity
3. artifact_entity
4. component
5. pipeline
6. app.py
```

---

## Run Training Pipeline

```bash
python -c "
from us_visa.pipeline.training_pipeline import TrainPipeline
TrainPipeline().run_pipeline()
"
```

Trained model is saved at `saved_models/model.pkl`.

---

## Run Web App

```bash
python app.py
```

Open `http://localhost:8080` in your browser.

- `GET /`        — prediction form
- `GET /train`   — retrain the model
- `POST /predict` — returns visa prediction

---

## Project Structure

```
us_visa/
├── components/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   └── model_trainer.py
├── pipeline/
│   ├── training_pipeline.py
│   └── prediction_pipeline.py
├── entity/
│   ├── config_entity.py
│   └── artifact_entity.py
├── cloud_storage/
│   └── aws_storage.py
├── ml/
│   └── model/
│       └── estimator.py
├── utils/
│   └── main_utils.py
├── configuration/
│   └── mongo_db_connection.py
├── constants/
│   └── __init__.py
├── logger/
│   └── __init__.py
└── exception/
    └── __init__.py

config/
├── schema.yaml
└── model.yaml

templates/
└── usvisa.html

saved_models/
└── model.pkl

app.py
Dockerfile
requirements.txt
```

---

## Model Performance

| Metric | Score |
|--------|-------|
| F1 Score | 0.887 |
| Precision | 0.905 |
| Recall | 0.870 |

Best model selected via 3-fold GridSearchCV across RandomForest, XGBoost, and GradientBoosting.

---

## Data Transformation Details

- `company_age` engineered from `yr_of_estab`
- `education_of_employee` encoded with OrdinalEncoder (High School → Doctorate)
- `continent`, `region_of_employment`, `unit_of_wage` encoded with OneHotEncoder
- Binary columns (`has_job_experience`, `requires_job_training`, `full_time_position`) mapped Y/N → 1/0
- Class imbalance handled with **SMOTEENN**
- Numerical features scaled with StandardScaler
