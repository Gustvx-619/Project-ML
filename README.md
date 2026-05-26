# PatrimonioCO — Deterioration Prediction of Colombian Architectural Heritage

A Flask web application implementing the **CRISP-ML** methodology to predict the structural deterioration risk level of Colombian Cultural Heritage Assets (Bienes de Interés Cultural — BIC) using open environmental, climatic, and urban variables.

## Description

This project applies Machine Learning on a dataset of **1,200 records** of BICs from 10 Colombian cities to classify structural deterioration risk into three levels: **Low**, **Medium**, and **High**. The application follows all 6 phases of the CRISP-ML methodology and is deployed publicly on Render.com.

## Repository Structure

```
Project-ML/
├── app.py                        # Main Flask application
├── HeritageModel.py              # ML model logic — training, prediction, metrics
├── requirements.txt              # Project dependencies
├── README.md                     # This file
├── data/
│   └── dataset_patrimonio_colombia_1200.csv   # 1,200 BIC records
├── models/                       # Auto-generated .pkl files (created on first run)
│   ├── rf_model.pkl              # Trained Random Forest
│   ├── lr_model.pkl              # Trained Logistic Regression
│   ├── knn_model.pkl             # Trained KNN
│   ├── scaler.pkl                # StandardScaler
│   ├── label_encoder.pkl         # LabelEncoder
│   └── feature_cols.pkl          # Feature column list
└── templates/
    ├── layout.html               # Base template with navbar
    ├── crisp_ml.html             # Home — CRISP-ML Methodology overview
    ├── business_understanding.html  # Phase 1: Business Understanding
    ├── data_understanding.html      # Phase 2: Data Understanding (EDA)
    ├── data_engineering.html        # Phase 2: Data Engineering
    ├── model_engineering.html       # Phase 3: Model Engineering
    ├── model_evaluation.html        # Phase 4: Model Evaluation
    └── heritage_application.html   # Phase 5: Prediction System
```

## Local Installation & Execution

```bash
# 1. Clone the repository
git clone https://github.com/Gustvx-619/Project-ML.git
cd Project-ML

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
# → http://127.0.0.1:5000
```

> **Note:** On first run, Flask automatically trains the 3 ML models and saves them as `.pkl` files in the `models/` directory. Subsequent runs load from disk — no retraining needed.

## Application Menus

| Route | Description |
|-------|-------------|
| `/` | Home — CRISP-ML Methodology overview with 6-phase timeline |
| `/business-understanding` | Phase 1: Problem context, objectives, key questions, feasibility, risks, expected impact |
| `/data-understanding` | Phase 2: EDA, variable analysis, statistics, distributions, correlations |
| `/data-engineering` | Phase 2: Cleaning, encoding, feature engineering, normalization |
| `/model-engineering` | Phase 3: 3 ML models, training code, hyperparameters, CV, confusion matrices |
| `/model-evaluation` | Phase 4: Metrics dashboard, ROC analysis, comparison table, best model selection |
| `/prediction-system` | Phase 5: Interactive prediction form with confidence % and probability bars |

## Dataset

- **Records:** 1,200 Colombian BICs
- **Cities:** 10 (Bogotá, Medellín, Cali, Cartagena, Santa Marta, Villa de Leyva, Barichara, Popayán, Tunja, Mompox)
- **Departments:** 8
- **Variables:** 11 columns (5 numerical, 4 categorical, 1 target, 2 identifiers)
- **Target classes:** Low: 713 (59.4%) · Medium: 469 (39.1%) · High: 18 (1.5%)
- **Sources:** IDEAM (climate), OpenStreetMap (road distances), Datos Abiertos Colombia (heritage registries)

## Machine Learning Models

Three models were trained and evaluated:

| Model | Accuracy | F1-Score | AUC-ROC | CV Score |
|-------|----------|----------|---------|----------|
| **Random Forest** ⭐ | **88.33%** | **87.76%** | **0.9571** | **89.58%** |
| Logistic Regression | 82.50% | 83.26% | 0.9150 | 82.71% |
| KNN | 77.50% | 76.83% | 0.8382 | 73.65% |

**Random Forest** was selected as the production model for its highest accuracy, best AUC-ROC, most stable cross-validation, and native class imbalance handling.

## Engineered Features

Two new features were created to capture synergistic risks:

- **Acid Rain Index** = `(Annual_Precipitation_mm × PM10_Avg_ugm3) / 1000`
- **Vibration Risk** = `Structure_Age_Years / Main_Road_Distance_m`

Together they account for **31.3%** of the model's total predictive power.

## CRISP-ML Methodology

1. **Business Understanding** — Problem context, objectives, feasibility, risks, expected impact
2. **Data Understanding** — EDA, variable analysis, statistics, correlations
3. **Data Engineering** — Cleaning, encoding, feature engineering, normalization
4. **Model Engineering** — Algorithm selection, training, hyperparameter configuration, cross-validation
5. **Model Evaluation** — Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrices, best model selection
6. **Deployment & Monitoring** — Deployed on Render.com, auto-train on first startup

## Deployment

Live application: **https://project-ml-bc97.onrender.com**

## Team

- María Esperanza Yara Martínez
- Sebastián Pedraza Bernal
- Gustavo Adolfo Pinto Marín

University of Cundinamarca · Systems and Computer Engineering · 6th Semester · May 2026

## References

- KDnuggets (2025). *The Machine Learning Lifecycle*
- ML-Ops.org. *CRISP-ML(Q): The ML Lifecycle Process*
- Schmid, Wurst & Wirth (2020). *Towards CRISP-ML(Q)*