import pandas as pd
import numpy as np
import os, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset_patrimonio_colombia_1200.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RF_PATH     = os.path.join(MODELS_DIR, "rf_model.pkl")
LR_PATH     = os.path.join(MODELS_DIR, "lr_model.pkl")
KNN_PATH    = os.path.join(MODELS_DIR, "knn_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LE_PATH     = os.path.join(MODELS_DIR, "label_encoder.pkl")
FEAT_PATH   = os.path.join(MODELS_DIR, "feature_cols.pkl")

RANDOM_SEED = 42

# ─── Feature columns (set after training) ─────────────────────────────────────
_feature_cols = None

# ─── Internal helpers ─────────────────────────────────────────────────────────
def _load_raw():
    return pd.read_csv(DATA_PATH)


def _build_features(raw_df):
    """Apply all feature-engineering steps and return X, y, feature list."""
    df = raw_df.copy()

    # Cleaning: PM10 cannot be negative
    df['PM10_Avg_ugm3'] = df['PM10_Avg_ugm3'].clip(lower=0)

    # Feature engineering: synergistic risk variables
    df['Acid_Rain_Index'] = (
        df['Annual_Precipitation_mm'] * df['PM10_Avg_ugm3']
    ) / 1000

    df['Vibration_Risk'] = (
        df['Structure_Age_Years'] / df['Main_Road_Distance_m']
    ).round(3)

    # Encoding: target variable → numeric labels
    label_encoder = LabelEncoder()
    df['target'] = label_encoder.fit_transform(df['Current_Deterioration_Level'])

    # Encoding: categorical features → one-hot binary columns
    df_encoded = pd.get_dummies(
        df,
        columns=['Main_Material', 'City'],
        prefix=['mat', 'city']
    )

    # Feature selection
    numerical_features = [
        'Structure_Age_Years',
        'Annual_Precipitation_mm',
        'Avg_Annual_Humidity_pct',
        'PM10_Avg_ugm3',
        'Main_Road_Distance_m',
        'Acid_Rain_Index',
        'Vibration_Risk'
    ]
    categorical_features = [
        col for col in df_encoded.columns
        if col.startswith('mat_') or col.startswith('city_')
    ]
    all_features = numerical_features + categorical_features

    X = df_encoded[all_features]
    y = df_encoded['target']
    return X, y, all_features, label_encoder


def _train_and_save():
    """Train the 3 models and persist them to disk."""
    global _feature_cols

    raw_df = _load_raw()
    X, y, all_features, label_encoder = _build_features(raw_df)
    _feature_cols = all_features

    # ── Dataset split: 80% training / 20% testing ─────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y           # preserves Low/Medium/High ratio in both sets
    )

    # ── Standard scaler for Logistic Regression and KNN ───────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Model 1: Random Forest ─────────────────────────────────────────────────
    # Hyperparameters:
    #   n_estimators=100  → 100 decision trees in the forest
    #   max_depth=10      → each tree limited to 10 levels (avoids overfitting)
    #   class_weight='balanced' → compensates for class imbalance (High=1.5%)
    random_forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    random_forest.fit(X_train, y_train)

    # ── Model 2: Logistic Regression ───────────────────────────────────────────
    # Hyperparameters:
    #   C=1.0         → regularization strength (1.0 = balanced)
    #   max_iter=1000 → max iterations for convergence
    #   class_weight='balanced' → handles imbalanced classes
    logistic_regression = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    logistic_regression.fit(X_train_scaled, y_train)

    # ── Model 3: K-Nearest Neighbors ───────────────────────────────────────────
    # Hyperparameters:
    #   n_neighbors=7      → use 7 closest neighbors to vote
    #   metric='euclidean' → Euclidean distance between feature vectors
    knn_classifier = KNeighborsClassifier(
        n_neighbors=7,
        metric='euclidean'
    )
    knn_classifier.fit(X_train_scaled, y_train)

    # ── Generate predictions ───────────────────────────────────────────────────
    rf_predictions  = random_forest.predict(X_test)
    lr_predictions  = logistic_regression.predict(X_test_scaled)
    knn_predictions = knn_classifier.predict(X_test_scaled)

    # ── Cross-validation scores ────────────────────────────────────────────────
    rf_cv_score  = cross_val_score(random_forest,        X_train,        y_train, cv=5, scoring='accuracy').mean()
    lr_cv_score  = cross_val_score(logistic_regression,  X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    knn_cv_score = cross_val_score(knn_classifier,       X_train_scaled, y_train, cv=5, scoring='accuracy').mean()

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = {
        "Random Forest": {
            "accuracy":    round(accuracy_score(y_test, rf_predictions),  4),
            "f1":          round(f1_score(y_test, rf_predictions,  average='weighted'), 4),
            "cv_accuracy": round(rf_cv_score,  4),
            "cm":          confusion_matrix(y_test, rf_predictions).tolist(),
            "report":      classification_report(
                               y_test, rf_predictions,
                               target_names=label_encoder.classes_,
                               output_dict=True),
            "hyperparams": {
                "n_estimators": 100,
                "max_depth": 10,
                "class_weight": "balanced",
                "random_state": RANDOM_SEED
            },
            "feature_importance": dict(
                zip(all_features,
                    [round(v, 4) for v in random_forest.feature_importances_])
            )
        },
        "Logistic Regression": {
            "accuracy":    round(accuracy_score(y_test, lr_predictions),  4),
            "f1":          round(f1_score(y_test, lr_predictions,  average='weighted'), 4),
            "cv_accuracy": round(lr_cv_score,  4),
            "cm":          confusion_matrix(y_test, lr_predictions).tolist(),
            "report":      classification_report(
                               y_test, lr_predictions,
                               target_names=label_encoder.classes_,
                               output_dict=True),
            "hyperparams": {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": RANDOM_SEED
            }
        },
        "KNN": {
            "accuracy":    round(accuracy_score(y_test, knn_predictions), 4),
            "f1":          round(f1_score(y_test, knn_predictions, average='weighted'), 4),
            "cv_accuracy": round(knn_cv_score, 4),
            "cm":          confusion_matrix(y_test, knn_predictions).tolist(),
            "report":      classification_report(
                               y_test, knn_predictions,
                               target_names=label_encoder.classes_,
                               output_dict=True),
            "hyperparams": {
                "n_neighbors": 7,
                "metric": "euclidean"
            }
        }
    }

    # ── Persist all artifacts to disk ──────────────────────────────────────────
    joblib.dump(random_forest,       RF_PATH)
    joblib.dump(logistic_regression, LR_PATH)
    joblib.dump(knn_classifier,      KNN_PATH)
    joblib.dump(scaler,              SCALER_PATH)
    joblib.dump(label_encoder,       LE_PATH)
    joblib.dump(all_features,        FEAT_PATH)

    return metrics, X_train, X_test, y_train, y_test, label_encoder, all_features


# ─── Public API ───────────────────────────────────────────────────────────────
def get_heritage_info():
    """Return summary statistics from the dataset."""
    df = _load_raw()
    return {
        "total_records":              len(df),
        "cities":                     df['City'].nunique(),
        "materials":                  df['Main_Material'].nunique(),
        "deterioration_distribution": df['Current_Deterioration_Level'].value_counts().to_dict(),
        "avg_age":                    round(df['Structure_Age_Years'].mean(), 2),
        "df_sample":                  df.head(10).to_dict('records')
    }


def get_model_metrics():
    """Train the 3 models and return all performance metrics."""
    metrics, *_ = _train_and_save()
    return metrics


def predict_deterioration_risk(age, material, precipitation,
                                humidity, pm10, road_distance):
    """
    Predict deterioration risk using the best model (Random Forest).

    Parameters:
        age            (float) : Structure age in years
        material       (str)   : Main construction material
        precipitation  (float) : Annual precipitation in mm
        humidity       (float) : Average annual relative humidity (%)
        pm10           (float) : PM10 particulate matter (μg/m³)
        road_distance  (float) : Distance to main road in meters

    Returns:
        (label, confidence) : Predicted class and confidence percentage
    """
    # Train models if .pkl files don't exist yet
    if not os.path.exists(RF_PATH):
        _train_and_save()

    # Load trained artifacts
    trained_rf      = joblib.load(RF_PATH)
    label_encoder   = joblib.load(LE_PATH)
    feature_columns = joblib.load(FEAT_PATH)

    # Apply same feature engineering as training pipeline
    pm10_clean     = max(pm10, 0)
    acid_rain      = (precipitation * pm10_clean) / 1000
    vibration_risk = round(age / road_distance, 3) if road_distance > 0 else 0

    # Build input row with all feature columns set to 0 by default
    input_row = {col: 0 for col in feature_columns}
    input_row['Structure_Age_Years']     = age
    input_row['Annual_Precipitation_mm'] = precipitation
    input_row['Avg_Annual_Humidity_pct'] = humidity
    input_row['PM10_Avg_ugm3']           = pm10_clean
    input_row['Main_Road_Distance_m']    = road_distance
    input_row['Acid_Rain_Index']         = acid_rain
    input_row['Vibration_Risk']          = vibration_risk

    # One-hot encode the material
    material_key = f'mat_{material}'
    if material_key in input_row:
        input_row[material_key] = 1

    # Predict
    X_input          = pd.DataFrame([input_row])[feature_columns]
    predicted_index  = trained_rf.predict(X_input)[0]
    probabilities    = trained_rf.predict_proba(X_input)[0]

    predicted_label  = label_encoder.inverse_transform([predicted_index])[0]
    confidence       = round(float(max(probabilities)) * 100, 1)

    return predicted_label, confidence