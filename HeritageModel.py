import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset_patrimonio_colombia_1200.csv")

def _load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df_heritage = _load_data()

def get_heritage_info():
    total_records = len(df_heritage)
    cities = df_heritage['Ciudad'].nunique()
    materials = df_heritage['Material_Principal'].nunique()
    deterioration_dist = df_heritage['Nivel_Deterioro_Actual'].value_counts().to_dict()
    
    return {
        "total_records": total_records,
        "cities": cities,
        "materials": materials,
        "deterioration_distribution": deterioration_dist,
        "avg_age": round(df_heritage['Edad_Estructura_Anios'].mean(), 2),
        "df_sample": df_heritage.head(10).to_dict('records')
    }

def predict_deterioration_risk(edad, material, precipitacion, humedad, pm10, distancia_via):
    """Simple risk simulation (can be improved later with a real trained model)"""
    risk_score = 0
    if edad > 200: risk_score += 30
    if precipitacion > 1500: risk_score += 25
    if humedad > 75: risk_score += 20
    if pm10 > 50: risk_score += 15
    if distancia_via < 50: risk_score += 10
    
    if risk_score > 70:
        return "Alto"
    elif risk_score > 40:
        return "Medio"
    else:
        return "Bajo"