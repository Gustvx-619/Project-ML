import joblib
import pandas as pd

# ==========================================
# 1. LOAD MODELS AND EXTRACT FEATURES
# ==========================================
try:
    print("Loading models...")
    rf_model = joblib.load('models/rf_model.pkl')
    lr_model = joblib.load('models/lr_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # MAGIC TRICK: Extract the exact columns the model was trained on
    expected_columns = list(rf_model.feature_names_in_)
    print("✅ Models loaded successfully.\n")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# Extract categories automatically from the trained columns
trained_materials = [col.replace('mat_', '') for col in expected_columns if col.startswith('mat_')]
trained_cities = [col.replace('city_', '') for col in expected_columns if col.startswith('city_')]

# ==========================================
# 2. DATA PREPARATION FUNCTION
# ==========================================
def prepare_data(age, material, city, precipitation, humidity, pm10, road_distance):
    # 1. Engineered Features
    vibration_risk = 1 / (road_distance + 1)
    acid_rain_index = precipitation * (pm10 / 100)

    # 2. Create a DataFrame initialized with 0s, using the EXACT columns the model expects
    df = pd.DataFrame(0, index=[0], columns=expected_columns)

    # 3. Assign numerical features 
    df['Structure_Age_Years'] = age
    df['Annual_Precipitation_mm'] = precipitation
    df['Avg_Annual_Humidity_pct'] = humidity
    df['PM10_Avg_ugm3'] = pm10
    df['Main_Road_Distance_m'] = road_distance
    df['Vibration_Risk'] = vibration_risk
    df['Acid_Rain_Index'] = acid_rain_index

    # 4. Activate categorical features (One-Hot Encoding)
    if f'mat_{material}' in expected_columns:
        df[f'mat_{material}'] = 1
        
    if f'city_{city}' in expected_columns:
        df[f'city_{city}'] = 1

    return df

# ==========================================
# 3. HELPER TO FORMAT PREDICTIONS
# ==========================================
def format_prediction(pred):
    """
    Translates the numerical prediction back into readable text.
    """
    if isinstance(pred, str):
        return pred.upper()
        
    risk_map = {
        0: 'LOW',
        1: 'MEDIUM',
        2: 'HIGH'
    }
    return risk_map.get(int(pred), f"Category {pred}")

# ==========================================
# 4. INTERACTIVE CONSOLE INTERFACE
# ==========================================
def main():
    print("="*50)
    print(" DETERIORATION PREDICTION SIMULATOR (CONSOLE)")
    print("="*50)
    
    while True:
        try:
            print("\n--- Enter Heritage Asset Data ---")
            
            # --- DYNAMIC CITY INPUT ---
            print("\nAvailable Cities (Auto-detected):")
            for i, c in enumerate(trained_cities, 1):
                print(f"{i}. {c.replace('_', ' ')}")
            city_idx = int(input(f"Select city (1-{len(trained_cities)}): ")) - 1
            city = trained_cities[city_idx] if 0 <= city_idx < len(trained_cities) else trained_cities[0]

            # --- DYNAMIC MATERIAL INPUT ---
            print("\nAvailable Materials (Auto-detected):")
            for i, m in enumerate(trained_materials, 1):
                print(f"{i}. {m.replace('_', ' ')}")
            mat_idx = int(input(f"Select material (1-{len(trained_materials)}): ")) - 1
            material = trained_materials[mat_idx] if 0 <= mat_idx < len(trained_materials) else trained_materials[0]

            # --- NUMERICAL INPUTS ---
            age = float(input("\nStructure Age (years): "))
            precipitation = float(input("Annual Precipitation (mm): "))
            humidity = float(input("Relative Humidity (%): "))
            pm10 = float(input("PM10 Level (µg/m³): "))
            road_distance = float(input("Distance to main road (meters): "))
            
            # --- MODEL SELECTION MENU ---
            print("\n--- Select Prediction Model ---")
            print("1. Random Forest (Main/Best Model)")
            print("2. Logistic Regression (Baseline)")
            print("3. K-Nearest Neighbors (KNN)")
            print("4. Compare ALL models")
            model_choice = input("Select an option (1-4): ").strip()
            if model_choice not in ['1', '2', '3', '4']:
                model_choice = '1' # Default to Random Forest if invalid input
            
        except (ValueError, IndexError):
            print("\n❌ Input Error: Please enter valid numbers corresponding to the options.")
            continue

        try:
            # Prepare data
            df_input = prepare_data(age, material, city, precipitation, humidity, pm10, road_distance)
            
            # Scale data (Required for LR and KNN)
            input_scaled = scaler.transform(df_input)

            # Predictions
            print("\n" + "="*40)
            print(" PREDICTION RESULTS")
            print("="*40)
            
            # --- Random Forest ---
            if model_choice in ['1', '4']:
                pred_rf = rf_model.predict(df_input)[0]
                prob_rf = max(rf_model.predict_proba(df_input)[0]) * 100
                print(f" Random Forest:")
                print(f"   ➤ Risk Level: {format_prediction(pred_rf)}")
                print(f"   ➤ Confidence: {prob_rf:.1f}%")

            # --- Logistic Regression ---
            if model_choice in ['2', '4']:
                pred_lr = lr_model.predict(input_scaled)[0]
                prob_lr = max(lr_model.predict_proba(input_scaled)[0]) * 100
                print(f"\n Logistic Regression:")
                print(f"   ➤ Risk Level: {format_prediction(pred_lr)}")
                print(f"   ➤ Confidence: {prob_lr:.1f}%")

            # --- KNN ---
            if model_choice in ['3', '4']:
                pred_knn = knn_model.predict(input_scaled)[0]
                prob_knn = max(knn_model.predict_proba(input_scaled)[0]) * 100
                print(f"\n K-Nearest Neighbors:")
                print(f"   ➤ Risk Level: {format_prediction(pred_knn)}")
                print(f"   ➤ Confidence: {prob_knn:.1f}%")
                
            print("="*40)

        except Exception as e:
            print(f"\n❌ Model Error: {e}")

        cont = input("\nWould you like to test another record? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting simulator. See you next time!")
            break

if __name__ == "__main__":
    main()