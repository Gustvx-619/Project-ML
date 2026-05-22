# PatrimonioCO — Predicción de Deterioro del Patrimonio Arquitectónico Colombiano

Aplicación web Flask que implementa la metodología **CRISP-ML** para predecir el nivel de deterioro de Bienes de Interés Cultural (BIC) colombianos utilizando variables ambientales, climáticas y urbanísticas.

## Descripción

Este proyecto aplica Machine Learning sobre un dataset de **1.200 registros** de BIC de 10 ciudades colombianas para clasificar el riesgo de deterioro estructural en tres niveles: **Bajo**, **Medio** y **Alto**.

## Estructura del Repositorio

```
Project-ML/
├── app.py                   # Aplicación Flask principal
├── HeritageModel.py         # Lógica del modelo predictivo
├── requirements.txt         # Dependencias del proyecto
├── README.md                # Este archivo
├── data/
│   └── dataset_patrimonio_colombia_1200.csv
└── templates/
    ├── layout.html                  # Plantilla base
    ├── crisp_ml.html                # Metodología CRISP-ML
    ├── business_understanding.html  # Fase 1: Entendimiento del negocio
    ├── data_understanding.html      # Fase 2: Entendimiento de datos (EDA)
    ├── data_engineering.html        # Fase 2: Ingeniería de datos
    └── heritage_application.html   # Predicción interactiva
```

## Instalación y Ejecución Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/Gustvx-619/Project-ML.git
cd Project-ML

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
python app.py
# → http://127.0.0.1:5000
```

## Menús de la Aplicación

| Ruta | Descripción |
|------|-------------|
| `/` | Inicio — Metodología CRISP-ML |
| `/business-understanding` | Fase 1: Contexto, objetivos, preguntas, factibilidad, riesgos e impacto |
| `/data-understanding` | Fase 2: EDA, variables, estadísticas, correlaciones |
| `/data-engineering` | Fase 2: Limpieza, encoding, feature engineering, normalización |
| `/heritage-application` | Predicción interactiva del nivel de deterioro |

## Dataset

- **Registros:** 1.200 BIC colombianos
- **Ciudades:** 10 (Bogotá, Medellín, Cali, Cartagena, Santa Marta, Villa de Leyva, Barichara, Popayán, Tunja, Mompox)
- **Variables:** 11 columnas (5 numéricas, 4 categóricas, 1 objetivo, 2 identificadores)
- **Fuentes:** IDEAM, OpenStreetMap, Datos Abiertos Colombia

## Metodología CRISP-ML

1. **Business Understanding** — Contexto, objetivos y preguntas clave
2. **Data Understanding** — EDA, estadísticas, correlaciones
3. **Data Engineering** — Limpieza, encoding, feature engineering, normalización
4. **Model Engineering** — Selección de algoritmos, entrenamiento
5. **Model Evaluation** — Métricas, validación cruzada
6. **Deployment & Monitoring** — Despliegue en Render, monitoreo

## Despliegue

Aplicación desplegada en: https://project-ml-bc97.onrender.com

## Referencias

- KDnuggets (2025). *The Machine Learning Lifecycle*
- ML-Ops.org. *CRISP-ML(Q): The ML Lifecycle Process*
- Schmid, Wurst & Wirth (2020). *Towards CRISP-ML(Q)*
