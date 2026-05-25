from flask import Flask, render_template, request, redirect, url_for
import HeritageModel

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("crisp_ml.html")


@app.route('/crisp-ml')
def crisp_ml():
    return render_template("crisp_ml.html")


@app.route('/business-understanding')
def business_understanding():
    return render_template("business_understanding.html")


@app.route('/data-understanding')
def data_understanding():
    return render_template("data_understanding.html")


@app.route('/data-engineering')
def data_engineering():
    return render_template("data_engineering.html")


@app.route('/model-engineering')
def model_engineering():
    return render_template("model_engineering.html")


@app.route('/model-evaluation')
def model_evaluation():
    return render_template("model_evaluation.html")


@app.route('/heritage-application', methods=["GET", "POST"])
def heritage_application():
    return redirect(url_for('prediction_system'))


@app.route('/prediction-system', methods=["GET", "POST"])
def prediction_system():
    prediction = None
    form_data  = {}
    error      = None

    if request.method == "POST":
        try:
            required = ["age", "material", "precipitation", "humidity", "pm10", "road_distance"]
            for field in required:
                if not request.form.get(field):
                    raise ValueError(f"Field '{field}' is required")

            form_data = {
                "age":           float(request.form.get("age")),
                "material":      request.form.get("material"),
                "precipitation": float(request.form.get("precipitation")),
                "humidity":      float(request.form.get("humidity")),
                "pm10":          float(request.form.get("pm10")),
                "road_distance": float(request.form.get("road_distance")),
            }

            if not (1 <= form_data["age"] <= 500):
                raise ValueError("Structure age must be between 1 and 500 years")
            if not (0 <= form_data["humidity"] <= 100):
                raise ValueError("Humidity must be between 0 and 100%")
            if form_data["road_distance"] <= 0:
                raise ValueError("Road distance must be greater than 0")

            label, confidence, probabilities = HeritageModel.predict_deterioration_risk(
                age=form_data["age"],
                material=form_data["material"],
                precipitation=form_data["precipitation"],
                humidity=form_data["humidity"],
                pm10=form_data["pm10"],
                road_distance=form_data["road_distance"]
            )
            prediction = (label, confidence, probabilities)

        except ValueError as ve:
            error = str(ve)
        except Exception as e:
            error = f"Unexpected error: {str(e)}"

    return render_template("heritage_application.html",
                           prediction=prediction,
                           form_data=form_data,
                           error=error)


if __name__ == '__main__':
    app.run(debug=True)