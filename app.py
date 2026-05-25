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
    model_info = HeritageModel.get_heritage_info()
    prediction = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = {
                "age":           float(request.form.get("age")),
                "material":      request.form.get("material"),
                "precipitation": float(request.form.get("precipitation")),
                "humidity":      float(request.form.get("humidity")),
                "pm10":          float(request.form.get("pm10")),
                "road_distance": float(request.form.get("road_distance")),
            }
            label, confidence = HeritageModel.predict_deterioration_risk(
                age=form_data["age"],
                material=form_data["material"],
                precipitation=form_data["precipitation"],
                humidity=form_data["humidity"],
                pm10=form_data["pm10"],
                road_distance=form_data["road_distance"]
            )
            prediction = (label, confidence)
        except Exception as e:
            print("Error:", e)
            prediction = "Error en los datos"

    return render_template("heritage_application.html",
                           model_info=model_info,
                           prediction=prediction,
                           form_data=form_data)


if __name__ == '__main__':
    app.run(debug=True)