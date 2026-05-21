from flask import Flask, render_template, request, redirect, url_for
import HeritageModel

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("crisp_ml.html")


@app.route('/crisp-ml')
def crisp_ml():
    return render_template("crisp_ml.html")


@app.route('/heritage-application', methods=["GET", "POST"])
def heritage_application():
    model_info = HeritageModel.get_heritage_info()
    prediction = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = {
                "edad": float(request.form.get("edad")),
                "material": request.form.get("material"),
                "precipitacion": float(request.form.get("precipitacion")),
                "humedad": float(request.form.get("humedad")),
                "pm10": float(request.form.get("pm10")),
                "distancia_via": float(request.form.get("distancia_via")),
            }
            prediction = HeritageModel.predict_deterioration_risk(
                form_data["edad"],
                form_data["material"],
                form_data["precipitacion"],
                form_data["humedad"],
                form_data["pm10"],
                form_data["distancia_via"]
            )
        except Exception as e:
            print("Error:", e)
            prediction = "Error en los datos"

    return render_template("heritage_application.html",
                           model_info=model_info,
                           prediction=prediction,
                           form_data=form_data)


if __name__ == '__main__':
    app.run(debug=True)