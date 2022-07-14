from flask import Flask, request, render_template, redirect
import database as db

app = Flask(__name__)

@app.route("/search")
def index():
    name = request.args.get("name", "world")
    # return f"<h1>Hello {name}!!</h1>"
    return render_template("index.html", name=name)

@app.route("/")
def main():
    return render_template("main.html")

@app.route("/car/add", methods=['GET','POST'])
def addCar():
    if request.method == "GET":
        status = request.args.get("status","")
        companies = db.getCompanies()
        cars = db.getCars()
        return render_template("addCar.html", companies = companies, cars=cars, status=status)
    else:
        name = request.form.get("name")
        model = request.form.get("model")
        color = request.form.get("color")
        launch = request.form.get("launch")
        company_id = request.form.get("company_id")
        status = db.addCar(name, model, color, launch, company_id)
        return redirect("/car/add?status="+status)


app.run(debug=True)