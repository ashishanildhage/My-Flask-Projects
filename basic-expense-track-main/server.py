from flask import Flask,render_template,request
import db_sql as db

app = Flask(__name__)
@app.route("/contact")
def index():
    return render_template("index.html")


@app.route("/addCategory", methods=["GET","POST"])
def about():
    categories = db.getCategory()
    if request.method == "GET":
        return render_template("addCategory.html", categories=categories)
    category_name = request.form.get("category")
    msg = db.addCategory(category_name)
    return render_template("addCategory.html",msg=msg,categories=categories)

app.run(debug=True)