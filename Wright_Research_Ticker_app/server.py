#Build a website from flask import Flask
from flask import Flask,render_template,request
import yfinance as yf
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('display.html')

@app.route("/", methods=["POST"]) 
def index_post():
    ticker = request.form['symbol']
    tData = yf.Ticker(ticker.upper())
    data = yf.download(tickers = ticker.upper(),auto_adjust = True,prepost = True)
    twoWeeksChange=((data[:-10:-1]['Close'][0]/data[:-10:-1]['Close'][-1])-1)*100
    output = [f"{tData.info['currentPrice']}",
        f"{round(((tData.info['currentPrice']/tData.info['previousClose'])-1)*100,2)}",
        f"{tData.info['open']}",
        f"{tData.info['dayHigh']}",
        f"{tData.info['dayLow']}",
        f"{tData.info['previousClose']}",
        f"{round(twoWeeksChange,2)}"]
    return render_template("display.html",output=output)

app.run(port=5000,debug=True)