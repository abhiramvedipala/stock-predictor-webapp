from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, symbol: str = Form(...)):
    try:
        data = yf.download(symbol, period="1y")
        data['Date'] = data.index
        data.reset_index(drop=True, inplace=True)

        data['Days'] = (data['Date'] - data['Date'].min()).dt.days
        X = data[['Days']]
        y = data['Close']

        model = LinearRegression()
        model.fit(X, y)

        future_days = 30
        last_day = data['Days'].iloc[-1]
        future_X = pd.DataFrame({'Days': [last_day + i for i in range(1, future_days + 1)]})
        future_preds = model.predict(future_X)

        future_dates = [data['Date'].max() + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
        predictions = zip(future_dates, future_preds)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "symbol": symbol.upper(),
            "predictions": list(predictions)
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "symbol": symbol,
            "error": f"Error fetching data: {e}"
        })
