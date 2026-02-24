import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="Car Price Prediction API (No Pickle Custom Class)")

BASE_DIR = Path(__file__).resolve().parent

preprocessor = joblib.load(BASE_DIR / "preprocessor.joblib")
model = joblib.load(BASE_DIR / "linear_model.joblib")
with open(BASE_DIR / "brand_mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)

brand_mean = mapping["brand_mean"]
global_mean = float(mapping["global_mean"])

def extract_brand(carname: str) -> str:
    b = str(carname).split(" ")[0].lower()
    fixes = {
        "vw": "volkswagen",
        "vokswagen": "volkswagen",
        "maxda": "mazda",
        "porcshce": "porsche",
        "toyouta": "toyota",
    }
    return fixes.get(b, b)

def brand_category_from_mean(mean_price: float) -> str:
    if mean_price < 10000:
        return "Budget"
    elif mean_price < 20000:
        return "Mid_Range"
    return "Luxury"

def add_brand_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["brand"] = df["CarName"].apply(extract_brand)
    df["brand_avg_price"] = df["brand"].map(brand_mean).fillna(global_mean)
    df["brand_category"] = df["brand_avg_price"].apply(brand_category_from_mean)
    for col in ["car_ID", "symboling", "CarName", "brand", "brand_avg_price"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


class CarInput(BaseModel):
    CarName: str
    fueltype: str
    aspiration: str
    carbody: str
    drivewheel: str
    wheelbase: float
    curbweight: float
    enginetype: str
    cylindernumber: str
    enginesize: float
    boreratio: float
    horsepower: float
    carlength: float
    carwidth: float
    citympg: float
    highwaympg: float
    car_ID: int | None = None
    symboling: int | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(inp: CarInput):
    df_raw = pd.DataFrame([inp.model_dump()])
    df = add_brand_category(df_raw)

    X_t = preprocessor.transform(df)
    pred = model.predict(X_t)[0]
    return {"predicted_price": float(pred)}