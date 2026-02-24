import json
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



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

def add_brand_category(df: pd.DataFrame, brand_mean_map: dict, global_mean: float) -> pd.DataFrame:
    df = df.copy()
    df["brand"] = df["CarName"].apply(extract_brand)
    df["brand_avg_price"] = df["brand"].map(brand_mean_map).fillna(global_mean)
    df["brand_category"] = df["brand_avg_price"].apply(brand_category_from_mean)

    # drop like notebook
    for col in ["car_ID", "symboling", "CarName", "brand", "brand_avg_price"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def main():
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv"
    data = pd.read_csv(url)

    y = data["price"].copy()
    X_raw = data.drop(columns=["price"])

    # learn brand mean prices from TRAIN ONLY (still leakage-ish, but matches your logic)
    X_tmp = X_raw.copy()
    X_tmp["brand"] = X_tmp["CarName"].apply(extract_brand)
    brand_mean = pd.DataFrame({"brand": X_tmp["brand"], "price": y}).groupby("brand")["price"].mean().to_dict()
    global_mean = float(np.mean(list(brand_mean.values())))

    # create engineered dataset
    X = add_brand_category(X_raw, brand_mean, global_mean)

    categorical_features = [
        "fueltype", "aspiration", "carbody", "drivewheel",
        "brand_category", "enginetype", "cylindernumber"
    ]
    numeric_features = [
        "wheelbase", "curbweight", "enginesize", "boreratio", "horsepower",
        "carlength", "carwidth", "citympg", "highwaympg"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
        ],
        remainder="drop"
    )

    model = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # fit preprocesser on train, then model
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model.fit(X_train_t, y_train)

    preds = model.predict(X_test_t)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    # Save artifacts (NO custom class in joblib)
    joblib.dump(preprocessor, "preprocessor.joblib")
    joblib.dump(model, "linear_model.joblib")

    with open("brand_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"brand_mean": brand_mean, "global_mean": global_mean}, f)

    print("Saved: preprocessor.joblib, linear_model.joblib, brand_mapping.json")


if __name__ == "__main__":
    main()