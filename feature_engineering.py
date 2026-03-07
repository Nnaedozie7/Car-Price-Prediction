import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, budget_threshold=10000, mid_threshold=20000):
        self.budget_threshold = budget_threshold
        self.mid_threshold = mid_threshold
        self.brand_mean_price_ = None
        self.feature_columns_ = [
            "fueltype", "aspiration", "carbody", "drivewheel", "wheelbase",
            "brand_category", "curbweight", "enginetype", "cylindernumber",
            "enginesize", "boreratio", "horsepower", "carlength", "carwidth",
            "citympg", "highwaympg",
        ]

    @staticmethod
    def _extract_brand(df: pd.DataFrame) -> pd.Series:
        brand = df["CarName"].astype(str).str.split(" ").str.get(0).str.lower()
        brand = brand.replace(
            {
                "vw": "volkswagen",
                "vokswagen": "volkswagen",
                "maxda": "mazda",
                "porcshce": "porsche",
                "toyouta": "toyota",
            }
        )
        return brand

    def fit(self, X, y=None):
        X = X.copy()
        if y is None:
            raise ValueError("fit requires y (price).")
        X["brand"] = self._extract_brand(X)
        brand_means = (
            pd.DataFrame({"brand": X["brand"], "price": y})
            .groupby("brand")["price"]
            .mean()
        )
        self.brand_mean_price_ = brand_means.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        X["brand"] = self._extract_brand(X)
        global_mean = float(np.mean(list(self.brand_mean_price_.values())))
        X["brand_avg_price"] = X["brand"].map(self.brand_mean_price_).fillna(global_mean)

        def to_category(v: float) -> str:
            if v < self.budget_threshold:
                return "Budget"
            elif v < self.mid_threshold:
                return "Mid_Range"
            return "Luxury"

        X["brand_category"] = X["brand_avg_price"].apply(to_category)

        for col in ["car_ID", "symboling", "CarName", "brand", "brand_avg_price"]:
            if col in X.columns:
                X = X.drop(columns=[col])

        missing = [c for c in self.feature_columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required features after transform: {missing}")

        return X[self.feature_columns_]
