# # DSML Group Project — Olist E-Commerce
# 
# ## Project title
# **Predicting Delivery Risk and Customer Dissatisfaction in E-Commerce Using Machine Learning**
# 
# Objectives:
# 1. Predict actual delivery time using regression.
# 2. Predict late-delivery risk using classification.
# 3. Segment sellers using clustering.
# 4. Forecast marketplace revenue using Holt-Winters.
# 5. Examine delivery performance and customer dissatisfaction.

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    silhouette_score
)

RANDOM_STATE = 42
pd.set_option("display.max_columns", 100)

# ## 1. Load all datasets

candidate_dirs = [Path("data"), Path("../data"), Path(".")]
DATA_DIR = next((d for d in candidate_dirs if (d / "olist_orders_dataset.csv").exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError("Place all Olist CSV files in data/, ../data/, or the current working directory.")

def first_match(pattern):
    m = sorted(DATA_DIR.glob(pattern))
    if not m:
        raise FileNotFoundError(pattern)
    return m[0]

orders = pd.read_csv(first_match("olist_orders_dataset.csv"))
items = pd.read_csv(first_match("olist_order_items_dataset.csv"))
payments = pd.read_csv(first_match("olist_order_payments_dataset.csv"))
reviews = pd.read_csv(first_match("olist_order_reviews_dataset.csv"))
products = pd.read_csv(first_match("olist_products_dataset.csv"))
sellers = pd.read_csv(first_match("olist_sellers_dataset.csv"))
customers = pd.read_csv(first_match("olist_customers_dataset*.csv"))
geo = pd.read_csv(first_match("olist_geolocation_dataset.csv"))
translation = pd.read_csv(first_match("product_category_name_translation.csv"))

tables = {
    "orders": orders, "items": items, "payments": payments, "reviews": reviews,
    "products": products, "sellers": sellers, "customers": customers,
    "geolocation": geo, "translation": translation
}

for name, df in tables.items():
    print(name, df.shape)

# ## 2. Data audit, missing values and duplicates

audit = []
for name, df in tables.items():
    audit.append({
        "table": name,
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": df.duplicated().sum(),
        "missing_cells": df.isna().sum().sum()
    })
display(pd.DataFrame(audit))

for name, df in tables.items():
    miss = df.isna().mean().mul(100)
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        print("\n", name)
        display(miss.to_frame("missing_pct"))

# ### Missingness treatment
# - Delivery dates are **not imputed** because missing delivery dates are related to order status.
# - Product numeric attributes are median-imputed in ML pipelines.
# - Categorical predictors use most-frequent imputation.
# - Review score is not imputed because missing reviews may be informative.

# ## 3. Cleaning and date conversion

for c in [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date"
]:
    orders[c] = pd.to_datetime(orders[c], errors="coerce")

items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")
reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce")
reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"], errors="coerce")

orders = orders.drop_duplicates()
items = items.drop_duplicates()
payments = payments.drop_duplicates()
reviews = reviews.drop_duplicates()
products = products.drop_duplicates()
sellers = sellers.drop_duplicates()
customers = customers.drop_duplicates()
geo = geo.drop_duplicates()

products = products.merge(translation, on="product_category_name", how="left")
products["product_category_name_english"] = (
    products["product_category_name_english"]
    .fillna(products["product_category_name"])
    .fillna("unknown")
)
products["product_volume_cm3"] = (
    products["product_length_cm"] *
    products["product_height_cm"] *
    products["product_width_cm"]
)

# ## 4. EDA and sampling

orders["order_status"].value_counts().plot(kind="bar", figsize=(8,4))
plt.title("Order Status Distribution")
plt.tight_layout()
plt.show()

reviews["review_score"].value_counts().sort_index().plot(kind="bar", figsize=(7,4))
plt.title("Review Score Distribution")
plt.tight_layout()
plt.show()

customers["customer_state"].value_counts().head(15).plot(kind="bar", figsize=(9,4))
plt.title("Top Customer States")
plt.tight_layout()
plt.show()

sample_10 = orders.sample(frac=0.10, random_state=RANDOM_STATE)
full_share = orders["order_status"].value_counts(normalize=True).rename("full_share")
sample_share = sample_10["order_status"].value_counts(normalize=True).rename("sample_share")
display(pd.concat([full_share, sample_share], axis=1).fillna(0))

# ## 5. Geolocation aggregation and distance

geo_zip = (
    geo.groupby("geolocation_zip_code_prefix", as_index=False)
       .agg(lat=("geolocation_lat", "median"),
            lng=("geolocation_lng", "median"))
)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ## 6. Build the order-level modelling table

def mode_or_nan(s):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else np.nan

item_detail = (
    items
    .merge(products, on="product_id", how="left")
    .merge(sellers, on="seller_id", how="left")
)

item_agg = (
    item_detail.groupby("order_id", as_index=False)
    .agg(
        item_count=("order_item_id", "count"),
        unique_products=("product_id", "nunique"),
        seller_count=("seller_id", "nunique"),
        order_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        avg_item_price=("price", "mean"),
        total_weight_g=("product_weight_g", "sum"),
        avg_weight_g=("product_weight_g", "mean"),
        mean_product_volume_cm3=("product_volume_cm3", "mean"),
        product_category=("product_category_name_english", mode_or_nan),
        seller_id_primary=("seller_id", mode_or_nan),
        seller_state_primary=("seller_state", mode_or_nan),
        seller_zip_primary=("seller_zip_code_prefix", mode_or_nan)
    )
)

payment_agg = (
    payments.groupby("order_id", as_index=False)
    .agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_types=("payment_type", "nunique"),
        payment_type_primary=("payment_type", mode_or_nan)
    )
)

review_agg = (
    reviews.groupby("order_id", as_index=False)
    .agg(review_score=("review_score", "mean"))
)

model = (
    orders
    .merge(customers, on="customer_id", how="left")
    .merge(item_agg, on="order_id", how="left")
    .merge(payment_agg, on="order_id", how="left")
    .merge(review_agg, on="order_id", how="left")
)

model = model.merge(
    geo_zip.rename(columns={
        "geolocation_zip_code_prefix": "customer_zip_code_prefix",
        "lat": "customer_lat", "lng": "customer_lng"
    }),
    on="customer_zip_code_prefix", how="left"
)

model = model.merge(
    geo_zip.rename(columns={
        "geolocation_zip_code_prefix": "seller_zip_primary",
        "lat": "seller_lat", "lng": "seller_lng"
    }),
    on="seller_zip_primary", how="left"
)

model["distance_km"] = haversine_km(
    model["customer_lat"], model["customer_lng"],
    model["seller_lat"], model["seller_lng"]
)

print("Order-level table:", model.shape)
display(model.head())

# ## 7. Feature engineering and targets

model["delivery_days"] = (
    model["order_delivered_customer_date"] -
    model["order_purchase_timestamp"]
).dt.total_seconds() / 86400

model["promised_days"] = (
    model["order_estimated_delivery_date"] -
    model["order_purchase_timestamp"]
).dt.total_seconds() / 86400

model["late_delivery"] = np.where(
    model["order_delivered_customer_date"].notna(),
    (model["order_delivered_customer_date"] >
     model["order_estimated_delivery_date"]).astype(int),
    np.nan
)

model["delivery_delay_days"] = (
    model["order_delivered_customer_date"] -
    model["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

model["dissatisfied"] = np.where(
    model["review_score"].notna(),
    (model["review_score"] <= 2).astype(int),
    np.nan
)

model["purchase_year"] = model["order_purchase_timestamp"].dt.year
model["purchase_month"] = model["order_purchase_timestamp"].dt.month
model["purchase_dow"] = model["order_purchase_timestamp"].dt.dayofweek
model["purchase_hour"] = model["order_purchase_timestamp"].dt.hour
model["is_weekend"] = model["purchase_dow"].isin([5,6]).astype(int)

model["freight_ratio"] = model["freight_value"] / model["order_value"].replace(0, np.nan)
model["same_state"] = (model["customer_state"] == model["seller_state_primary"]).astype(int)

model.loc[model["delivery_days"] < 0, "delivery_days"] = np.nan

display(model[[
    "delivery_days", "promised_days", "late_delivery",
    "delivery_delay_days", "review_score", "dissatisfied",
    "distance_km", "freight_ratio"
]].describe())

# ## 8. Delivery performance and dissatisfaction

review_delivery = model[
    model["review_score"].notna() &
    model["late_delivery"].notna()
].copy()

summary = (
    review_delivery.groupby("late_delivery")
    .agg(
        orders=("order_id", "count"),
        avg_review_score=("review_score", "mean"),
        dissatisfaction_rate=("dissatisfied", "mean"),
        avg_delivery_days=("delivery_days", "mean")
    )
)
display(summary)

display(pd.crosstab(
    review_delivery["review_score"].round().astype(int),
    review_delivery["late_delivery"],
    normalize="columns"
))

# ## 9. Predictors and preprocessing pipeline

NUMERIC_FEATURES = [
    "item_count", "unique_products", "seller_count",
    "order_value", "freight_value", "avg_item_price",
    "total_weight_g", "avg_weight_g", "mean_product_volume_cm3",
    "payment_value", "payment_installments", "payment_types",
    "promised_days", "purchase_month", "purchase_dow",
    "purchase_hour", "is_weekend", "freight_ratio",
    "distance_km", "same_state"
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state_primary",
    "payment_type_primary",
    "product_category"
]

NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in model.columns]
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c in model.columns]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, NUMERIC_FEATURES),
    ("cat", categorical_pipe, CATEGORICAL_FEATURES)
])

print(NUMERIC_FEATURES)
print(CATEGORICAL_FEATURES)

# ## 10. Feature selection — Filter, Wrapper and Embedded

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, RFE
from sklearn.linear_model import LinearRegression, Lasso

fs_df = model[
    NUMERIC_FEATURES + ["delivery_days", "late_delivery"]
].copy()

fs_df[NUMERIC_FEATURES] = fs_df[NUMERIC_FEATURES].fillna(
    fs_df[NUMERIC_FEATURES].median()
)

fs_df = fs_df.dropna(subset=["delivery_days", "late_delivery"])
fs_df = fs_df.sample(
    n=min(20000, len(fs_df)),
    random_state=RANDOM_STATE
)

# FILTER
mi_reg = pd.Series(
    mutual_info_regression(
        fs_df[NUMERIC_FEATURES],
        fs_df["delivery_days"],
        random_state=RANDOM_STATE
    ),
    index=NUMERIC_FEATURES
).sort_values(ascending=False)

mi_cls = pd.Series(
    mutual_info_classif(
        fs_df[NUMERIC_FEATURES],
        fs_df["late_delivery"].astype(int),
        random_state=RANDOM_STATE
    ),
    index=NUMERIC_FEATURES
).sort_values(ascending=False)

print("FILTER — Mutual Information for delivery_days")
display(mi_reg.to_frame("MI_regression"))

print("FILTER — Mutual Information for late_delivery")
display(mi_cls.to_frame("MI_classification"))

# WRAPPER — RFE
X_fs = StandardScaler().fit_transform(fs_df[NUMERIC_FEATURES])

rfe = RFE(
    LinearRegression(),
    n_features_to_select=min(8, len(NUMERIC_FEATURES))
)
rfe.fit(X_fs, fs_df["delivery_days"])

display(pd.DataFrame({
    "feature": NUMERIC_FEATURES,
    "selected": rfe.support_,
    "rank": rfe.ranking_
}).sort_values(["rank", "feature"]))

# EMBEDDED — Lasso
lasso_fs = Lasso(alpha=0.01, max_iter=10000)
lasso_fs.fit(X_fs, fs_df["delivery_days"])

coef = pd.Series(lasso_fs.coef_, index=NUMERIC_FEATURES)
coef = coef.reindex(coef.abs().sort_values(ascending=False).index)

display(coef.to_frame("Lasso_coefficient"))

# ## 11. PCA — Principal Component Analysis

from sklearn.decomposition import PCA

pca_df = model[NUMERIC_FEATURES].copy()
pca_df = pca_df.fillna(pca_df.median())
pca_df = pca_df.sample(
    n=min(40000, len(pca_df)),
    random_state=RANDOM_STATE
)

X_pca = StandardScaler().fit_transform(pca_df)

pca = PCA()
pca.fit(X_pca)

pca_summary = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)
})

display(pca_summary)

plt.figure(figsize=(8,4))
plt.plot(
    range(1, len(pca_summary)+1),
    pca_summary["cumulative_variance"],
    marker="o"
)
plt.axhline(0.80, linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.tight_layout()
plt.show()

# # 12. Regression

# ### 12.1 Simple Linear Regression

from sklearn.linear_model import LinearRegression

simple = model[["distance_km", "delivery_days"]].dropna().copy()
simple = simple[
    simple["delivery_days"] <= simple["delivery_days"].quantile(0.995)
]

X_train, X_test, y_train, y_test = train_test_split(
    simple[["distance_km"]],
    simple["delivery_days"],
    test_size=0.20,
    random_state=RANDOM_STATE
)

simple_lr = LinearRegression()
simple_lr.fit(X_train, y_train)
pred = simple_lr.predict(X_test)

print("Intercept:", simple_lr.intercept_)
print("Distance coefficient:", simple_lr.coef_[0])
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R2:", r2_score(y_test, pred))

# ### 12.2 Multiple OLS + regression assumptions

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

diag_cols = [
    "distance_km", "order_value", "freight_value",
    "total_weight_g", "promised_days",
    "item_count", "payment_installments"
]

diag = model[diag_cols + ["delivery_days"]].dropna().copy()
diag = diag[
    diag["delivery_days"] <= diag["delivery_days"].quantile(0.995)
]
diag = diag.sample(
    n=min(25000, len(diag)),
    random_state=RANDOM_STATE
)

X_diag = sm.add_constant(diag[diag_cols])
ols_sm = sm.OLS(diag["delivery_days"], X_diag).fit()
print(ols_sm.summary())

resid = ols_sm.resid
fitted = ols_sm.fittedvalues

print("Durbin-Watson:", durbin_watson(resid))

bp = het_breuschpagan(resid, X_diag)
print("Breusch-Pagan LM p-value:", bp[1])
print("Breusch-Pagan F p-value:", bp[3])

plt.figure(figsize=(7,4))
plt.scatter(fitted, resid, alpha=0.25)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.tight_layout()
plt.show()

sm.qqplot(resid, line="45")
plt.title("Q-Q Plot")
plt.tight_layout()
plt.show()

# ### 12.3 OLS, Ridge, Lasso, Elastic Net

from sklearn.linear_model import Ridge, Lasso, ElasticNet

reg_data = model[
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["delivery_days"]
].dropna(subset=["delivery_days"]).copy()

reg_data = reg_data[
    reg_data["delivery_days"] <=
    reg_data["delivery_days"].quantile(0.995)
]

X = reg_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = reg_data["delivery_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

reg_models = {
    "OLS": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=10000),
    "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
}

reg_results = []

for name, estimator in reg_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    reg_results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred)
    })

reg_results = pd.DataFrame(reg_results).sort_values("RMSE")
display(reg_results)

# ### 12.4 Non-linear Regression

nl = model[
    ["delivery_days", "distance_km", "order_value",
     "freight_value", "total_weight_g"]
].dropna().copy()

nl = nl[
    (nl["delivery_days"] > 0) &
    (nl["distance_km"] >= 0) &
    (nl["order_value"] > 0) &
    (nl["freight_value"] > 0) &
    (nl["total_weight_g"] > 0)
]

nl = nl[
    nl["delivery_days"] <= nl["delivery_days"].quantile(0.995)
]

train, test = train_test_split(
    nl, test_size=0.20, random_state=RANDOM_STATE
)

rows = []

# Log-linear
m1 = LinearRegression().fit(
    train[["distance_km"]],
    np.log1p(train["delivery_days"])
)
p1 = np.expm1(m1.predict(test[["distance_km"]]))
rows.append({
    "Model": "Log-linear",
    "RMSE": mean_squared_error(test["delivery_days"], p1) ** 0.5,
    "R2": r2_score(test["delivery_days"], p1)
})

# Linear-log
m2 = LinearRegression().fit(
    np.log1p(train[["distance_km"]]),
    train["delivery_days"]
)
p2 = m2.predict(np.log1p(test[["distance_km"]]))
rows.append({
    "Model": "Linear-log",
    "RMSE": mean_squared_error(test["delivery_days"], p2) ** 0.5,
    "R2": r2_score(test["delivery_days"], p2)
})

# Log-log
m3 = LinearRegression().fit(
    np.log1p(train[["distance_km"]]),
    np.log1p(train["delivery_days"])
)
p3 = np.expm1(m3.predict(np.log1p(test[["distance_km"]])))
rows.append({
    "Model": "Log-log",
    "RMSE": mean_squared_error(test["delivery_days"], p3) ** 0.5,
    "R2": r2_score(test["delivery_days"], p3)
})

# Polynomial
poly = PolynomialFeatures(degree=2, include_bias=False)
Xp_train = poly.fit_transform(train[["distance_km"]])
Xp_test = poly.transform(test[["distance_km"]])

m4 = LinearRegression().fit(Xp_train, train["delivery_days"])
p4 = m4.predict(Xp_test)

rows.append({
    "Model": "Polynomial degree 2",
    "RMSE": mean_squared_error(test["delivery_days"], p4) ** 0.5,
    "R2": r2_score(test["delivery_days"], p4)
})

display(pd.DataFrame(rows).sort_values("RMSE"))

# # 13. Time Series Forecasting — Holt-Winters

from statsmodels.tsa.holtwinters import ExponentialSmoothing

order_revenue = (
    items.groupby("order_id", as_index=False)["price"]
    .sum()
    .rename(columns={"price": "revenue"})
    .merge(
        orders[["order_id", "order_purchase_timestamp"]],
        on="order_id", how="left"
    )
    .dropna(subset=["order_purchase_timestamp"])
)

daily_revenue = (
    order_revenue
    .set_index("order_purchase_timestamp")["revenue"]
    .resample("D")
    .sum()
    .asfreq("D", fill_value=0)
)

HORIZON = 30
train_ts = daily_revenue.iloc[:-HORIZON]
test_ts = daily_revenue.iloc[-HORIZON:]

hw = ExponentialSmoothing(
    train_ts,
    trend="add",
    seasonal="add",
    seasonal_periods=7,
    initialization_method="estimated"
).fit(optimized=True)

forecast = hw.forecast(HORIZON)

print("Holt-Winters MAE:", mean_absolute_error(test_ts, forecast))
print("Holt-Winters RMSE:", mean_squared_error(test_ts, forecast) ** 0.5)

plt.figure(figsize=(12,5))
plt.plot(train_ts.index[-120:], train_ts.iloc[-120:], label="Train")
plt.plot(test_ts.index, test_ts, label="Actual")
plt.plot(forecast.index, forecast, label="Forecast")
plt.title("30-Day Revenue Forecast")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.tight_layout()
plt.show()

# # 14. Clustering — Seller Segmentation

seller_order = (
    items[["order_id", "seller_id", "price", "freight_value"]]
    .merge(
        model[[
            "order_id", "delivery_days", "late_delivery",
            "review_score", "dissatisfied"
        ]],
        on="order_id", how="inner"
    )
)

seller_perf = (
    seller_order.groupby("seller_id", as_index=False)
    .agg(
        orders=("order_id", "nunique"),
        revenue=("price", "sum"),
        avg_item_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        late_rate=("late_delivery", "mean"),
        avg_review_score=("review_score", "mean"),
        dissatisfaction_rate=("dissatisfied", "mean")
    )
)

SELLER_FEATURES = [
    "orders", "revenue", "avg_item_price", "avg_freight",
    "avg_delivery_days", "late_rate",
    "avg_review_score", "dissatisfaction_rate"
]

seller_matrix = seller_perf[SELLER_FEATURES].copy()
seller_matrix = seller_matrix.fillna(seller_matrix.median())

seller_scaler = StandardScaler()
Z = seller_scaler.fit_transform(seller_matrix)

display(seller_perf.head())

# ### 14.1 K-Means

from sklearn.cluster import KMeans

k_rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(Z)
    k_rows.append({
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(Z, labels)
    })

k_eval = pd.DataFrame(k_rows)
display(k_eval)

best_k = int(k_eval.loc[k_eval["silhouette"].idxmax(), "k"])
print("Best k:", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
seller_perf["kmeans_cluster"] = kmeans.fit_predict(Z)

display(
    seller_perf.groupby("kmeans_cluster")[SELLER_FEATURES].mean()
)

plt.figure(figsize=(7,4))
plt.plot(k_eval["k"], k_eval["silhouette"], marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("K-Means Model Selection")
plt.tight_layout()
plt.show()

# ### 14.2 Hierarchical Clustering

from sklearn.cluster import AgglomerativeClustering

seller_h = seller_perf.sample(
    n=min(1500, len(seller_perf)),
    random_state=RANDOM_STATE
).copy()

Zh = seller_scaler.transform(
    seller_h[SELLER_FEATURES].fillna(seller_matrix.median())
)

hier = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

seller_h["hier_cluster"] = hier.fit_predict(Zh)
display(seller_h["hier_cluster"].value_counts().sort_index())

# ### 14.3 DBSCAN

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1.2, min_samples=8)
seller_perf["dbscan_cluster"] = dbscan.fit_predict(Z)

print("-1 means noise/outlier")
display(seller_perf["dbscan_cluster"].value_counts().sort_index())

# # 15. Classification — Predict Late Delivery

# ### 15.1 Logistic Regression, Decision Tree, Random Forest

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

cls_data = model[
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES +
    ["late_delivery"]
].dropna(subset=["late_delivery"]).copy()

Xc = cls_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
yc = cls_data["late_delivery"].astype(int)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=yc
)

base_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=25,
        class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

classification_results = []

for name, estimator in base_models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])
    pipe.fit(Xc_train, yc_train)
    pred = pipe.predict(Xc_test)
    prob = pipe.predict_proba(Xc_test)[:,1]

    classification_results.append({
        "Model": name,
        "Accuracy": accuracy_score(yc_test, pred),
        "Precision": precision_score(yc_test, pred, zero_division=0),
        "Recall": recall_score(yc_test, pred, zero_division=0),
        "F1": f1_score(yc_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(yc_test, prob)
    })

base_cls_results = pd.DataFrame(classification_results)
display(base_cls_results.sort_values("ROC_AUC", ascending=False))

# ### 15.2 Naive Bayes and SVM

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

small_cls = cls_data[
    NUMERIC_FEATURES + ["late_delivery"]
].copy()

small_cls[NUMERIC_FEATURES] = small_cls[NUMERIC_FEATURES].fillna(
    small_cls[NUMERIC_FEATURES].median()
)

small_cls = small_cls.sample(
    n=min(25000, len(small_cls)),
    random_state=RANDOM_STATE
)

Xs = small_cls[NUMERIC_FEATURES]
ys = small_cls["late_delivery"].astype(int)

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    Xs, ys,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=ys
)

nb_svm_models = {
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
}

nb_svm_results = []

for name, est in nb_svm_models.items():
    est.fit(Xs_train, ys_train)
    pred = est.predict(Xs_test)
    prob = est.predict_proba(Xs_test)[:,1]

    nb_svm_results.append({
        "Model": name,
        "Accuracy": accuracy_score(ys_test, pred),
        "Precision": precision_score(ys_test, pred, zero_division=0),
        "Recall": recall_score(ys_test, pred, zero_division=0),
        "F1": f1_score(ys_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(ys_test, prob)
    })

display(pd.DataFrame(nb_svm_results).sort_values("ROC_AUC", ascending=False))

# ### 15.3 Bagging, Gradient Boosting and Stochastic Gradient Boosting

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier

ens = cls_data[
    NUMERIC_FEATURES + ["late_delivery"]
].copy()

ens[NUMERIC_FEATURES] = ens[NUMERIC_FEATURES].fillna(
    ens[NUMERIC_FEATURES].median()
)

ens = ens.sample(
    n=min(50000, len(ens)),
    random_state=RANDOM_STATE
)

Xe = ens[NUMERIC_FEATURES]
ye = ens["late_delivery"].astype(int)

Xe_train, Xe_test, ye_train, ye_test = train_test_split(
    Xe, ye,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=ye
)

ensemble_models = {
    "Bagging": BaggingClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    ),
    "Stochastic Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.80,
        random_state=RANDOM_STATE
    )
}

ensemble_results = []

for name, est in ensemble_models.items():
    est.fit(Xe_train, ye_train)
    pred = est.predict(Xe_test)
    prob = est.predict_proba(Xe_test)[:,1]

    ensemble_results.append({
        "Model": name,
        "Accuracy": accuracy_score(ye_test, pred),
        "Precision": precision_score(ye_test, pred, zero_division=0),
        "Recall": recall_score(ye_test, pred, zero_division=0),
        "F1": f1_score(ye_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(ye_test, prob)
    })

display(pd.DataFrame(ensemble_results).sort_values("ROC_AUC", ascending=False))

# ### 15.4 XGBoost

try:
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    xgb.fit(Xe_train, ye_train)
    xgb_pred = xgb.predict(Xe_test)
    xgb_prob = xgb.predict_proba(Xe_test)[:,1]

    xgb_result = pd.DataFrame([{
        "Model": "XGBoost",
        "Accuracy": accuracy_score(ye_test, xgb_pred),
        "Precision": precision_score(ye_test, xgb_pred, zero_division=0),
        "Recall": recall_score(ye_test, xgb_pred, zero_division=0),
        "F1": f1_score(ye_test, xgb_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(ye_test, xgb_prob)
    }])
    display(xgb_result)

except ImportError:
    print("Install XGBoost with: pip install xgboost")

# ## 16. Confusion Matrix and Feature Importance

rf_num = cls_data[
    NUMERIC_FEATURES + ["late_delivery"]
].copy()

rf_num[NUMERIC_FEATURES] = rf_num[NUMERIC_FEATURES].fillna(
    rf_num[NUMERIC_FEATURES].median()
)

Xrf_train, Xrf_test, yrf_train, yrf_test = train_test_split(
    rf_num[NUMERIC_FEATURES],
    rf_num["late_delivery"].astype(int),
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=rf_num["late_delivery"].astype(int)
)

rf = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(Xrf_train, yrf_train)

rf_pred = rf.predict(Xrf_test)
rf_prob = rf.predict_proba(Xrf_test)[:,1]

print("Confusion Matrix")
print(confusion_matrix(yrf_test, rf_pred))
print("\nClassification Report")
print(classification_report(yrf_test, rf_pred))
print("ROC-AUC:", roc_auc_score(yrf_test, rf_prob))

importance = pd.Series(
    rf.feature_importances_,
    index=NUMERIC_FEATURES
).sort_values(ascending=False)

display(importance.to_frame("importance"))

importance.head(15).sort_values().plot(
    kind="barh", figsize=(8,6)
)
plt.title("Top Predictors of Late Delivery")
plt.tight_layout()
plt.show()

# ## 17. Final Model Comparison

print("REGRESSION")
display(reg_results)

print("\nCLASSIFICATION — BASE MODELS")
display(base_cls_results.sort_values("ROC_AUC", ascending=False))

print("\nNAIVE BAYES / SVM")
display(pd.DataFrame(nb_svm_results).sort_values("ROC_AUC", ascending=False))

print("\nENSEMBLE MODELS")
display(pd.DataFrame(ensemble_results).sort_values("ROC_AUC", ascending=False))

# # 18. Managerial Interpretation
# 
# Use the actual outputs to answer:
# 
# 1. **Which factors drive delivery time?**
# 2. **Which model best predicts late delivery?**
# 3. **Which seller clusters are operationally risky?**
# 4. **How different are review scores for late vs on-time orders?**
# 5. **What does the 30-day Holt-Winters forecast imply for capacity planning?**
# 
# For classification, do not use accuracy alone because late deliveries are a minority class.
# Focus on **Recall, F1 and ROC-AUC**.
# 
# Do not claim causality from this observational dataset.