# E-Commerce Delivery Risk & Customer Dissatisfaction Prediction

An end-to-end machine-learning project built on the **Olist Brazilian E-Commerce** marketplace dataset. The project uses approximately **99K orders** from multiple relational tables to predict delivery time and late-delivery risk, segment sellers by operational performance, study customer dissatisfaction, and forecast short-term marketplace revenue.

## Business Problem

Can an e-commerce marketplace identify high-risk deliveries before they fail, distinguish operationally risky sellers, and use historical demand patterns to support better customer experience and capacity planning?

## Project Objectives

- Predict **actual delivery time** using regression.
- Predict **late-delivery risk** using classification.
- Identify **seller operational segments** using clustering.
- Analyze the relationship between delivery reliability and **customer dissatisfaction**.
- Forecast **short-term marketplace revenue** using time-series modelling.

## Dataset

Source: **Brazilian E-Commerce Public Dataset by Olist**.

The project integrates the following tables:

1. Orders
2. Order Items
3. Payments
4. Reviews
5. Products
6. Sellers
7. Customers
8. Geolocation
9. Product Category Translation

Raw data files are intentionally **not included** in this repository. See [`data/README.md`](data/README.md) for the expected filenames and setup instructions.

## Machine-Learning Workflow

### Data Preparation
- Duplicate handling
- Missing-value analysis and business-rule-based imputation
- Datetime conversion
- Relational table merging
- Sampling and train/test validation

### Feature Engineering
- Seller-customer Haversine distance
- Promised delivery window
- Item count and unique products
- Order value and freight ratio
- Product weight and volume
- Purchase month, weekday, hour and weekend flag
- Same-state shipment indicator

### Feature Selection & Transformation
- Mutual Information
- Recursive Feature Elimination (RFE)
- Lasso embedded selection
- Principal Component Analysis (PCA)

### Regression
- Simple Linear Regression
- Multiple OLS
- Ridge
- Lasso
- Elastic Net
- Log-linear
- Linear-log
- Log-log
- Polynomial Regression

### Classification
- Logistic Regression
- Naive Bayes
- Decision Tree
- Support Vector Machine (RBF)
- Random Forest
- Bagging
- Gradient Boosting
- Stochastic Gradient Boosting
- XGBoost

### Clustering
- K-Means
- Hierarchical Clustering
- DBSCAN

### Time Series
- Holt-Winters Exponential Smoothing

## Key Results

### Delivery-Time Regression
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Ridge** | **4.836** | **6.881** | **0.304** |
| OLS | 4.837 | 6.882 | 0.303 |
| Lasso | 4.855 | 6.896 | 0.301 |
| Elastic Net | 4.855 | 6.896 | 0.300 |

**Ridge Regression** achieved the lowest test RMSE among the main linear/regularized models.

### Late-Delivery Classification
| Model | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|
| **Random Forest** | 28.6% | **30.6%** | **0.766** |
| Decision Tree | **62.9%** | 28.1% | 0.743 |
| Logistic Regression | 62.6% | 22.7% | 0.698 |
| SVM (RBF)* | 58.6% | 21.2% | 0.671 |
| Naive Bayes* | 96.8% | 14.8% | 0.574 |

\* Naive Bayes and SVM were evaluated on a reproducible 10,000-order sample for computational efficiency.

Random Forest produced the strongest overall **ROC-AUC**, while Decision Tree achieved substantially higher **recall** at the default threshold.

### Seller Segmentation
K-Means selected **4 seller clusters** using the silhouette score (**0.493**). One operationally risky cluster showed approximately:

- **28.3% late-delivery rate**
- **20.14 average delivery days**
- **2.14 average review score**
- **68.5% dissatisfaction rate**

### Customer Experience
For orders with both delivery and review outcomes:

| Metric | On-time | Late |
|---|---:|---:|
| Average review score | 4.29 | 2.57 |
| Dissatisfaction rate | 9.19% | 53.98% |
| Average delivery days | 10.88 | 31.38 |

Late delivery was associated with an approximately **6× increase in dissatisfaction**.

### Forecasting
Holt-Winters was used as a baseline 30-day revenue forecasting model. The holdout-period RMSE was approximately **19,355 revenue units/day**, indicating that trend and weekly seasonality alone do not capture all marketplace demand variation.

## Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- XGBoost
- Matplotlib
- Jupyter Notebook

## Repository Structure

```text
ecommerce-delivery-risk-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── DSML_Olist_Complete_Project.ipynb
│   └── DSML_Olist_Complete_Project_EXECUTED_RESULTS.ipynb
├── src/
│   └── dsml_olist_project.py
├── data/
│   └── README.md
├── results/
│   ├── README.md
│   ├── tables/
│   └── plots/
├── presentation/
│   ├── DSML_Olist_ECommerce_Project_Presentation.pptx
│   └── DSML_Olist_ECommerce_Project_Presentation.pdf
└── docs/
    └── PROJECT_SUMMARY.md
```

## How to Run

1. Download the Olist dataset.
2. Place the CSV files listed in [`data/README.md`](data/README.md) inside the `data/` folder.
3. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

4. Open:

```text
notebooks/DSML_Olist_Complete_Project.ipynb
```

5. Run the notebook from top to bottom.

The notebook searches for the dataset in `../data`, `data`, and the current working directory.

## Business Takeaway

The project demonstrates how machine learning can support proactive e-commerce operations by combining **delivery-risk scoring, seller segmentation, customer-experience analysis, and demand forecasting** in one analytical framework.

## Notes

- Results are predictive/associational and should not be interpreted as causal effects.
- Classification metrics should be interpreted with class imbalance in mind; accuracy alone is not sufficient.
- The raw Olist data is not included in this repository.
