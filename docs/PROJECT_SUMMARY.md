# Project Summary

## Project
**E-Commerce Delivery Risk & Customer Dissatisfaction Prediction**

## Problem
E-commerce marketplaces need to identify delivery failures early, manage sellers based on operational reliability, and plan short-term capacity without relying only on descriptive reporting.

## Solution
An integrated machine-learning framework was created from nine relational Olist datasets. The analysis combines regression, classification, clustering, and time-series forecasting.

## Main analytical tasks
1. Predict delivery days.
2. Predict late delivery.
3. Segment sellers by operational risk.
4. Compare customer dissatisfaction for on-time vs late orders.
5. Forecast short-term marketplace revenue.

## Key findings
- Ridge was the strongest main delivery-time regression model with RMSE ≈ 6.88 days.
- Random Forest achieved the strongest overall classification ROC-AUC ≈ 0.766.
- Decision Tree achieved higher late-delivery recall ≈ 62.9% at the default threshold.
- K-Means identified four seller segments.
- The operationally risky seller cluster had a late rate ≈ 28.3% and dissatisfaction ≈ 68.5%.
- Late deliveries were associated with dissatisfaction rising from ≈ 9.19% to ≈ 53.98%.
- Holt-Winters provided a useful forecasting baseline but over-forecasted the holdout period.

## Business recommendations
- Score delivery risk at order placement.
- Use proactive customer communication for high-risk orders.
- Apply differentiated SLA monitoring to seller clusters.
- Track late rate, dissatisfaction rate, cluster migration, and forecast error.
- Improve future forecasting with richer demand and operational drivers.

## Important limitation
The study uses historical observational data. Results should be interpreted as predictive associations rather than causal effects.
