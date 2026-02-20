# 🪙 Bitcoin Price Prediction using SVM

This project demonstrates how to use **Support Vector Regression (SVR)** with a **Radial Basis Function (RBF) kernel** to predict future Bitcoin prices.  
The model is trained on historical price data and forecasts Bitcoin’s price for the next 30 days.
 
---

## 📌 Features
- Loads Bitcoin price data from a CSV file.
- Preprocesses data by dropping the `Date` column and creating a shifted `Prediction` column.
- Splits the dataset into **training** and **testing** subsets.
- Uses **Support Vector Machine (SVR)** regression with RBF kernel.
- Evaluates the model using the **R² accuracy score**.
- Predicts Bitcoin prices for the next **30 days**.
- Compares actual vs. predicted prices.

---
