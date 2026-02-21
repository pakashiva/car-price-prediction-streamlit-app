# 🚗 Car Price Prediction — Streamlit Web App

## 📌 Overview

This Streamlit application provides an interactive web interface to predict used car prices using a trained machine learning model.

Users can input vehicle specifications and receive an instant predicted price.

---

## 🖥 Features

* Clean 2-column layout for better UI experience
* Light background theme
* Dropdown inputs for categorical variables
* Automatic feature engineering:

  * Age calculation
  * Mileage_per_Year computation
* Log transformation of mileage before prediction
* Automatic reverse log transformation of predicted price

---

## 📥 User Inputs

The application collects:

* Levy
* Manufacturer
* Model
* Production Year
* Category
* Leather Interior
* Fuel Type
* Engine Volume
* Mileage
* Cylinders
* Gearbox Type
* Drive Wheels
* Doors
* Wheel
* Color
* Airbags
* Turbo

---

## 🔄 Internal Processing

The app performs:

1. Target Encoding for:

   * Manufacturer
   * Category
   * Fuel Type
   * Model
   * Colour

2. One-Hot Encoding for:

   * Gear box type
   * Drive wheels
   * Wheel

3. Binary Mapping:

   * Leather Interior
   * Turbo

4. Feature Engineering:

   * Age
   * Mileage per Year

5. Log transformation:

   * Mileage (input)
   * Reverse log for predicted output

---

## 📂 Required Files

Ensure the following files are present in the same directory as `app.py`:

* `car_price_predictor.pkl`
* `model_columns.pkl`
* `target_encoding_maps.pkl`

---

## ▶️ How to Run

1. Install dependencies:

   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```

2. Run the app:

   ```bash
   streamlit run app.py
   ```

---

## 📊 Model Information

* Model: RandomForestRegressor
* R² Score: 0.95
* RMSE (Log Scale): 0.22

---

## Live Demo
link : https://car-price-predictor-us-market-app.streamlit.app/

---

## 🚀 Future Improvements

* Convert preprocessing into full sklearn Pipeline
* Deploy to Streamlit Cloud
* Add feature importance visualization
* Add confidence interval prediction

---

This application demonstrates a complete end-to-end machine learning deployment workflow using Streamlit.
