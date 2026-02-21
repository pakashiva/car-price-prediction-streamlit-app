import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# PAGE CONFIG

st.set_page_config(page_title="Car Price Predictor", layout="wide")

encoding_maps = joblib.load("target_encoding_maps.pkl")


st.title("🚗 Car Price Prediction App")

# LOAD MODEL

model = joblib.load("car_price_predictor_model.pkl")  # change path if needed
model_columns = joblib.load("model_columns.pkl")  # Save during training

current_year = datetime.now().year

# INPUT SECTION (2 per row)

def two_cols():
    return st.columns(2)

# Row 1
col1, col2 = two_cols()
with col1:
    levy = st.number_input("Levy ($)")
with col2:
    mileage = st.number_input("Mileage (KM)")

# Row 2
col1, col2 = two_cols()
with col1:
    manufacturer = st.selectbox("Manufacturer", ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 
                                                 'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN', 'AUDI', 'RENAULT', 
                                                 'NISSAN', 'SUBARU', 'DAEWOO', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'MAZDA', 
                                                 'GMC', 'FIAT', 'INFINITI', 'ALFA ROMEO', 'SUZUKI', 'LINCOLN', 'VAZ', 'GAZ',
                                                'CITROEN', 'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR', 'ISUZU', 'ACURA', 
                                                'SKODA', 'DAIHATSU', 'BUICK', 'TESLA', 'CADILLAC', 'PEUGEOT', 'BENTLEY', 'VOLVO',  'HAVAL', 
                                                'HUMMER', 'SCION', 'UAZ', 'MERCURY', 'ZAZ', 'ROVER', 'SEAT', 'LANCIA', 'MOSKVICH', 'FERRARI', 'MASERATI', 
                                                'SAAB', 'PONTIAC', 'SATURN', 'ASTON MARTIN', 'GREATWALL'])  
with col2:
    model_name = st.selectbox("Model", ['Sonata', 'Prius', 'Elantra', 'Camry', 'Santa FE', 'H1', 'E 350',
                                                    'Tucson', 'FIT', 'X5', 'Aqua', 'Cruze', 'Fusion', 'Optima', 'Transit',
                                                    'Actyon', 'Jetta', 'Highlander', 'GX 460', 'Civic', 'REXTON', 'ML 350',
                                                    'RAV 4', 'Astra', 'Captiva', 'Lacetti', 'Juke', 'GX 470', 'Forester',
                                                    'Volt', 'RX 450', 'Passat', 'CT 200h', 'Prius C', 'Genesis', 'Escape',
                                                    'Orlando', 'Insight', 'Malibu', 'Focus', '328', 'Korando', 'CHR',
                                                    'Vito', 'Tacoma', 'Sprinter', 'Vectra', 'Tiida', 'Golf', 'Vitz', 'Cr-v',
                                                    'Corolla', 'A4', 'C 180', '528', 'E 320', 'GLE 350', 'RX 350',
                                                    'Outlander', 'C-MAX', 'Accent', '535', 'Fiesta', 'Note', 'XV', 'Pajero',
                                                    'X6', 'Compass', '320', 'Sportage'])

# Row 3
col1, col2 = two_cols()
with col1:
    prod_year = st.selectbox("Prod. year", [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 
                                            1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                                              2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
with col2:
    category = st.selectbox("Category", ['Cabriolet', 'Coupe', 'Goods wagon', 'Hatchback', 'Jeep', 'Limousine',
                                        'Microbus', 'Minivan', 'Pickup', 'Sedan', 'Universal'])

# Row 4
col1, col2 = two_cols()
with col1:
    leather = st.selectbox("Leather interior", ["Yes", "No"])
with col2:
    fuel = st.selectbox("Fuel type", ['Hybrid', 'Petrol', 'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG', 'Hydrogen'])

# Row 5
col1, col2 = two_cols()
with col1:
    engine_volume = st.number_input(
    "Engine volume", 
    min_value=1.0, 
    max_value=7.5, 
    value=2.0,  
    step=0.1  
)

# Row 6
col1, col2 = two_cols()
with col1:
    cylinders = st.selectbox("Cylinders", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0])
with col2:
    gearbox = st.selectbox("Gear box type", ['Automatic', 'Manual', 'Tiptronic', 'Variator'])

# Row 7
col1, col2 = two_cols()
with col1:
    drive = st.selectbox("Drive wheels", ['4x4', 'Front', 'Rear'])
with col2:
    doors = st.selectbox("Doors", [2 , 4  , 5])

# Row 8
col1, col2 = two_cols()
with col1:
    wheel = st.selectbox("Wheel", ['Left wheel', 'Right-hand drive'])
with col2:
    color = st.selectbox("Color", ['Silver', 'Black', 'White', 'Grey', 'Blue', 'Green', 'Red', 'Sky blue',
                                    'Orange', 'Yellow', 'Golden', 'Beige', 'Brown', 'Carnelian red', 'Purple', 'Pink'])

# Row 9
col1, col2 = two_cols()
with col1:
    airbags = st.selectbox("Airbags", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
with col2:
    turbo = st.selectbox("Turbo", ["Yes", "No"])

# PREDICTION BUTTON

if st.button("Predict Price"):

    # PREPROCESSING

    leather_map = {"Yes": 1, "No": 0}
    turbo_map = {"Yes": 1, "No": 0}

    age = current_year - int(prod_year)
    mileage_logged = np.log1p(mileage)
    mileage_per_year = mileage / (age + 1)

    # Base dataframe
    input_data = pd.DataFrame({
        "Levy": [levy],
        "Manufacturer": [manufacturer],
        "Model": [model_name],
        "Prod. year": [prod_year],
        "Category": [category],
        "Leather interior": [leather_map.get(leather, 0)],
        "Fuel type": [fuel],
        "Engine volume": [engine_volume],
        "Mileage": [mileage_logged],
        "Cylinders": [cylinders],
        "Gear box type": [gearbox],
        "Drive wheels": [drive],
        "Doors": [doors],
        "Wheel": [wheel],
        "Color": [color],
        "Airbags": [airbags],
        "Turbo": [turbo_map.get(turbo, 0)],
        "Age": [age],
        "Mileage_per_Year": [mileage_per_year]
    })

    # ONE HOT ENCODING

    one_hot_cols = ['Gear box type', 'Drive wheels', 'Wheel']
    input_data = pd.get_dummies(input_data, columns=one_hot_cols)

    # TARGET ENCODING

    for col in ['Manufacturer', 'Category', 'Fuel type', 'Model', 'Color']:
        input_data[col] = input_data[col].map(encoding_maps[col])

    input_data.fillna(0, inplace=True)

    # ALIGN COLUMNS

    input_data = input_data.reindex(columns=model_columns, fill_value=0)
      

    # PREDICTION

    prediction_log = model.predict(input_data)[0]
    prediction = np.expm1(prediction_log)

    st.success(f"💰 Predicted Price: ${round(prediction,2)} ± 5%")