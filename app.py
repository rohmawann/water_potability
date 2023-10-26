#import library
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#8B0000;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Water Potability Prediction App </h1>
		    <h4 style="color:white;text-align:center;">Battleship Team </h4>
		    </div>
            """

desc_temp = """
            ### Water Potability Prediction App
            This app will be used to predict water potability
            #### Data Source
            - https://raw.githubusercontent.com/rene-gith/water-potability/main/water_potability.csv
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Section
            """

def main():
    """
    Membuat framework
    """
    stc.html(html_temp)

    menu = ["Home","Machine Learning Water Potability"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning Water Potability":
        run_ml_app()

if __name__ == '__main__':
    main()