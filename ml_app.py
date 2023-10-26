import streamlit as st
import numpy as np

import joblib
import os

attribute_info = """
                - pH: The pH level of the water.
                - Hardness: Water hardness, a measure of mineral content.
                - Solids: Total dissolved solids in the water.
                - Chloramines: Chloramines concentration in the water.
                - Sulfate: Sulfate concentration in the water.
                - Conductivity: Electrical conductivity of the water.
                - Organic_carbon: Organic carbon content in the water.
                - Trihalomethanes: Trihalomethanes concentration in the water.
                - Turbidity: Turbidity level, a measure of water clarity.
                - Potability: Target variable; indicates water potability with values 1 (potable) and 0 (not potable).
                 """


@st.cache        
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

        
def run_ml_app():
    """
    Framework ML and its Prediction
    """

    st.subheader("ML Section")

    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    ph = st.number_input("pH levels", 0, 14)
    hardness = st.number_input("Hardness",47,324)
    solid = st.number_input("Solids",320,61228)
    chloramines = st.number_input("Chloramines Concentration",0,14)
    sulfate = st.number_input("Sulfate Concentration",129,482)
    conductivity = st.number_input("Electrical Conductivity",181,754)
    organic_carbon = st.number_input("Organic Carbon",2,29)
    trihalomethanes = st.number_input("Trihalomethanes Concentration",0,124)
    turbidity = st.number_input("Turbidity",1,7)

    with st.expander("Your Selected Options"):
        result = {
            'PH_Levels':ph,
            'water_hardness':hardness,
            'solid':solid,
            'chloramines':chloramines,
            'sulfate':sulfate,
            'conductivity':conductivity,
            'organic_carbon':organic_carbon,
            'trihalomethanes':trihalomethanes,
            'turbidity':turbidity,
        }

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)

    # st.write(encoded_result)

    ## prediction section
    st.subheader("Prediction Result")
    single_sample = np.array(encoded_result).reshape(1,-1)

    # st.write(single_sample)

    ## ML model
    model = load_model('model_grad.pkl')

    # prediction
    prediction = model.predict(single_sample)
    pred_prob = model.predict_proba(single_sample)

    # st.write(prediction)
    # st.write(pred_prob)

    pred_prob_score = {'potable':round(pred_prob[0][1]*100,4),
                       'not potable':round(pred_prob[0][0]*100,4)}


    if prediction == 1:
        st.success("Water is potable")
        st.write(pred_prob_score)

    elif prediction == 0:
        st.warning("Water is not potable")
        st.write(pred_prob_score)