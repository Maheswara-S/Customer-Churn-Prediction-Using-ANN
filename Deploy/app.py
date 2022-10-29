import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

model = tf.keras.models.load_model('churn_seq_model.h5')
preprocess = pickle.load(open('preprocess.pkl','rb'))

st.header('Customer Churn Prediction')

tenure = st.number_input('tenure')
MonthlyCharges = st.number_input('MonthlyCharges')


OnlineSecurity = st.selectbox('OnlineSecurity', ['Yes','No','No internet service'])
TechSupport = st.selectbox('TechSupport',['Yes','No','No internet service'])
Contract = st.selectbox('Contract',['One year','Two year','Month-to-month'])
PaperlessBilling = st.selectbox('PaperlessBilling',['Yes','No'])

if st.button('Submit'):

    X = pd.DataFrame([[tenure,MonthlyCharges,OnlineSecurity,TechSupport,Contract,PaperlessBilling]],
                       columns=['tenure','MonthlyCharges','OnlineSecurity','TechSupport','Contract','PaperlessBilling'])

    pipe = preprocess.transform(X)
                       
    pred = model.predict(pipe)

    if pred[0] == 1:
        ans = "Yes"
    else:
        ans = "No"

    st.text(f'Churn {ans[0]}')