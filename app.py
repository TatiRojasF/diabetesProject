import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.datasets import load_diabetes


st.title("This app is to predict the glucose level in the blood of a diabetic patient")

model_lr=pickle.load(open(r'Models\model_lr.pkl','rb'))
model_en=pickle.load(open(r'Models\model_en.pkl','rb'))
model_ridge=pickle.load(open(r'Models\model_ridge.pkl','rb'))

#load the dataset
diab=load_diabetes()
X=pd.DataFrame(diab.data, columns=diab.feature_names)

#user data
user_input={}

for col in X.columns:#le estamos dando la posibilidad al usuario que seleccione en el rango que ya esta entrenado
    user_input[col]=st.slider(col, X[col].min(),X[col].max()) 
                                                            
df=pd.DataFrame(user_input,index=[0])

st.write(df)

models={"Linear Regression":model_lr,"Elastic Net":model_en,"Ridge":model_ridge} # Cree un diccionario con key modelos 
selected_model=st.selectbox("Select a model",("Linear Regression","Elastic Net","Ridge")) # modelo que seleccione el usuario


if st.button("Predict"):
    prediction=models[selected_model].predict(df)[0]
    st.write(f'The predicted glucose level is{prediction}')
