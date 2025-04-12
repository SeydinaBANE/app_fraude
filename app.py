import streamlit as st
import numpy as np
import pickle

# Charger le modèle
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Détection de Fraude Bancaire")

st.write("Remplissez les données de la transaction à analyser.")

# Lister toutes les variables V1 à V28 + Amount + Time
features = []

features.append(st.number_input("⏱️ Time (en secondes)", 0.0,20000.0,10000.0))
for i in range(1,29):
    features.append(st.number_input(f"V{i}, value=0.0"))
features.append(st.number_input("💰 Total (en €)",0.0,100.0))

input_array = np.array(features).reshape(1,-1)

#prediction
if st.button("Analyser la transaction"):
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0][1]
    if prediction ==1:
        st.error(f"⚠️ Fraude detecté !avec la probabilité :{proba:.2f}")
        
    else:
        st.success(f"✅ Transanction normal ! probabilité de fraude:{proba:.2f}")    