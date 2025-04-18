import streamlit as st
import numpy as np
import pickle

params = pickle.load(open("iris_ffnn.pkl","rb"))
def predict_iris(x):  
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    h = np.maximum(0, x @ W1 + b1)         
    logits = h @ W2 + b2
    return np.argmax(logits, axis=1)[0]     

st.title("Iris Species Predictor")
sl = st.slider("Sepal length", 4.0, 8.0, 5.0)
sw = st.slider("Sepal width", 2.0, 4.5, 3.0)
pl = st.slider("Petal length", 1.0, 7.0, 2.0)
pw = st.slider("Petal width", 0.1, 2.5, 0.5)

if st.button("Classify"):
    cls = predict_iris(np.array([[sl, sw, pl, pw]]))
    st.write(f"Predicted class: {['setosa','versicolor','virginica'][cls]}")
