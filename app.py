import streamlit as st
import pickle
import pandas as pd
from sklearn.neural_network import MLPClassifier

# Memuat model yang telah dilatih
jst = pickle.load(open('diabetes-model.sav', 'rb'))

# Pra-pemrosesan data
def preprocess_input(gender, smoking_history):
    gender_dict = {'Female': 0, 'Male': 1, 'Other': 2}
    smoking_history_dict = {'No Info': 0, 'never': 1, 'former': 2, 'current': 3, 'not current': 4, 'ever': 5}
    
    return gender_dict[gender], smoking_history_dict[smoking_history]

# Mendefinisikan fungsi prediksi
def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level):
    gender, smoking_history = preprocess_input(gender, smoking_history)
    
    data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })
    
    # Melakukan Prediksi
    prediction = jst.predict(data)
    return prediction[0]

# Tampilan aplikasi web
def main():
    st.title("Website Prediksi Diabetes")

    # Tampilan form input
    st.subheader("Masukkan Data Pasien")
    gender = st.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'])
    age = st.number_input("Usia", min_value=0.0, step=0.1, format="%.1f")
    hypertension = st.selectbox("Hipertensi", [0, 1])
    heart_disease = st.selectbox("Penyakit Jantung", [0, 1])
    smoking_history = st.selectbox("Riwayat Merokok", ['No Info', 'never', 'former', 'current', 'not current', 'ever'])
    bmi = st.number_input("BMI", min_value=0.00, format="%.2f")
    HbA1c_level = st.number_input("Tingkat HbA1c", min_value=0.0, format="%.1f")
    blood_glucose_level = st.number_input("Tingkat Glukosa Darah", min_value=0, step=1)

    # Prediksi saat tombol ditekan
    if st.button("Prediksi"):
        result = predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level)
        if result == 0:
            st.success("Anda tidak terkena diabetes")
        else:
            st.error("Anda memiliki diabetes")

if __name__ == '__main__':
    main()
