
# https://docs.streamlit.io/
# Run: streamlit run main.py
# pip install -U scikit-learn

import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC

st.title('Heart Failure?  :heart:')
st.caption('Created by: Amadin Shivani Lasha')
st.text("")
st.markdown('This project aims to develop a machine learning model to predict the risk of heart failure in patients. Heart Failure or otherwise known as Cardiovascular diseases (CVDs) are the number 1 cause of death globally , taking an estimated 17.9 million lives each year. We want to tackle this issue. The ultimate goal of this project is to develop a tool that can aid both patients and clinicians in identifying heart failure at an early stage.  ')
st.text("")

user_input = [62.0, 1, 447, 1, 30, 1, 265000.00, 2.5, 132, 1, 1, 7]


# age	
age = st.number_input('What is your age?')
st.write('Age:', age)
st.text("")
st.text("")
# anaemia
anaemia = st.radio(
    "Do you have anaemia?",
    ('Yes', 'No'))


if anaemia == 'Yes':
    anaemia = 1
else:
    anaemia = 0 

st.text("")
st.text("")

# creatinine_phosphokinase
creatinine_phosphokinase = st.number_input('What is your creatinine phosphokinase levels?')
st.write('Creatinine Phosphokinase:', creatinine_phosphokinase)

st.text("")
st.text("")

# diabetes
diabetes = st.radio(
    "Do you have diabetes?",
    ('Yes', 'No'))

if diabetes == 'Yes':
    diabetes = 1
else:
    diabetes = 0 

st.text("")
st.text("")

# ejection_fraction
ejection_fraction = st.number_input('What is your ejection fraction levels?')
st.write('ejection_fraction:', ejection_fraction)

st.text("")
st.text("")

# high_blood_pressure


high_blood_pressure = st.radio(
    "Do you have high blood pressure?",
    ('Yes', 'No'))

if high_blood_pressure == 'Yes':
    high_blood_pressure = 1
else:
    high_blood_pressure = 0
    
st.text("")
st.text("")

# platelets
platelets = st.number_input('What is your platelets levels?')
st.write('Platelets:', platelets)

st.text("")
st.text("")

# serum_creatinine
serum_creatinine = st.number_input('What is your serum creatinine levels?')
st.write('serum_creatinine:', serum_creatinine)

st.text("")
st.text("")

# serum_sodium
serum_sodium = st.number_input('What is your serum sodium?')
st.write('serum_sodium:', serum_sodium)

st.text("")
st.text("")

# sex

sex = st.radio(
    "What is your sex?",
    ('Male', 'Female'))

if sex == 'Male':
    sex = 1
else:
    sex = 0
  

st.text("")
st.text("")  

# smoking

smoking = st.radio(
    "Do you smoke?",
    ('Yes', 'No'))

if smoking == 'Yes':
    smoking = 1
else:
    smoking = 0

st.text("")
st.text("")

# time
# Follow-up period (days)

time = st.number_input(' time?')
st.write('time:', time)



user_input = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]


def predicts(user_input):
    heart_failure = pd.read_csv('heart_failure_clinical_records_dataset.csv')

    # target of the dataset is DEATH_EVENT (value = either 0 or 1). In features, we drop the target variable
    target = heart_failure['DEATH_EVENT']
    features = heart_failure.drop('DEATH_EVENT', axis=1)

    # scaling the features
    features_scaled = (features - features.mean()) / features.std()

    # after testing, we chose 9 as the value of n_components
    # applying PCA on the dataset
    pca = PCA(n_components=9)
    pca.fit(features_scaled)
    transformed_features = pca.transform(features_scaled)
    heart_failure_pca = pd.DataFrame(transformed_features, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9'])

    # ----------------------------------------------------------------------------------------------------------------------------------

    # SVM
    model = SVC()
    model.fit(heart_failure_pca, target)

    # user_input only used for testing; replace the list with real answers
    #user_input = [62.0, 1, 447, 1, 30, 1, 265000.00, 2.5, 132, 1, 1, 7]
    user_input_df = pd.DataFrame(columns=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'])
    user_input_df.loc[len(user_input_df)] = user_input

    # reducing dimensionality of user_input using the same PCA
    user_input_transformed = pca.transform(user_input_df)
    user_input_pca = pd.DataFrame(user_input_transformed, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9'])

    # prediction
    prediction = model.predict(user_input_pca)

    print(prediction)
    if prediction == [0]:
        st.error(f' This person is unlikely to have heart falure.')
    elif prediction == [1]:
        st.success(f'This person is likely to have heart falure')
    

if st.button('Predict heart failure'):

    prediction = predicts(user_input)
    