import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Telco Customer Churn" ,
    layout='wide'
)

st.title("Telco Customer Churn")
st.markdown('Find cutomers who will more likely to churn to target them with discounts and ads')

data_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']

cat_cols = {
 'gender': ['Male','Female'],
 'SeniorCitizen' : ['Yes','No'],
 'Partner': ['Yes','No'],
 'Dependents':['Yes','No'],
 'PhoneService':['Yes','No'],
 'MultipleLines' : ['Yes','No' , 'No phone service'] ,
 'InternetService' : ['Fiber optic' ,'DSL','No'],
 'OnlineSecurity' : ['Yes','No','No internet service'],
 'OnlineBackup' : ['Yes','No','No internet service'],
 'DeviceProtection' : ['Yes','No','No internet service'],
 'TechSupport' : ['Yes','No','No internet service'],
 'StreamingTV' : ['Yes','No','No internet service'],
 'StreamingMovies' : ['Yes','No','No internet service'],
 'Contract': ['Month-to-month','Two year','One year'],
 'PaperlessBilling': ['Yes','No'],
 'PaymentMethod' : ['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']
}

num_col = ['tenure', 'MonthlyCharges', 'TotalCharges']

user_input = {}

#categorical variables
col1,col2= st.columns(2)

for i , item in enumerate(cat_cols.items() , start = 1):
    name , options = item
    if i <= 8:
        with col1:
            user_input[name] = st.selectbox(name,options)
    else:
        with col2:
            user_input[name] = st.selectbox(name,options)

#numerical values
col3,col4,col5  = st.columns(3)

cols = [col3,col4,col5]

for i,col in enumerate(cols):
    with col:
        user_input[num_col[i]] = st.number_input(num_col[i])


#change SeniorCitizen
if user_input['SeniorCitizen'] == 'Yes':
    user_input['SeniorCitizen'] = 1
else:
    user_input['SeniorCitizen'] = 0

#output
if st.button("Predict Churn"):
    try:
        user_input = pd.DataFrame([user_input])
        bundle = joblib.load("churn_model.pkl")
        model = bundle['model']
        encoder = bundle['encoder']

        cat_data = encoder.transform(user_input[list(cat_cols.keys())])

        cat_input = pd.DataFrame(
            cat_data.toarray(),
            columns = encoder.get_feature_names_out(list(cat_cols.keys())),
            index= user_input.index
        )


        user_input = pd.concat(
            [user_input.drop(columns=cat_cols) , cat_input],
            axis = 1
        )

        #debuggin
        print(user_input.shape)


        prediction = model.predict(user_input)
        if prediction == 1:
            st.success("Customer is most likely to churn")
        else:
            st.success("Customer most likely to stay")

    except Exception as e:
        st.error('please ensure to fill all the data correctly')
        st.error(e)