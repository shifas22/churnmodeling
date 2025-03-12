import streamlit as st
import pandas as pd
import joblib 

# loading model , scaler and encoders

model = joblib.load('LogisticRegression.pkl')
le= joblib.load('Encoder.pkl')
ohe=joblib.load('onehot.pkl')
scaler=joblib.load('scaler.pkl')



# user input 

creditScore = st.number_input('Enter Your CreditScore')

Gender = st.selectbox('Gender',["Male","Female"])
Age = st.number_input('Age')
Tenure = st.number_input('Tenure')
Balance= st.number_input('Balance')
NumOfProducts=st.number_input('Numberof Products')
HasCrCard=st.selectbox('Has Credit Card',['YES','NO'])
isActiveMember=st.selectbox('Active Member',['YES','NO'])

EstimatedSalary = st.number_input('Salary')

Country = st.selectbox('Country',['France','Germany','Spain'])


# prepocess input
HasCrCard=1 if HasCrCard  == 'YES' else 0

isActiveMember = 1 if isActiveMember == 'YES' else 0

# Encode Gepgraphy

Country= ohe.transform([[Country]]).toarray()
Country=pd.DataFrame(Country,columns=ohe.get_feature_names_out())



# Encode Gender
gender=le.transform([[Gender]])
# st.write(gender)

# # Create a DataFrame for the inputs
input_data = pd.DataFrame({
    'CreditScore': [creditScore],
    'Gender': [gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [isActiveMember],
    'EstimatedSalary': [EstimatedSalary],
})

data=pd.concat([input_data,Country],axis=1)
# st.write(data)

# # Scale the input data
data_scaled = scaler.transform(data)

# # Make prediction
prediction = model.predict(data_scaled)
if st.button("Predict"):
    # # Display the result
    if prediction[0] == 1:
        st.success("The customer is likely to churn.")
    else:
        st.error("The customer is not likely to churn.")