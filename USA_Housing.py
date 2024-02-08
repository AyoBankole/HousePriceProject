import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
import streamlit as st
import joblib
from sklearn.linear_model import LinearRegression
# import data
data= pd.read_csv('USA_Housing.csv')
# import the model
model= joblib.load('USA Housing predict.pkl')

st.markdown("<h1 style = 'color: #1F4172; text-align: center; font-family: helvetica '>HOUSING PRICE PREDICTION</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By The Mushin Data Guy</h4>", unsafe_allow_html = True)

##st.write('By analyzing a diverse set of parameters, including Market Expense, Administrative Expense, and Research and Development Spending, our team seeks to develop a robust predictive model that can offer valuable insights into the future financial performance of startups. This initiative not only empowers investors and stakeholders to make data-driven decisions but also provides aspiring entrepreneurs with a comprehensive framework to evaluate the viability of their business models and refine their strategies for long-term success')
st.image('pngwing.com.png', width = 350, use_column_width = True )
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h4 style = 'color: #1F4172; text-align: center; font-family:sans serif '>Project Overview</h4>", unsafe_allow_html = True)
st.markdown("<p style = 'text-align: justify'>The predictive house price modeling project aims to leverage machine learning techniques to develop an accurate and robust model capable of predicting the market value of residential properties. By analyzing historical data, identifying key features influencing house prices, and employing advanced regression algorithms, the project seeks to provide valuable insights for homebuyers, sellers, and real estate professionals. The primary objective of this project is to create a reliable machine learning model that accurately predicts house prices based on relevant features such as location, size, number of bedrooms, amenities, and other influencing factors. The model should be versatile enough to adapt to different real estate markets, providing meaningful predictions for a wide range of properties.", unsafe_allow_html=True)
st.sidebar.image('admin-user-icon-4.jpg', caption= 'Welcome User')
st.markdown("<br>", unsafe_allow_html = True)

st.dataframe(data, use_container_width = True )

input_choice= st.sidebar.radio('Choose Your Input Type', ['Slider Input', 'Number Input'])

if input_choice== 'Slider Input':
    area_income= st.sidebar.slider('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age= st.sidebar.slider('Average Area House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_num= st.sidebar.slider('Average Area Number of Rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms= st.sidebar.slider('Average Area Number of Bedrooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    area_population= st.sidebar.slider('Area Population', data['Area Population'].min(), data['Area Population'].max())
else:
    area_income= st.sidebar.number_input('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age= st.sidebar.number_input('Average Area House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_num= st.sidebar.number_input('Average Area Number of Rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms= st.sidebar.number_input('Average Area Number of Bedrooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    area_population= st.sidebar.number_input('Area Population', data['Area Population'].min(), data['Area Population'].max())

input_var = pd.DataFrame({'Avg. Area Income': [area_income],
                           'Avg. Area House Age': [house_age], 
                           'Avg. Area Number of Rooms': [room_num],
                          'Avg. Area Number of Bedrooms':[bedrooms],
                           'Area Population':[area_population] })
# st.markdown(css + "<hr class= 'colorful-divider>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h5 style= 'margin: -30px; color:olive; font:sans serif' >", unsafe_allow_html= True)
st.dataframe(input_var)

predicted = model.predict(input_var)
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The Predicted price of your house is {predicted}')

with interprete:
    st.header('The Interpretation Of The Model')
    st.write(f'The intercept of the model is: {round(model.intercept_, 2)}')
    st.write(f'A unit change in the average area income causes the price to change by {model.coef_[0]} naira')
    st.write(f'A unit change in the average house age causes the price to change by {model.coef_[1]} naira')
    st.write(f'A unit change in the average number of rooms causes the price to change by {model.coef_[2]} naira')
    st.write(f'A unit change in the average number of bedrooms causes the price to change by {model.coef_[3]} naira')
    st.write(f'A unit change in the average number of populatioin causes the price to change by {model.coef_[4]} naira')
# ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
#        'Avg. Area Number of Bedrooms', 'Area Population']
 
# st.markdown("<br>", unsafe_allow_html= True)
# st.markdown("<h4>", unsafe_allow_html= True)
# st.write('Input Variables')
# input_var = pd.DataFrame({'R&D Spend': [rd_spend], 'Administration': [admin], 'Marketing Spend': [mkt_spend]})
# st.dataframe(input_var)

# model = joblib.load('StartUpModel.pkl')


# predicter =st.button('Predict Profit')
# if predicter:
#     prediction = model.predict(input_var)
#     st.success(f"The Predicted value for your company is {prediction}")
#     st.balloons()