from cProfile import label
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta # to add days or years
import datetime as dt
st.set_page_config(
   page_title="Stock Trend Predictor",
   page_icon="🧊",
   layout="centered",
   initial_sidebar_state="expanded",
)

start = '2010-01-01'
#end = '2022-03-31'

st.title('Stock Trend Predictor')

#taking end date from user
end = st.date_input(
     "Enter end date for Prediction",
    #  datetime.date(datetime.now())
    # value=(datetime(2020, 1, 1), datetime(2030, 1, 1))
        min_value= datetime(2011, 1, 1),
        max_value=datetime.date(datetime.now()),
     )
user_input = st.text_input('Enter Stock Ticker','TSLA') #taking the stick ticker from the user

@st.cache
def getdata():
   return data.DataReader(user_input,'yahoo',start,end) #stock ticker entered
df = getdata()

s = 'Working on ' + str(user_input) + ' Data From '+start+ ' to ' + str(end)
st.subheader(s)
st.write(df.describe())
st.write()
# st.write(df.info())
df.dropna()
st.subheader(" ")
st.subheader("Predictions for Closing Price")

#doing the visualization
#1st simple closing price vs time chart
st.header('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'blue',label='Closing Price')
plt.legend()
st.pyplot(fig)

#2nd printing the moving average..
st.header('Closing Price vs Time Chart with 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'red',label='100 days moving average')
plt.plot(df.Close,'green',label='Closing Price')
plt.legend()
st.pyplot(fig)

#3rd printing the 200 moving average..
st.header('Closing Price vs Time Chart with 100 and 200 Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'red',label='100 days moving average')
plt.plot(ma200,'green',label='200 days moving average')
plt.plot(df.Close,'blue',label='Closing Price')
plt.legend()
st.pyplot(fig)


#spliting the Data into testing and traning
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])  
#we will use 70% of total data-set as training

#for datatesting
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
#rest 30% data will be test_data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#split data in x_train and y_train==>  no need for tarining
#load the model
model = load_model('stock_close_predictor.h5')

#Testing part
#take the past 100 days data
past_100_days = data_training.tail(100)

#last 100 days of data_testing and data_trainning are appended in final_df
final_df = past_100_days.append(data_testing,ignore_index = True) 
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

#again define the step size as 100
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])  #oth coulmn i.e. closing price column


#converting the testing into numpy array
x_test, y_test = np.array(x_test),np.array(y_test)

#Make The predictions
y_predicted = model.predict(x_test)

 #this is the scaleing factor, now we need to divide y_predicted and 
#y_test data by this factor..

scaler = scaler.scale_

scale_factor = 1/scaler[0] #scaler will conatin the scale factor at 0th index

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#now plotting them to get the analysis

#visualization of predicted values..
#final Graph

st.header('Stock-Market Trend Prediction Final Graph')
st.subheader('Predicted Values vs Original Values')
fig2 = plt.figure(figsize= (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)




                        #opening price

st.subheader(" ")
st.subheader("Predictions for Opening Price")

#doing the visualization
#1st simple closing price vs time chart
st.header('Opening Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close,'blue',label='Opening Price')
plt.legend()
st.pyplot(fig)

#2nd printing the moving average..
st.header('Opening Price vs Time Chart with 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'red',label='100 days moving average')
plt.plot(df.Close,'green',label='Closing Price')
plt.legend()
st.pyplot(fig)

#3rd printing the 200 moving average..
st.header('Opening Price vs Time Chart with 100 and 200 Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'red',label='100 days moving average')
plt.plot(ma200,'green',label='200 days moving average')
plt.plot(df.Close,'blue',label='Opening Price')
plt.legend()
st.pyplot(fig)


#spliting the Data into testing and traning
data_training = pd.DataFrame(df['Open'][0:int(len(df)*0.70)])  
#we will use 70% of total data-set as training

#for datatesting
data_testing = pd.DataFrame(df['Open'][int(len(df)*0.70):int(len(df))])
#rest 30% data will be test_data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#split data in x_train and y_train==>  no need for tarining
#load the model
model = load_model('Opening_predictor.h5')

#Testing part
#take the past 100 days data
past_100_days = data_training.tail(100)

#last 100 days of data_testing and data_trainning are appended in final_df
final_df = past_100_days.append(data_testing,ignore_index = True) 
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

#again define the step size as 100
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])  #oth coulmn i.e. closing price column


#converting the testing into numpy array
x_test, y_test = np.array(x_test),np.array(y_test)

#Make The predictions
y_predicted = model.predict(x_test)

 #this is the scaleing factor, now we need to divide y_predicted and 
#y_test data by this factor..

scaler = scaler.scale_

scale_factor = 1/scaler[0] #scaler will conatin the scale factor at 0th index

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#now plotting them to get the analysis

#visualization of predicted values..
#final Graph

st.header('Stock-Market Trend Prediction Final Graph')
st.subheader('Predicted Values vs Original Values')
fig3 = plt.figure(figsize= (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)
