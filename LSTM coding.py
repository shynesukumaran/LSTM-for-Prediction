#Load libraries
import numpy as np
from math import sqrt
from numpy import split
from numpy import array
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import math
import glob
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import load_model

#Create path to the file and load the data
path = r'*YOUR PATH*'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

dataframe= pd.concat(li, axis=0, ignore_index=True)

print(dataframe)
dataframe.to_csv("gen_data_final")

#


dataframe = dataframe.replace(np.inf, np.nan)
dataframe = dataframe.fillna(0)
dataframe['Date and Time'] = pd.to_datetime(dataframe['Date and Time'])
dataframe1 = dataframe

dataframe = dataframe.set_index('Date and Time')
scaler = MinMaxScaler(feature_range=(0, 1))
dataframe = scaler.fit_transform(dataframe)

def split(data):
    train, test = data[0:86400],data[86400:108000]
    train = np.array(np.array_split(train,len(train)/96))
    test = np.array(np.array_split(test,len(test)/96))
    return train,test

  
def to_supervized(train,n_in,n_out):
    data=train.reshape((train.shape[0]*train.shape[1],train.shape[2]))
    X,y=[],[]
    in_start=0

    for i in range(len(data)):
        in_end=in_start+n_in
        out_end=in_end+n_out
        if out_end<len(data):
            x_in=data[in_start:in_end,0]
            x_in=x_in.reshape((len(x_in),1))
            X.append(x_in)
            y.append(data[in_end:out_end,0])
        in_start+=1
    return np.array(X),np.array(y)
 

def build_model(train,n_input):
    train_x,train_y=to_supervized(train,n_input,n_input)
    verbose=1
    epochs=4
    batch_size=96
    n_timesteps=train_x.shape[1]
    n_features=train_x.shape[2]
    n_outputs=train_y.shape[1]
    model=Sequential()
    model.add(LSTM(200,activation='relu', return_sequences=True, input_shape=(n_timesteps,n_features)))
    model.add(LSTM(100,activation='relu'))  # returns a sequence of vectors of dimension 100
    model.add(Dense(32,activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse',optimizer='adam', metrics= ['accuracy'])
    early_stop = EarlyStopping(monitor = 'loss', patience = 3, verbose = verbose)
    model.fit(train_x,train_y,epochs=epochs,batch_size=batch_size,verbose=verbose, callbacks = [early_stop])
    return model

  
  def forecasts(act,pred):
    scores=[]
    for i in range(act.shape[1]):
        mse=mean_squared_error(act[:,i],pred[:,i])
        rmse=np.sqrt(mse)
        scores.append(rmse)
    s=0
    for row in range(act.shape[0]):
        for col in range(act.shape[1]):
            s+=(act[row,col]-pred[row,col])**2
    score=np.sqrt(s/(act.shape[0]*act.shape[1]))
    return score,scores
  
  
  def forecast(model,history,n_input):
    data=np.array(history) #array of train data
    data=data.reshape((data.shape[0]*data.shape[1],data.shape[2]))
    in_x=data[-n_input:,0]
    #print(in_x)
    in_x=in_x.reshape((1,len(in_x),1))
    yhat=model.predict(in_x,verbose=0)
    #print(yhat[0])
    return yhat[0]
 

def evaluate_model(model,train,test,n_input):
    #model=build_model(train,n_input)
    history=[x for x in train]  # list of train data
    #return history
    prediction=[]
    for i in range(len(test)):
        yhat_s=forecast(model,history,n_input)
        prediction.append(yhat_s)
        history.append(test[i,:])
    prediction=np.array(prediction)
    return prediction
    #score,scores=forecasts(test[:,:,0],prediction)
    #return score,scores
    

 def train_model(train,test,n_input):
    model=build_model(train,n_input)
    return model

  
 train, test = split(dataframe)


print('Train \n', train)
print('Test \n', test)


train = train.reshape(900,96,1)
test = test.reshape(225, 96, 1)
print(train.shape)
print(test.shape)



test_x,test_y = to_supervized(test,96,96)
train_x,train_y = to_supervized(train,96,96)


model = train_model(train,test,96)


import pickle
with open('model_pickle', 'wb') as g:
    pickle.dump(model,g)
    
    
prediction = evaluate_model(model, train, test, 96)
print(prediction)
print(prediction.shape)

score,scores=forecasts(test[:,:,0],prediction)


lst  = list(range(1440))
minutes = lst[0::15]
plt.figure(figsize=(10,5))
plt.plot()
plt.plot(minutes,scores)
plt.xlabel('Future datapoints')
plt.ylabel('RMSE score')
plt.title('RMSE score plot for the prediction of Active Power Generation',fontsize=12) 
plt.savefig('RMSE_Gen.png')
plt.show()



prediction_future = []
flat_list = []
output_list = list(test[len(test)-1])
flat_list = [item for sublist in output_list for item in sublist]

x_in = []
for j in range(1, 97):
    x_in.append(flat_list[len(flat_list)-97+j])
x_in = np.array(x_in)
x_in = x_in.reshape((1,len(x_in),1))
#print(x_in.shape)
yhat =  saved_gen_model.predict(x_in,verbose=1)
#test_pred = reload_model.predict(test_x)
#train_pred = reload_model.predict(train_x)

yhat = yhat.tolist()

yhat_new = [item for sublist in yhat for item in sublist]
for k in yhat_new:
    flat_list.append(k)
    prediction_future.append(k)
#prediction.append(yhat)
prediction_future  = scaler.inverse_transform([prediction_future])
print(prediction_future)
print(prediction_future.shape)



import datetime
DateTime =[]
date_time = datetime.datetime(2020,3,12,9,15,0)
for i in range(96): 
    date_time += datetime.timedelta(minutes=15)
    DateTime.append(str(date_time))

plt.figure(figsize=(25,5))
plt.plot(DateTime,prediction_future[0])
plt.xticks(rotation='vertical')
plt.ylabel('Active Power') 
plt.title('Prediction of Active Power Generation') 
plt.show()






