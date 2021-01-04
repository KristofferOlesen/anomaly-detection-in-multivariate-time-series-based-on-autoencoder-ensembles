
#from google.colab import drive
import io
import os
import json
import torch
import copy
import pickle
import math
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#!pip install optuna
import optuna
#path = "C:/Users/kein/Documents/PhD/Kurser/Deep learning/Deep learning project/To remote/"
df = pd.read_csv("woodsense-sensor-data-2020-11-30-cleaned.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Split into train/test based on sensor id. We have 66 sensors. Sensors [20, 25, 26, 27, 50, 51] have been identified by woodsense to be test sensors. We randomly chose 3 and 5 remaining sensors to be test and validation sensors
test_sensors = [20, 25, 26, 27, 50, 51, 12, 42, 17]
validation_sensors = [14, 55, 34, 66, 44]
df_sensor_train = df[(~df['sensor_id'].isin(validation_sensors)) & (~df['sensor_id'].isin(test_sensors))].drop('weather_wind_min',axis=1)
df_sensor_val = df[df['sensor_id'].isin(validation_sensors)].drop('weather_wind_min',axis=1)
df_sensor_test = df[df['sensor_id'].isin(test_sensors)].drop('weather_wind_min',axis=1)

df_sensor_train_features = df_sensor_train.drop(['sensor_id', 'timestamp'],axis=1)
df_sensor_val_features = df_sensor_val.drop(['sensor_id', 'timestamp'],axis=1)
df_sensor_test_features = df_sensor_test.drop(['sensor_id', 'timestamp'],axis=1)

df_sensor_train.head()

#Normalize data
# Initialize a scaler using the training data.
scaler = StandardScaler().fit(df_sensor_train_features)

#Normalize the data
df_sensor_train_features_scaled = scaler.transform(df_sensor_train_features)
df_sensor_val_features_scaled = scaler.transform(df_sensor_val_features)
df_sensor_test_features_scaled = scaler.transform(df_sensor_test_features)

#Test transform
print('colwise mean', np.mean(df_sensor_train_features_scaled, axis=0).round(6))
print('colwise variance', np.var(df_sensor_train_features_scaled, axis=0))

cols = df_sensor_train_features.columns

#put the 
df_sensor_train[cols] = df_sensor_train_features_scaled
df_sensor_val[cols] = df_sensor_val_features_scaled
df_sensor_test[cols] = df_sensor_test_features_scaled

df_sensor_train.head()

#Make overlapping windows of length selected above
def temporalize(X, lookback):
    '''
    Inputs
    X         A 2D numpy array ordered by time of shape: 
              (n_observations x n_features)
    lookback  The window size to look back in the past 
              records. Shape: a scalar.

    Output
    output_X  A 3D numpy array of shape: 
              ((n_observations-lookback-1) x lookback x 
              n_features)
    '''
    output_X = []
    for i in range(len(X) - lookback + 1):
        t = []
        for j in range(0, lookback):
            # Gather the past records upto the lookback period
            t.append(X[[(i + j)], :])
        output_X.append(t)
    return np.squeeze(np.array(output_X))

window_length = 48
data_train_sequence = []
data_val_sequence = []
data_test_sequence = []
data_train_windowed = []
data_val_windowed = []
data_test_windowed = []

#Run temporalize on each sensor and concatenate the results along rows (axis 0)
for id in pd.unique(df_sensor_train["sensor_id"]):
  tmp = df_sensor_train[df_sensor_train['sensor_id'] == id].to_numpy()
  data_train_sequence.append(tmp)
  
  tmp = temporalize(tmp, window_length)
  data_train_windowed.append(tmp)

data_train_windowed = np.concatenate(data_train_windowed)

#Run temporalize on each sensor and concatenate the results along rows (axis 0)
for id in pd.unique(df_sensor_val["sensor_id"]):
  tmp = df_sensor_val[df_sensor_val['sensor_id'] == id].to_numpy()
  data_val_sequence.append(tmp)

  tmp = temporalize(tmp, window_length)
  data_val_windowed.append(tmp)

data_val_windowed = np.concatenate(data_val_windowed)

#Run temporalize on each sensor and concatenate the results along rows (axis 0)
for id in pd.unique(df_sensor_test["sensor_id"]):
  tmp = df_sensor_test[df_sensor_test['sensor_id'] == id].to_numpy()
  data_test_sequence.append(tmp)

  tmp = temporalize(tmp, window_length)
  data_test_windowed.append(tmp)

data_test_windowed = np.concatenate(data_test_windowed)

print(len(data_train_sequence))
print(len(data_val_sequence))
print(len(data_test_sequence))
print(data_train_sequence[0].shape)
print(data_val_sequence[0].shape)
print(data_test_sequence[0].shape)
print(data_train_windowed.shape)
print(data_val_windowed.shape)
print(data_test_windowed.shape)

#create dataset
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        # Return the size of the dataset
        return len(self.inputs)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        X = X[:,2:]
        X = torch.tensor(np.array(list(X[:,:]), dtype=np.float)).float()
        
        return X, X[:,[0,1,3]]

    def getTimestamps(self):
        return self.inputs[:,:,1]

    def getSensors(self):
        return self.inputs[:,:,0]

#Create LSTM dataset
rnnTrainData = Dataset(data_train_sequence)
rnnValData = Dataset(data_val_sequence)
rnnTestData = Dataset(data_test_sequence)

#Create Windowed dataset
windowedTrainData = Dataset(data_train_windowed)
windowedValData = Dataset(data_val_windowed)
windowedTestData = Dataset(data_test_windowed)

x, y = rnnTrainData[0]
print(x.shape)
print(y.shape)
x, y = windowedTrainData[0]
print(x.shape)
print(y.shape)

batch_size = 32
num_epochs = 100
num_workers = 0
dataloader_train_seq = data.DataLoader(
  dataset=rnnTrainData,
  batch_size=1,
  shuffle=True,
  num_workers=num_workers
)
dataloader_val_seq = data.DataLoader(
  dataset=rnnValData,
  batch_size=1,
  shuffle=False,
  num_workers=num_workers
)
dataloader_test_seq = data.DataLoader(
  dataset=rnnTestData,
  batch_size=1,
  shuffle=False,
  num_workers=num_workers
)
dataloader_train = data.DataLoader(
  dataset=windowedTrainData,
  batch_size=batch_size,
  shuffle=True,
  num_workers=num_workers
)
dataloader_val = data.DataLoader(
  dataset=windowedValData,
  batch_size=batch_size,
  shuffle=False,
  num_workers=num_workers
)
dataloader_test = data.DataLoader(
  dataset=windowedTestData,
  batch_size=batch_size,
  shuffle=False,
  num_workers=num_workers
)
class LSTMEncoder(nn.Module):

  def __init__(self, n_features, embedding_dim,drop):
    super(LSTMEncoder, self).__init__()

    self.n_features = n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=self.n_features,
      hidden_size=self.embedding_dim,
      num_layers=1,
      batch_first=True
    )
    
    #self.rnn2 = nn.LSTM(
    #  input_size=self.hidden_dim,
    #  hidden_size=self.embedding_dim,
    #  num_layers=1,
    #  batch_first=True
    #)

    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(drop)
  def forward(self, x):

    x, (embedding_n, _) = self.rnn1(x)
    #x = self.activation(x)
    #x, (embedding_n, _) = self.rnn2(x)
    out = self.activation(embedding_n)
    out = self.dropout(out)

    return out.squeeze() #batchxembedding_dim

class LSTMDecoder(nn.Module):

  def __init__(self, embedding_dim, n_features,drop):
    super(LSTMDecoder, self).__init__()

    self.embedding_dim = embedding_dim
    self.hidden_dim, self.n_features = 2 * embedding_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size= self.embedding_dim,
      hidden_size= self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    #self.rnn2 = nn.LSTM(
    #  input_size= self.hidden_dim,
    #  hidden_size=self.hidden_dim,
    #  num_layers=1,
    #  batch_first=True
    #)

    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(drop)

    self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

  def forward(self, x, seq_len):
    
    x = x.unsqueeze(1).repeat(1, seq_len, 1) #Add a dimension of length 1 in front #batchXseq_lenXembedding_dim

    x, (_, _) = self.rnn1(x)
    x = self.activation(x)
    x = self.dropout(x)
    #x, (_, _) = self.rnn2(x) 

    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):

  def __init__(self, n_features, out_features, embedding_dim,drop):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = LSTMEncoder(n_features, embedding_dim,drop).to(device)
    self.decoder = LSTMDecoder(embedding_dim, out_features,drop).to(device)

  def forward(self, x):
    seq_len = x.size(1)
    x = self.encoder(x)
    x = self.decoder(x, seq_len)

    return x


x, y = windowedTrainData[0]
seq_len, n_features = x.size()
seq_len, out_features = y.size()
modelLSTM = RecurrentAutoencoder(n_features, out_features, embedding_dim= 128,drop=0.1)
modelLSTM = modelLSTM.to(device)
modelLSTM

def train_modelLSTM(params,save_model=False):
  n_epochs = 100
  modelLSTM = RecurrentAutoencoder(n_features,
                                   out_features, 
                                   embedding_dim= 128,
                                   drop = params["drop"]
                                   )
  modelLSTM = modelLSTM.to(device)
  optimizer = torch.optim.Adam(modelLSTM.parameters(), lr=params["learning_rate"])
  criterion = nn.L1Loss(reduction='mean').to(device)
  criterion_unreduced = nn.L1Loss(reduction='none').to(device)
  history = dict(train=[], val=[], train_best=[], val_best=[])

  best_model_wts = copy.deepcopy(modelLSTM.state_dict())
  best_loss = 10000.0
  early_stopping_counter = 0
  early_stopping = 20
  for epoch in range(1, n_epochs + 1):
    #print(epoch)
    modelLSTM = modelLSTM.train()

    train_losses = []
    train_losses_unreduced = []
    for inputs, seq_true in dataloader_train:
      optimizer.zero_grad()

      inputs = inputs.to(device)
      seq_true = seq_true.to(device)
      seq_pred = modelLSTM(inputs)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      #torch.nn.utils.clip_grad_norm_(modelLSTM.parameters(),10)
      optimizer.step()

      train_losses.append(loss.item())

      loss_unreduced = criterion_unreduced(seq_pred, seq_true)
      train_losses_unreduced.append(loss_unreduced)

    val_losses = []
    val_losses_unreduced = []
    modelLSTM = modelLSTM.eval()
    with torch.no_grad():
      for inputs, seq_true in dataloader_val:

        inputs = inputs.to(device)
        seq_true = seq_true.to(device)
        seq_pred = modelLSTM(inputs)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

        loss_unreduced = criterion_unreduced(seq_pred, seq_true)
        val_losses_unreduced.append(loss_unreduced)

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)
    
    
    if val_loss < best_loss:
      early_stopping_counter = 0
      best_loss = val_loss
      if save_model:
          torch.save(modelLSTM.state_dict(),"model_lstm.bin")
    else:
          early_stopping_counter += 1
         # print(f'Early stop counter: {early_stopping_counter}')
    #print(val_loss)
    #print(train_loss)
    if early_stopping_counter > early_stopping:
           break
    #print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')  
  return best_loss


#params = {"drop": 0.1,
#          "learning_rate":0.001}  
#train_modelLSTM(params)

def objective(trial):
    params = {
        "drop":trial.suggest_uniform("drop",0,0.3),
        "learning_rate": trial.suggest_loguniform("learning_rate",1e-6,1e-3)
        }
    
    loss = train_modelLSTM(params)
    return loss

import csv
study = optuna.create_study(direction="minimize")
study.optimize(objective,n_trials = 20)
trial_ = study.best_trial
#print(f"BayesOpt for LSTM is done. Best parameters was: {trial_.params}")
# Save model and parameters
train_modelLSTM(trial_.params,save_model=True)

with open('Best_model_hyperparameters_LSTM.csv', 'w') as f:  
    w = csv.DictWriter(f, trial_.params.keys())
    w.writeheader()
    w.writerow(trial_.params)