
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
#df = pd.read_csv("C:/Users/kein/Documents/PhD/Kurser/Deep learning/Deep learning project/To_remote/woodsense-sensor-data-2020-11-30-cleaned.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""# Preprocessing

Seems around 30-50 lags is a good start. We often find few significant lags larger than 30 but it does happen (we would expect about 5 false positives) and some sensors shows signs of seasonality with period about 20. this makes sense as we could expect it to vary along the day so a seasonality of 24 makes sense. Thus making the length a multiple of 24 would make sense. Therefore we decide on a window length 48
"""

#Map time of day and day of year to features
#df['tod_sin'] = np.sin(df['timestamp'].dt.hour / 24 * 2 * np.pi)
#df['tod_cos'] = np.cos(df['timestamp'].dt.hour / 24 * 2 * np.pi)
#df['doy_sin'] = np.sin(df['timestamp'].dt.dayofyear / 365 * 2 * np.pi)
#df['doy_cos'] = np.cos(df['timestamp'].dt.dayofyear / 365 * 2 * np.pi)

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
num_epochs = 150
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

class CNNEncoder(nn.Module):
  def __init__(self, num_features, seq_len, hidden_channels, embedding_dim,drop1):
    super(CNNEncoder, self).__init__()

    self.seq_len, self.num_features = seq_len, num_features

    self.encoder = nn.Sequential(
        nn.Conv1d(in_channels=num_features, out_channels=hidden_channels, kernel_size=7, padding=3, stride=2),
        #nn.BatchNorm1d(hidden_channels),
        nn.ReLU(),
        nn.Dropout(drop1),
        nn.Conv1d(in_channels=hidden_channels, out_channels=embedding_dim, kernel_size=7, padding=3, stride=2)
    )


  def forward(self, x):

    x = x.permute(0, 2, 1) #Change to batch X features X seq_len

    x = self.encoder(x)

    return x

class CNNDecoder(nn.Module):
  def __init__(self, num_features, seq_len, hidden_channels, embedding_dim,drop1):
    super(CNNDecoder, self).__init__()

    self.seq_len, self.num_features = seq_len, num_features

    self.decoder = nn.Sequential(
        nn.ConvTranspose1d(in_channels=embedding_dim, out_channels=hidden_channels, kernel_size=7, padding=3, stride=2, output_padding=1),
        #nn.BatchNorm1d(hidden_channels),
        nn.ReLU(),
        nn.Dropout(drop1),
        nn.ConvTranspose1d(in_channels=hidden_channels, out_channels=num_features, kernel_size=7, padding=3, stride=2, output_padding=1)
    )


  def forward(self, x):

    x = self.decoder(x)

    x = x.permute(0, 2, 1) #Change to batch X seq_len X features

    return x


class CNNAutoEncoder(nn.Module):
    def __init__(self, num_features, out_features, seq_len, hidden_channels, embedding_dim,drop1):
        super(CNNAutoEncoder, self).__init__()

        self.encoder = CNNEncoder( num_features, seq_len, hidden_channels, embedding_dim,drop1).to(device)
        self.decoder = CNNDecoder( out_features, seq_len, hidden_channels, embedding_dim,drop1).to(device)

    def forward(self, x): 
        outputs = {}
        # we don't apply an activation to the bottleneck layer
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x

x, y = windowedTrainData[0]
seq_len, n_features = x.size()
seq_len, out_features = y.size()
modelCNN = CNNAutoEncoder(num_features=n_features, out_features=out_features, seq_len=seq_len, hidden_channels=12, embedding_dim=11,drop1 = 0)
modelCNN = modelCNN.to(device)
modelCNN

def train_modelCNN(params,save_model=True):
  n_epochs = 100
  modelCNN = CNNAutoEncoder(num_features=n_features,
                            out_features=out_features,
                            seq_len=seq_len, 
                            hidden_channels=params["channel1"],
                            embedding_dim=11,
                            drop1=params["drop"])
  modelCNN = modelCNN.to(device)

  optimizer = torch.optim.Adam(modelCNN.parameters(), lr=params["learning_rate"])
  criterion = nn.L1Loss(reduction='mean').to(device)
  criterion_unreduced = nn.L1Loss(reduction='none').to(device)
  history = dict(train=[], val=[], train_best=[], val_best=[])

  best_model_wts = copy.deepcopy(modelCNN.state_dict())
  best_loss = 10000.0
  early_stopping_counter = 0
  early_stopping = 20
  for epoch in range(1, n_epochs + 1):
    modelCNN = modelCNN.train()

    train_losses = []
    train_losses_unreduced = []
    for inputs, seq_true in dataloader_train:
      optimizer.zero_grad()

      inputs = inputs.to(device)
      seq_true = seq_true.to(device)
      seq_pred = modelCNN(inputs)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

      loss_unreduced = criterion_unreduced(seq_pred, seq_true)
      train_losses_unreduced.append(loss_unreduced)

    val_losses = []
    val_losses_unreduced = []
    modelCNN = modelCNN.eval()
    with torch.no_grad():
      for inputs, seq_true in dataloader_val:

        inputs = inputs.to(device)
        seq_true = seq_true.to(device)
        seq_pred = modelCNN(inputs)

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
          torch.save(modelCNN.state_dict(),"model_cnn.bin")
    else:
          early_stopping_counter += 1

      
    if early_stopping_counter > early_stopping:
           break
    

  return best_loss

def objective(trial):
    params = {
        "channel1": trial.suggest_int("channel1",20,40),
        "drop":trial.suggest_uniform("drop",0,0.3),
        "learning_rate": trial.suggest_loguniform("learning_rate",1e-6,1e-3)
        }
    
    loss = train_modelCNN(params)
    return loss

import csv
study = optuna.create_study(direction="minimize")
study.optimize(objective,n_trials = 20)
trial_ = study.best_trial
#print(f"BayesOpt for CNN is done. Best parameters was: {trial_.params}")
# Save model and parameters
train_modelCNN(trial_.params,save_model=True)


with open('Best_model_hyperparameters_CNN.csv', 'w') as f:  
    w = csv.DictWriter(f, trial_.params.keys())
    w.writeheader()
    w.writerow(trial_.params)